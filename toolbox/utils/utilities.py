from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
import isx
import json
import logging
from numpy import ndarray
import numpy as np
import pandas as pd
import os
import struct
from toolbox.utils.exceptions import IdeasError
from typing import List

logger = logging.getLogger()


def read_isxc_metadata(input_filename):
    """Read the metadata of an isxc file as a json-formatted dictionary.

    :param input_filename: path to the input file (.isxc)
    :return: metadata of the isxc file
    """
    _, file_extension = os.path.splitext(input_filename)
    if file_extension.lower() != ".isxc":
        raise IdeasError("Metadata can only be extracted from isxc files")

    try:
        with open(input_filename, "rb") as f:
            # the number of bytes used to represent the variable size_t in c++
            sizeof_size_t = 8

            # location of the frame count in the header (8th field)
            frame_count_location = sizeof_size_t * 7

            # need the size of data descriptors to get to the session offset
            isx_comp_desc_offset = 32

            # multiplied by 2 since there are 2 descriptors (frame and meta data)
            session_offset_location = (
                frame_count_location + isx_comp_desc_offset * 2
            )

            # extract the session offset from the header to get location of the session data
            f.seek(session_offset_location, 0)
            session_offset = int.from_bytes(
                f.read(sizeof_size_t), byteorder="little"
            )

            # move reader to the session data
            f.seek(session_offset, 0)

            # read the session data and decode the string
            session = str(f.read().decode("utf-8"))

            # convert the string into a json file format
            session_json = json.loads(session)
            return session_json
    except Exception:
        raise IdeasError(
            "The isxc file metadata cannot be read, it may be missing or corrupted."
        )


def read_isxd_metadata(input_filename):
    """Read the metadata of an isxd file as a json-formatted dictionary.

    :param input_filename: path to the input file (.isxd)
    :return: metadata of the isxd file
    """
    _, file_extension = os.path.splitext(input_filename)
    if file_extension.lower() != ".isxd":
        raise IdeasError("Metadata can only be extracted from isxd files")

    try:
        with open(input_filename, "rb") as f:
            footer_size_offset = 8
            f.seek(-footer_size_offset, 2)
            footer_size = int.from_bytes(
                f.read(footer_size_offset), byteorder="little"
            )
            offset = footer_size + footer_size_offset + 1
            f.seek(-offset, 2)
            metadata = json.loads(f.read(footer_size))
            return metadata
    except Exception:
        # error message from IDPS: "Error while seeking to beginning of JSON header at end"
        raise IdeasError(
            "The isxd file metadata cannot be read, it may be missing or corrupted. "
            "File recovery may help recover the data and save it into a new isxd file."
        )


def write_isxd_metadata(
    filename, json_metadata, sizeof_size_t=8, endianness="little"
):
    """
    Writes json metadata to an .isxd file

    Arguments
    ---------
    filename : str
        The .isxd filename
    json_metadata : dict
        Metadata represented as a json dictionary
    sizeof_size_t : int > 0
        Number of bytes used to represent a size_t type variable in C++
    endianness
        Endianness of your machine
    """

    with open(filename, "rb+") as infile:
        infile.seek(-sizeof_size_t, 2)
        header_size = infile.read(sizeof_size_t)
        header_size = int.from_bytes(header_size, endianness)
        bottom_offset = header_size + 1 + sizeof_size_t
        infile.seek(-bottom_offset, 2)

        infile.truncate()
        # process non-ascii characters correctly
        string_json = (
            json.dumps(json_metadata, indent=4, ensure_ascii=False) + "\0"
        )
        infile.write(bytes(string_json, "utf-8"))

        # calculate number of bytes in string by encoding to utf-8
        string_json = string_json.encode("utf-8")
        json_length = int.to_bytes(
            len(string_json) - 1, sizeof_size_t, endianness
        )
        infile.write(json_length)


def copy_isxd_extra_properties(
    input_isxd_files: List[str],
    original_input_indices: List[int],
    outputs_isxd_files: List[List[str]],
):
    """
    Copy extra properties from input isxd files to output isxd files.
    This can be important for downstream tools like "Map Annotations to ISXD Data"
    which rely on certain keys in the extra properties for aligning miniscope data
    to synchronized nVision data. IDPS does this within algorithms however since isxd
    files are being generated outside of IDPS, the extra properties need to be manually copied.
    """

    num_files = len(input_isxd_files)
    for output_isxd_files in outputs_isxd_files:
        assert num_files == len(output_isxd_files)

    for i in range(num_files):
        input_index = original_input_indices[i]
        input_isxd_metadata = read_isxd_metadata(input_isxd_files[input_index])

        for output_isxd_files in outputs_isxd_files:
            output_isxd_metadata = read_isxd_metadata(output_isxd_files[i])
            output_isxd_metadata["extraProperties"] = input_isxd_metadata[
                "extraProperties"
            ]
            write_isxd_metadata(output_isxd_files[i], output_isxd_metadata)


def compute_sampling_rate(period_num: int, period_den: int) -> float:
    """Compute the sampling rate given the period numerator and denominator.

    :param period_num: numerator in the period
    :param period_den: denominator in the period
    :return: the sampling rate or None if there is a division by zero error.
    """
    try:
        return np.round(1 / (period_num / period_den), 2)
    except ZeroDivisionError:
        return None


def compute_end_time(timing_info):
    """Compute end time of an isxd file.
    :param isxd_timing_info: json object containing isxd timing information
    :return: end time json object of the form
             {
                "secsSinceEpoch": {"num": 123456, "den": 1000}
                "utcOffset": 0
             }
    """
    # extract start time and period contained in the timing info object
    start_time = (
        timing_info["start"]["secsSinceEpoch"]["num"]
        / timing_info["start"]["secsSinceEpoch"]["den"]
    )
    period = timing_info["period"]["num"] / timing_info["period"]["den"]

    # compute end time
    end_time = start_time + (period * timing_info["numTimes"])
    end_time_den = timing_info["start"]["secsSinceEpoch"]["den"]
    end_time_num = round(end_time * end_time_den)

    # return properly formatted end time
    return {
        "secsSinceEpoch": {"den": end_time_den, "num": end_time_num},
        "utcOffset": timing_info["start"]["utcOffset"],
    }


def sort_isxd_files_by_start_time(input_files):
    """Sort isxd files by their start time.
    :param input_files: list of isxd file paths
    :return: sorted list of isxd file paths
    """
    start_times = []
    for f in input_files:
        isxd_metadata = read_isxd_metadata(f)

        if isxd_metadata["type"] == 5:
            # isxd events
            start_time = (
                isxd_metadata["global times"][0]["secsSinceEpoch"]["num"]
                / isxd_metadata["global times"][0]["secsSinceEpoch"]["den"]
            )
        else:
            # other isxd files
            start_time = (
                isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["num"]
                / isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["den"]
            )
        start_times.append(start_time)

    sorted_indices = np.argsort(start_times)
    sorted_files = np.array(input_files)[sorted_indices]
    return sorted_files.tolist()


def get_num_cells_by_status(cellset_filename: str):
    """Count the number of cells for each cell status"""
    cell_set = isx.CellSet.read(cellset_filename)

    num_accepted_cells = 0
    num_undecided_cells = 0
    num_rejected_cells = 0

    for index in range(cell_set.num_cells):
        cell_status = cell_set.get_cell_status(index)
        if cell_status == "accepted":
            num_accepted_cells += 1
        elif cell_status == "undecided":
            num_undecided_cells += 1
        elif cell_status == "rejected":
            num_rejected_cells += 1

    return num_accepted_cells, num_undecided_cells, num_rejected_cells


def get_file_size(file_path: str) -> str:
    """Compute and format the size of a file on disk.

    :param file_path: path to the input file
    """
    file_size = abs(os.path.getsize(file_path))
    for unit_prefix in ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]:
        if file_size < 1024.0:
            return f"{file_size:.2f} {unit_prefix}B"
        file_size /= 1024.0
    return f"{os.path.getsize(file_path)} B"


def check_file_exists(file: str) -> None:
    """check if a file exists, fail if not"""

    if not os.path.exists(file):
        raise Exception(f"{file} does not exist.")


def check_file_extention_is(
    file_name: str,
    *,
    ext: str = ".isxd",
):
    """small util func to check the extention of a file
    and fail otherwise"""

    _, ext_ = os.path.splitext(os.path.basename(file_name))
    if ext_.lower() != ext:
        raise Exception(
            f"{file_name} does not have the extension: {ext}. It instead has {ext_}"
        )


def _footer_length(isxd_file: str) -> int:
    """find the length of the footer in bytes"""

    with open(isxd_file, mode="rb") as file:
        file.seek(-8, os.SEEK_END)
        data = file.read()
    footer_length = struct.unpack("ii", data)[0]

    return footer_length


def _extract_footer(isxd_file: str) -> dict:
    """extract movie footer from ISXD file"""

    footer_length = _footer_length(isxd_file)

    with open(isxd_file, mode="rb") as file:
        file.seek(-8 - footer_length - 1, os.SEEK_END)
        data = file.read(footer_length)

    footer = data.decode("utf-8")
    return json.loads(footer)


def _sort_isxd_files_by_start_time(
    input_files: List[str],
) -> List:
    """Sort isxd files by their start time.
    :param input_files: list of isxd file paths
    :return: sorted list of isxd file paths
    """
    start_times = []
    for file in input_files:
        isxd_metadata = _extract_footer(file)

        if isxd_metadata["type"] == 5:
            # isxd events
            start_time = (
                isxd_metadata["global times"][0]["secsSinceEpoch"]["num"]
                / isxd_metadata["global times"][0]["secsSinceEpoch"]["den"]
            )
        else:
            # other isxd files
            start_time = (
                isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["num"]
                / isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["den"]
            )
        start_times.append(start_time)

    sorted_indices = np.argsort(start_times)
    sorted_files = np.array(input_files)[sorted_indices]
    return sorted_files.tolist()


def movie_series(files: List[str]) -> List[str]:
    """function validates a list of ISXD movies
    and returns a re-ordered list if they can form a valid
    series. throws an error otherwise.

    A list of movies is a valid series if:

    - frame sizes are all the same
    - each movie has a unique start time
    - all movies have the same frame rate

    """
    if len(files) == 0:
        return []

    if len(files) == 1:
        return files

    start_times = np.zeros(len(files))
    pixel_shapes = [None] * len(files)
    periods = np.zeros(len(files))

    for i, file in enumerate(files):
        check_file_exists(file)
        check_file_extention_is(file, ext=".isxd")

        # ensure it consists of an isxd MOVIE
        movie = isx.Movie.read(file)

        # read the metadata and ensure that all the pixel shapes are the same
        pixel_shapes[i] = movie.spacing.num_pixels

        # read start time of the movie
        start_times[i] = movie.timing.start._to_secs_since_epoch().secs_float

        # check that frame rates are the same
        periods[i] = movie.timing.period.secs_float

    for i in range(len(files)):
        if not np.isclose(periods[0], periods[i], atol=1e-6):
            raise Exception(
                f"""[INVALID SERIES] The input files do 
            not form a valid movie series. 
            Frame rates are different across these files.
            Differing frame rates are: {periods[0]} which
            is not the same as {periods[i]}.
            """,
            )

    if pixel_shapes.count(pixel_shapes[0]) != len(pixel_shapes):
        raise Exception(
            """[INVALID SERIES] The input files do 
            not form a valid movie series.
            The pixel sizes of the files provided do
            not match.""",
        )

    if len(np.unique(start_times)) != len(start_times):
        raise Exception(
            """[INVALID SERIES] Files in the series
             do not have unique start times"""
        )

    return _sort_isxd_files_by_start_time(files)


def cell_set_series(files: List[str]) -> List:
    """Validate isxd file paths for existence and cell set format.

    :param isxd_cellset_files: list of paths to the
    input isxd cell set files
    :return: list of paths to the input isxd cell set
    files ordered by start time.
    """
    if len(files) == 0:
        return []

    if len(files) == 1:
        return files

    for file in files:
        check_file_exists(file)

        check_file_extention_is(file, ext=".isxd")

        # ensure it consists of an isxd CELL SET
        metadata = _extract_footer(file)
        isxd_type = metadata["type"]
        if isxd_type != 1:
            raise Exception(f"{file} is not a ISXD cell set file")

    start_times = np.zeros(len(files))
    cell_lists = [None] * len(files)

    # to be a valid series, all cell sets must have the
    # same status
    cellset = isx.CellSet.read(files[0])
    num_cells = cellset.num_cells
    status0 = [cellset.get_cell_status(i) for i in range(num_cells)]
    periods = np.zeros(len(files))

    for i, file in enumerate(files):
        metadata = _extract_footer(file)

        cellset = isx.CellSet.read(file)
        num_cells = cellset.num_cells
        status = [cellset.get_cell_status(i) for i in range(num_cells)]

        if status != status0:
            raise Exception(
                f"""[INVALID SERIES] {file} and {files[0]} 
    cannot be part of a valid series because they have 
    different cell statuses"""
            )

        cell_lists[i] = metadata["CellNames"]

        start_times[i] = (
            metadata["timingInfo"]["start"]["secsSinceEpoch"]["num"]
            / metadata["timingInfo"]["start"]["secsSinceEpoch"]["den"]
        )

        # check that frame rates are the same
        periods[i] = (
            metadata["timingInfo"]["period"]["num"]
            / metadata["timingInfo"]["period"]["den"]
        )

    for i in range(len(files)):
        if not np.isclose(periods[0], periods[i], atol=1e-6):
            raise Exception(
                f"""[INVALID SERIES] The input files do 
            not form a valid cell set series. 
            Frame rates are different across these files.
            Differing frame rates are: {periods[0]} which
            is not the same as {periods[i]}.
            """,
            )

    # ensure all cell sets contain the same cells
    if cell_lists.count(cell_lists[0]) != len(cell_lists):
        raise Exception(
            """[INVALID SERIES]
            The input files do not form a series. 
            The cell set files do not describe 
            the same cells.""",
        )

    return _sort_isxd_files_by_start_time(files)


@beartype
def validate_caiman_msr_inputs(
    *,
    cellset_paths: List[str],
    template_paths: List[str],
    eventset_paths: Optional[List[str]],
    enclosed_thr: Optional[Union[float, int]],
    use_cell_status: str,
    min_n_regist_sess: int,
) -> Tuple[List[Optional[str]], Optional[Union[float, int]], int]:
    """
    Validate CaImAn Multi-Session Registration input files and parameters.
    """
    # validate input files
    # # ensure there are at least 2 cellsets
    assert len(cellset_paths) > 1, (
        "Only 1 cellset was provided. Please provide at least 2 cellsets and "
        "corresponding template images as input of this tool."
    )

    # # ensure there are as many template images as there are cellsets
    assert len(cellset_paths) == len(template_paths), (
        f"There were {len(cellset_paths)} cellsets and "
        f"{len(template_paths)} template images. Please provide as many "
        "template images as cellsets."
    )

    # # ensure there are as many eventsets, if provided, as there are cellsets
    if (
        isinstance(eventset_paths, list)
        and isinstance(eventset_paths[0], str)
        and len(eventset_paths[0]) > 0
    ):
        assert len(cellset_paths) == len(eventset_paths), (
            f"There were {len(cellset_paths)} cellsets and "
            f"{len(eventset_paths)} eventsets. Please provide as many "
            "eventsets as cellsets."
        )
    else:
        eventset_paths = [None] * len(cellset_paths)

    # validate parameters
    # 1. ensure type of `enclosed_thr`
    if enclosed_thr == 0:
        enclosed_thr = None

    # 2. ensure correct value of `use_cell_status``
    valid_statuses = ["accepted", "accepted & undecided", "all"]
    assert use_cell_status in valid_statuses, (
        f"`use_cell_status` {use_cell_status} not recognized."
        f"It should be one of {valid_statuses}."
    )

    # 3. log a warning if `min_n_regist_sess` is above the number of
    # provided cellsets
    if min_n_regist_sess > len(cellset_paths):
        logger.warning(
            f"`min_n_regist_sess` ({min_n_regist_sess}) > number of provided "
            f"cellsets ({len(cellset_paths)}). Setting `min_n_regist_sess` "
            f"to {len(cellset_paths)} instead."
        )
        min_n_regist_sess = len(cellset_paths)

    return eventset_paths, enclosed_thr, min_n_regist_sess


@beartype
def transform_assignments_matrix(
    *,
    assignments: ndarray,
    min_n_regist_sess: int,
    cell_names_list: List[List[str]],
) -> Tuple[pd.DataFrame, int]:
    """
    Transform and filter assignments matrix to add homogeneous cell names
    across sessions and to only keep cells registered across at least
    `min_n_regist_sess` sessions.
    """
    n_sessions = assignments.shape[1]

    # transform the assignments matrix into a pandas DataFrame
    # for easier transformations
    df_assignments = pd.DataFrame(data=assignments)

    # count number of registered sessions for each cell
    df_assignments["n_reg_sess"] = df_assignments.notna().sum(axis=1)

    # get position of the first NaN value for each row,
    # to enable advanced sorting
    df_first_nan = pd.DataFrame(
        np.where(df_assignments.isna()), index=["idx_row", "nan_loc"]
    ).T.drop_duplicates(subset="idx_row")
    df_assignments.loc[df_first_nan["idx_row"].values, "first_NaN_pos"] = (
        df_first_nan["nan_loc"].values
    )

    # sort df by decreasing order of both number of registered sessions
    # and position of first NaN
    df_assignments = df_assignments.sort_values(
        by=["n_reg_sess", "first_NaN_pos"], ascending=False
    )

    # get number of cells registered across all sessions
    n_reg_cells_all = len(df_assignments.query(f"n_reg_sess == {n_sessions}"))

    # add registered cell names
    n_digits = len(str(assignments.shape[0]))
    df_assignments.insert(
        loc=0,
        column="name",
        value=[f"C{idx:0{n_digits}g}" for idx in range(assignments.shape[0])],
    )

    # only keep cells registered across at least `min_n_regist_sess` sessions
    df_assignments = df_assignments.query(f"n_reg_sess >= {min_n_regist_sess}")

    # add input cellsets' corresponding cell names
    for idx_sess, cell_names in enumerate(cell_names_list):
        col_name = f"name_in_{idx_sess}"
        df_assignments.insert(df_assignments.shape[1], col_name, "")
        cell_indices = df_assignments[idx_sess].dropna().astype(int)
        df_assignments.loc[cell_indices.index, col_name] = [
            cell_names[x] for x in cell_indices.values
        ]

    # clean up df
    df_assignments = df_assignments.drop(
        columns=["first_NaN_pos"]
    ).reset_index(drop=True)

    return df_assignments, n_reg_cells_all
