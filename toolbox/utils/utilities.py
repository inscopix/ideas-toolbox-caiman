import os
import json
import isx
import struct
import numpy as np
from typing import List
from toolbox.utils.exceptions import IdeasError
import logging

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
        metadata = _extract_footer(file)
        if metadata["type"] != 0:
            raise Exception(f"{file} is not a ISXD movie")

        # read the metadata and ensure that all the pixel shapes are the same
        pixel_shapes[i] = [
            metadata["spacingInfo"]["numPixels"]["x"],
            metadata["spacingInfo"]["numPixels"]["y"],
        ]

        # read start time of the movie
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
