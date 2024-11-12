import os
import json
import isx
from typing import List
from toolbox.utils.utilities import (
    compute_sampling_rate,
    compute_end_time,
)
from toolbox.utils.utilities import (
    get_num_cells_by_status,
    read_isxc_metadata,
    read_isxd_metadata,
)
from toolbox.utils.exceptions import IdeasError
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import logging

logger = logging.getLogger()


def is_multicolor(metadata, check_interleaved=True):
    """Check if an isxd file is from a multicolor recording

    :param metadata: dictionary containing metadata of isxd file
    :param check_interleaved: if False, only checks if a file originates from
    a multicolor movie and does not check if it has been deinterleaved. If True,
    an additional check is done to ensure that the movie has not been deinterleaved.

    :return: if check_deinterleaved=False: True for multicolor movies and any files derived
    from multicolor movies (e.g. processed movies, cell sets, event set, etc.)
    if check_deinterleaved=True: True only for multicolor movies that have not been deinterleaved
    """
    if (
        "extraProperties" not in metadata
        or metadata["extraProperties"] is None
    ):
        logger.warn(
            "Unable to determine whether the input is a multicolor movie"
        )
        return False

    # return False if we cannot determine if the file is multicolor
    if metadata["extraProperties"] is None:
        return False

    try:
        # check if the movie has the metadata originating from a multicolor movie
        # this does not determine if a movie originates from a multicolor movie
        # but has been deinterleaved
        has_multicolor_metadata = (
            "dualColor" in metadata["extraProperties"]["microscope"]
            and "enabled"
            in metadata["extraProperties"]["microscope"]["dualColor"]
            and metadata["extraProperties"]["microscope"]["dualColor"]["mode"]
            == "multiplexing"
        )

        # check if file has been deinterleaved
        # IDPS adds metadata fields once a movie is deinterleaved, so we can check for that
        if has_multicolor_metadata and check_interleaved:
            is_deinterleaved = (
                "idps" in metadata["extraProperties"]
                and "channel" in metadata["extraProperties"]["idps"]
            )
            return not is_deinterleaved
        return has_multicolor_metadata
    except KeyError:
        return False


def is_multiplane(metadata, check_interleaved=True):
    """Check if an isxd file is from a multiplane recording

    :param metadata: dictionary containing metadata of isxd file
    :param check_interleaved: if False, only checks if a file originates from
    a multiplane movie and does not check if it has been deinterleaved. If True,
    an additional check is done to ensure that the movie has not been deinterleaved.

    :return: if check_deinterleaved=False: True for multiplane movies and any files derived
    from multiplane movies (e.g. processed movies, cell sets, event set, etc.)
    if check_deinterleaved=True: True only for multiplane movies that have not been deinterleaved
    """
    if is_multicolor(metadata):
        return False

    # return False if we cannot determine if the file is multiplane
    if (
        "extraProperties" not in metadata
        or metadata["extraProperties"] is None
    ):
        logger.warn(
            "Unable to determine whether the input is a multiplane movie"
        )
        return False

    has_multiplane_metadata = False
    # check if metadata contains fields originating from multiplane movie
    # this does not determine if a movie originates from a multiplane movie
    # but has been deinterleaved
    try:
        for w in metadata["extraProperties"]["auto"]["waveforms"]:
            if "name" in w and "efocus" in w["name"]:
                has_multiplane_metadata = True
                break
    except KeyError:
        return False

    # check if movie has been deinterleaved
    # IDPS adds metadata fields once a movie is deinterleaved, so we can check for that
    if has_multiplane_metadata and check_interleaved:
        is_deinterleaved = (
            "idps" in metadata["extraProperties"]
            and "efocus" in metadata["extraProperties"]["idps"]
        )
        return not is_deinterleaved

    return has_multiplane_metadata


def get_multiplane_efocus_vals(metadata):
    """Retrieve efocus values from multiplane isxd file metadata

    :param metadata: dictionary containing metadata of multiplane isxd file

    :return: set containing unique efocus values
    """
    # only need unique efocus values
    efocus_vals = set()

    # using initial values from isxPreprocessMovie.cpp getEfocusFromMovieJsonHeader() in IDPS
    v_max = 100
    max_amplitude = 892
    try:
        # get efocus values
        for waveform in metadata["extraProperties"]["auto"]["waveforms"]:
            if (
                "name" in waveform
                and isinstance(waveform["name"], str)
                and "efocus" in waveform["name"]
            ):
                # IDAS > 1.2.1
                if len(waveform["data"]) == 1:
                    for vertices in waveform["data"][0]["vertices"]:
                        efocus_vals.add(vertices["y"])
                # for IDAS <= 1.2.0
                else:
                    # get efocus sensor configurations
                    if metadata["extraProperties"]["adMode"] == "auto":
                        for triggered in metadata["extraProperties"]["auto"][
                            "triggered"
                        ]:
                            if "efocus" in triggered["destination"]["device"]:
                                v_max = triggered["destination"]["vMax"]
                                max_amplitude = triggered["destination"][
                                    "maxAmplitude"
                                ]
                    # get efocus values
                    for data in waveform["data"]:
                        if isinstance(data, str):
                            # need to filter relevant fields. The first 16 bits are used
                            # as an identifier, so right shifting by 16 allows us to see
                            # the identifier only - needs to match 0x3200
                            if int(data, 16) >> 16 == 0x3200:
                                efocus = (
                                    (int(data, 16) & 0xFFFF)
                                    * v_max
                                    / max_amplitude
                                )
                                efocus_vals.add(int(efocus))
        return sorted(list(efocus_vals))
    except KeyError as ke:
        raise IdeasError(
            f"The file metadata is missing required fields: {ke.args}"
        )


def get_multicolor_efocus_vals(metadata):
    """Retrieve efocus values from multicolor isxd file

    :param metadata: dictionary containing metadata of multicolor isxd file

    :return: set containing unique efocus values
    """
    green_efocus = metadata["extraProperties"]["microscope"]["dualColor"][
        "single"
    ]["green"]["focus"]
    red_efocus = metadata["extraProperties"]["microscope"]["dualColor"][
        "single"
    ]["red"]["focus"]
    return [green_efocus, red_efocus]


def get_efocus(input_filename, tmp_dir="/tmp"):
    """Return the efocus value associated with the input ISXD/IMU/GPIO file.
    For multiplane and multicolor data, the list of efocus values will be returned.

    :param input_filename: path to the input file
    :return: efocus value
    """
    # parse input file name
    file_basename, file_extension = os.path.splitext(
        os.path.basename(input_filename)
    )
    file_extension = file_extension.lower()

    # extract file metadata
    if file_extension == ".gpio":
        isx.export_gpio_to_isxd(input_filename, tmp_dir)
        tmp_gpio_isxd_filename = os.path.join(
            tmp_dir, file_basename + "_gpio.isxd"
        )
        metadata = read_isxd_metadata(tmp_gpio_isxd_filename)
        os.remove(tmp_gpio_isxd_filename)
    elif file_extension == ".imu":
        isx.export_gpio_to_isxd(input_filename, tmp_dir)
        tmp_imu_isxd_filename = os.path.join(
            tmp_dir, file_basename + "_imu.isxd"
        )
        metadata = read_isxd_metadata(tmp_imu_isxd_filename)
        os.remove(tmp_imu_isxd_filename)
    elif file_extension == ".isxc":
        metadata = read_isxc_metadata(input_filename)
    elif file_extension == ".isxd":
        metadata = read_isxd_metadata(input_filename)
    else:
        logger.warning(
            f"The efocus value cannot be extracted from a file with file extension '{file_extension}'"
        )
        return None

    # get efocus value from IDPS metadata
    try:
        efocus = metadata["extraProperties"]["idps"]["efocus"]
        if efocus not in [None, ""]:
            return efocus
    except (KeyError, TypeError):
        pass

    # if movie has multiple planes, retrieve all efocus values
    try:
        if is_multiplane(metadata, check_interleaved=True):
            return get_multiplane_efocus_vals(metadata)
        elif is_multicolor(metadata, check_interleaved=True):
            return get_multicolor_efocus_vals(metadata)
    except Exception:
        logger.warning(
            "Could not retrieve multiplane/multicolor efocus values"
        )

    # get efocus value from IDAS metadata
    try:
        efocus = metadata["extraProperties"]["microscope"]["focus"]
        if efocus not in [None, ""]:
            return efocus
    except (KeyError, TypeError):
        pass

    return None


def generate_cell_set_metadata(
    cellset_filename: str,
    file_key: str,
    metadata: dict,
    efocus_vals: List[int],
):
    """Generate metadata for a single cell set file.

    :param cellset_filename: path to the input cell set
    :param file_key: key associated with the file for which metadata is constructed
    :param metadata: metadata dictionary
    :param efocus_vals: list of efocus values
    """
    cell_set = isx.CellSet.read(cellset_filename)
    cell_set_metadata = read_isxd_metadata(cellset_filename)

    (
        num_accepted_cells,
        num_undecided_cells,
        num_rejected_cells,
    ) = get_num_cells_by_status(cellset_filename)

    output_metadata = {
        "metrics": {
            "num_accepted_cells": num_accepted_cells,
            "num_undecided_cells": num_undecided_cells,
            "num_rejected_cells": num_rejected_cells,
            "total_num_cells": cell_set.num_cells,
        },
        "timingInfo": cell_set_metadata["timingInfo"]
        | {
            "sampling_rate": compute_sampling_rate(
                period_num=cell_set_metadata["timingInfo"]["period"]["num"],
                period_den=cell_set_metadata["timingInfo"]["period"]["den"],
            )
        },
        "spacingInfo": cell_set_metadata["spacingInfo"],
        "microscope": {"focus": efocus_vals},
        "isx": cell_set_metadata,
    }

    # add method and units metadata if available
    try:
        output_metadata["dataset"]["signal"][0]["method"] = cell_set_metadata[
            "extraProperties"
        ]["idps"]["cellset"]["method"]
    except Exception:
        pass

    try:
        output_metadata["dataset"]["signal"][0]["units"] = cell_set_metadata[
            "extraProperties"
        ]["idps"]["cellset"]["units"]
    except Exception:
        pass

    # add end time to timing info
    output_metadata["timingInfo"]["end"] = compute_end_time(
        output_metadata["timingInfo"]
    )

    metadata[file_key] = output_metadata


def generate_event_set_metadata(
    eventset_filename: str,
    file_key: str,
    metadata: dict,
    efocus_vals: List[int],
):
    """Generate metadata for a single event set file.

    :param eventset_filename: path to the input event set
    :param file_key: key associated with the file for which metadata is constructed
    :param metadata: metadata dictionary
    :param efocus_vals: list of efocus values
    """
    event_set = isx.EventSet.read(eventset_filename)
    event_set_metadata = read_isxd_metadata(eventset_filename)
    metadata[file_key] = {
        "metrics": {"total_num_cells": event_set.num_cells},
        "timingInfo": {
            "start": event_set_metadata["global times"][0],
            "end": event_set_metadata["global times"][1],
            "numSamples": event_set_metadata["numSamples"],
            "numFrames": event_set.timing.num_samples,
            "period": event_set_metadata["signalSteps"][0],
            "sampling_rate": compute_sampling_rate(
                period_num=event_set_metadata["signalSteps"][0]["num"],
                period_den=event_set_metadata["signalSteps"][0]["den"],
            ),
            "dropped": [],
            "cropped": [],
        },
        "microscope": {"focus": efocus_vals},
        "isx": event_set_metadata,
    }


def generate_caiman_output_metadata(
    caiman_output_filename: str, file_key: str, metadata: dict
):
    """Generate metadata for the H5 output produced by CaImAn.

    :param caiman_output_filename: path to the output h5 file produced by CaImAn
    :param file_key: key associated with the output file
    :param metadata: metadata dictionary
    """
    model = load_CNMF(caiman_output_filename)
    num_cells = len(model.estimates.C)
    num_accepted_cells = len(model.estimates.idx_components)
    num_rejected_cells = len(model.estimates.idx_components_bad)

    output_metadata = {
        "metrics": {
            "num_accepted_cells": num_accepted_cells,
            "num_undecided_cells": 0,
            "num_rejected_cells": num_rejected_cells,
            "total_num_cells": num_cells,
        },
    }

    metadata[file_key] = output_metadata


def generate_caiman_workflow_metadata(
    caiman_output_filename: str,
    cellset_raw_filenames: List[str],
    cellset_denoised_filenames: List[str],
    eventset_filenames: List[str],
    original_input_indices: List[int],
    input_movies_files: List[str],
    output_dir: str = None,
):
    """Generate metadata for files produced by the CaImAn workflow.

    :param caiman_output_filename: path to the output h5 file produced by CaImAn
    :param cellset_raw_filenames: path to the raw cell set files
    :param cellset_denoised_filenames: path to the denoised cell set files
    :param eventset_filenames: path to the event set files
    :param original_input_indices: original order of the input files prior to sorting
    :param input_movies_files: list of input movie file paths
    :param output_dir: path to the output directory
    """
    if output_dir is None:
        output_dir = os.getcwd()

    metadata = {}

    # extract e-focus values from input movies if available
    efocus_vals = [None] * len(input_movies_files)
    for i, f in enumerate(input_movies_files):
        try:
            if f.lower().endswith(".isxd"):
                efocus_vals[i] = get_efocus(f)
        except Exception:
            pass

    # caiman h5 output file metadata
    generate_caiman_output_metadata(
        caiman_output_filename=caiman_output_filename,
        file_key="caiman_output",
        metadata=metadata,
    )

    # raw cell set metadata
    for i, c in zip(original_input_indices, cellset_raw_filenames):
        generate_cell_set_metadata(
            cellset_filename=c,
            file_key=f"cellset_raw.{str(i).zfill(3)}",
            metadata=metadata,
            efocus_vals=efocus_vals[i],
        )

    # denoised cell set metadata
    for i, c in zip(original_input_indices, cellset_denoised_filenames):
        generate_cell_set_metadata(
            cellset_filename=c,
            file_key=f"cellset_denoised.{str(i).zfill(3)}",
            metadata=metadata,
            efocus_vals=efocus_vals[i],
        )

    # event set metadata
    for i, e in zip(original_input_indices, eventset_filenames):
        generate_event_set_metadata(
            eventset_filename=e,
            file_key=f"neural_events.{str(i).zfill(3)}",
            metadata=metadata,
            efocus_vals=efocus_vals[i],
        )

    # save metadata to file
    output_metadata_filename = os.path.join(output_dir, "output_metadata.json")
    with open(output_metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)
