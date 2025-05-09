from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import cv2
import isx
import json
import logging
from numpy import ndarray
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.sparse import csc_matrix

# from typing import List
from toolbox.utils import config
from toolbox.utils.exceptions import IdeasError
from toolbox.utils.metrics import (
    compute_alignment_metrics,
    compute_distance_matrix_metrics,
)
from toolbox.utils.utilities import (
    compute_sampling_rate,
    compute_end_time,
    get_num_cells_by_status,
    read_isxc_metadata,
    read_isxd_metadata,
)

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
    num_accepted_cells = (
        len(model.estimates.idx_components)
        if model.estimates.idx_components is not None
        else 0
    )
    num_rejected_cells = (
        len(model.estimates.idx_components_bad)
        if model.estimates.idx_components_bad is not None
        else 0
    )

    output_metadata = {
        "metrics": {
            "num_accepted_cells": num_accepted_cells,
            "num_undecided_cells": num_cells
            - num_accepted_cells
            - num_rejected_cells,
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

    for output_file_index, i in enumerate(original_input_indices):
        # raw cell set metadata
        generate_cell_set_metadata(
            cellset_filename=cellset_raw_filenames[output_file_index],
            file_key=f"cellset_raw.{str(output_file_index).zfill(3)}",
            metadata=metadata,
            efocus_vals=efocus_vals[i],
        )

        # denoised cell set metadata
        generate_cell_set_metadata(
            cellset_filename=cellset_denoised_filenames[output_file_index],
            file_key=f"cellset_denoised.{str(output_file_index).zfill(3)}",
            metadata=metadata,
            efocus_vals=efocus_vals[i],
        )

        # event set metadata
        generate_event_set_metadata(
            eventset_filename=eventset_filenames[output_file_index],
            file_key=f"neural_events.{str(output_file_index).zfill(3)}",
            metadata=metadata,
            efocus_vals=efocus_vals[i],
        )

    # save metadata to file
    output_metadata_filename = os.path.join(output_dir, "output_metadata.json")
    with open(output_metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)


def generate_movie_metadata(
    movie_filename: str,
    file_key: str,
    metadata: dict,
    efocus_vals: List[int],
    sampling_rate: float,
):
    """Generate metadata for a single movie file.

    :param movie_filename: path to the input cell set
    :param file_key: key associated with the file for which metadata is constructed
    :param metadata: metadata dictionary
    :param efocus_vals: list of efocus values
    :param sampling_rate: frame rate of the movie
    """
    file_ext = os.path.splitext(movie_filename)[1][1:]
    if file_ext in ["isxd"]:
        isx_metadata = read_isxd_metadata(movie_filename)
        timing_info = isx_metadata["timingInfo"]
        timing_info["sampling_rate"] = float(np.round(sampling_rate, 2))
        spacing_info = isx_metadata["spacingInfo"]
    elif file_ext in ["tif", "tiff"]:
        image_stack = Image.open(movie_filename)
        timing_info = {
            "numTimes": image_stack.n_frames,
            "sampling_rate": float(np.round(sampling_rate, 2)),
        }
        spacing_info = {
            "numPixels": {
                "x": image_stack.width,
                "y": image_stack.height,
                "z": image_stack.n_frames,
            },
        }
    elif file_ext in ["avi", "mp4"]:
        cap = cv2.VideoCapture(movie_filename)
        timing_info = {
            "numTimes": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "sampling_rate": float(np.round(sampling_rate, 2)),
        }
        spacing_info = {
            "numPixels": {
                "x": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "y": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            },
        }
        del cap
    else:
        timing_info = None
        spacing_info = None

    metadata[file_key] = {
        "timingInfo": timing_info,
        "spacingInfo": spacing_info,
        "microscope": {"focus": efocus_vals},
    }


def generate_motion_correction_qc_metadata(
    mc_obj,
    file_key: str,
    metadata: dict,
):
    """Generate metadata for motion correction quality control data.

    :param mc_obj: CaImAn motion correction object
    :param file_key: key associated with the file for which metadata is constructed
    :param metadata: metadata dictionary
    """
    # rigid shifts metadata
    x_shifts_rig, y_shifts_rig = list(zip(*mc_obj.shifts_rig))
    mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": np.min(x_shifts_rig),
            "max_x_rigid_shift": np.max(x_shifts_rig),
            "min_y_rigid_shift": np.min(y_shifts_rig),
            "max_y_rigid_shift": np.max(y_shifts_rig),
        }
    }

    # piecewise rigid shifts metadata
    if mc_obj.pw_rigid:
        mc_qc_metadata["metrics"]["min_x_pw_rigid_shift"] = np.min(
            mc_obj.x_shifts_els
        )
        mc_qc_metadata["metrics"]["max_x_pw_rigid_shift"] = np.max(
            mc_obj.x_shifts_els
        )
        mc_qc_metadata["metrics"]["min_y_pw_rigid_shift"] = np.min(
            mc_obj.y_shifts_els
        )
        mc_qc_metadata["metrics"]["max_y_pw_rigid_shift"] = np.max(
            mc_obj.y_shifts_els
        )
        mc_qc_metadata["metrics"]["num_patches"] = len(mc_obj.x_shifts_els[0])

    # update metadata dictionary
    metadata[file_key] = mc_qc_metadata


def generate_caiman_motion_correction_metadata(
    mc_movie_filenames: str,
    mc_obj,
    original_input_indices: List[int],
    input_movies_files: List[str],
    sampling_rate: float,
    output_dir: str = None,
):
    """Generate metadata for files produced by the CaImAn workflow.

    :param mc_movie_filenames: path to the motion-corrected movie files
    :param mc_obj: CaImAn motion correction object
    :param original_input_indices: original order of the input files prior to sorting
    :param input_movies_files: list of input movie file paths
    :param sampling_rate: frame rate of the output movie
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

    # motion-corrected movie metadata
    for output_file_index, i in enumerate(original_input_indices):
        generate_movie_metadata(
            movie_filename=mc_movie_filenames[output_file_index],
            file_key=f"mc_movie.{str(output_file_index).zfill(3)}",
            metadata=metadata,
            efocus_vals=efocus_vals[i],
            sampling_rate=sampling_rate,
        )

    # motion correction quality assessment metadata
    generate_motion_correction_qc_metadata(
        mc_obj=mc_obj, file_key="mc_qc_data", metadata=metadata
    )

    # save metadata to file
    output_metadata_filename = os.path.join(output_dir, "output_metadata.json")
    with open(output_metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)


def generate_caiman_spike_extraction_metadata(
    cellset_denoised_filenames: List[str],
    eventset_filenames: List[str],
    original_input_indices: List[int],
    input_cellset_files: List[str],
    output_dir: str = None,
):
    """Generate metadata for files produced by CaImAn spike extraction.

    :param cellset_denoised_filenames: path to the denoised cell set files
    :param eventset_filenames: path to the event set files
    :param original_input_indices: original order of the input files prior to sorting
    :param input_cellset_files: list of input cellset file paths
    :param output_dir: path to the output directory
    """
    if output_dir is None:
        output_dir = os.getcwd()

    metadata = {}

    # extract e-focus values from input cell sets if available
    efocus_vals = [None] * len(input_cellset_files)
    for i, f in enumerate(input_cellset_files):
        try:
            if f.lower().endswith(".isxd"):
                efocus_vals[i] = get_efocus(f)
        except Exception:
            pass

    for output_file_index, i in enumerate(original_input_indices):
        # denoised cell set metadata
        generate_cell_set_metadata(
            cellset_filename=cellset_denoised_filenames[output_file_index],
            file_key=f"cellset_denoised.{str(output_file_index).zfill(3)}",
            metadata=metadata,
            efocus_vals=efocus_vals[i],
        )

        # event set metadata
        generate_event_set_metadata(
            eventset_filename=eventset_filenames[output_file_index],
            file_key=f"neural_events.{str(output_file_index).zfill(3)}",
            metadata=metadata,
            efocus_vals=efocus_vals[i],
        )

    # save metadata to file
    output_metadata_filename = os.path.join(output_dir, "output_metadata.json")
    with open(output_metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)


@beartype
def create_msr_output_metadata(
    *,
    spatial: List[csc_matrix],
    dims: Tuple[int, int],
    df_assignments: pd.DataFrame,
    n_reg_cells_all: int,
    templates: List[ndarray],
    aligned_templates: List[ndarray],
    xy_shifts: Union[List[List[Tuple[float, float]]], List[ndarray]],
    D: List[List[ndarray]],
    D_cm: List[List[ndarray]],
    cs_out_paths: List[str],
    es_out_paths: List[Optional[str]],
    output_dir: str,
) -> None:
    """
    Create metadata for all output files of CaImAn Multi-Session Registration.
    """
    # get metadata values for CaImAn_MSR_output.h5
    n_sessions = len(spatial)
    n_cells_list = [x.shape[1] for x in spatial]
    yield_perc = 100 * n_reg_cells_all / np.min(n_cells_list)
    (
        mean_overlap,
        mean_centroid_dist,
        mean_overlap_nonreg,
        mean_centroid_dist_nonreg,
    ) = compute_distance_matrix_metrics(
        df_assignments=df_assignments,
        D=D,
        D_cm=D_cm,
    )

    metadata_msr_output_dict = {
        "n_sessions": n_sessions,
        "cell_counts": n_cells_list,
        "n_regist_cells": n_reg_cells_all,
        "yield": round(yield_perc, 2),
        "mean_overlap": round(mean_overlap, 2),
        "mean_centroid_dist": round(mean_centroid_dist, 2),
    }

    msr_output_key = Path(config.MSR_OUTPUT_FNAME).stem
    metadata = {
        msr_output_key: metadata_msr_output_dict,
    }

    # get metadata values for alignment_qc_metrics.csv
    r_pre, r_post, mean_shifts, max_shifts = compute_alignment_metrics(
        templates=templates,
        aligned_templates=aligned_templates,
        xy_shifts=xy_shifts,
    )
    mean_shifts = [round(x, 2) for x in mean_shifts]
    max_shifts = [round(x, 2) for x in max_shifts]

    metadata_align_output_dict = {
        "n_session_pairs": n_sessions - 1,
        "r_pre": round(r_pre, 3),
        "r_post": round(r_post, 3),
        "mean_shifts": f"({mean_shifts[0]}, {mean_shifts[1]})",
        "max_shifts": f"({max_shifts[0]}, {max_shifts[1]})",
    }

    align_output_key = Path(config.ALIGN_METRICS_CSV_FNAME).stem
    metadata.update({align_output_key: metadata_align_output_dict})

    # get metadata values for CaImAn_MSR_metrics.csv
    metadata_metrics_output_dict = {
        "n_sessions": n_sessions,
        "n_regist_cells": n_reg_cells_all,
        "mean_overlap": round(mean_overlap, 2),
        "mean_centroid_dist": round(mean_centroid_dist, 2),
        "mean_overlap_nonreg": round(mean_overlap_nonreg, 2),
        "mean_centroid_dist_nonreg": round(mean_centroid_dist_nonreg, 2),
    }

    metrics_output_key = Path(config.MSR_METRICS_CSV_FNAME).stem
    metadata.update({metrics_output_key: metadata_metrics_output_dict})

    # get metadata values for Inscopix cellsets and optional eventsets
    for idx_sess, (cs_out_path, es_out_path) in enumerate(
        zip(cs_out_paths, es_out_paths)
    ):
        cs = isx.CellSet.read(cs_out_path)
        L = cs.timing.num_samples
        fs = 1 / cs.timing.period.secs_float
        N = cs.num_cells
        status_list = [cs.get_cell_status(x) for x in range(N)]

        metadata_isxd_outputs_dict = {
            "Multisession registration method": "CaImAn MSR",
            "Session index among registered sessions": idx_sess,
            "Number of Frames": L,
            "Sampling Rate (Hz)": round(fs, 2),
            "Recording Duration (s)": round(L / fs, 3),
            "spacingInfo": {"numPixels": {"x": dims[0], "y": dims[1]}},
        }

        accept_list = [x == "accepted" for x in status_list]
        N_acc = int(np.sum(accept_list))
        N_rej = int(np.sum([x == "rejected" for x in status_list]))
        if N_acc > 0:
            idx_cells = np.where(accept_list)[0]
        else:
            idx_cells = np.where([x != "rejected" for x in status_list])[0]
        cs_traces = [cs.get_cell_trace_data(x) for x in idx_cells]
        avg_amp = round(float(np.mean(cs_traces)), 3)

        metadata_cs_output_dict = metadata_isxd_outputs_dict.copy()
        metadata_cs_output_dict.update(
            {
                "Number of Extracted ROIs": N,
                "Number of Accepted ROIs": N_acc,
                "Number of Rejected ROIs": N_rej,
                "Average trace amplitude": avg_amp,
            }
        )
        cs_output_name = f"{config.CS_REG_PREFIX}.{idx_sess:03g}.isxd"
        cs_output_key = Path(cs_output_name).stem
        metadata.update({cs_output_key: metadata_cs_output_dict})

        if es_out_path is not None:
            es = isx.EventSet.read(es_out_path)

            # could use values obtained from cs above, but safer this way
            # to ensure coherence between cs and es
            L = es.timing.num_samples
            fs = 1 / es.timing.period.secs_float
            N = es.num_cells

            avg_evt_rates = [
                np.sum(es.get_cell_data(x)[1] > 1) / L * fs for x in idx_cells
            ]

            metadata_es_output_dict = metadata_isxd_outputs_dict.copy()
            metadata_es_output_dict.update(
                {
                    "Number of cells": N,
                    "Average event rate (Hz)": round(
                        float(np.mean(avg_evt_rates)), 4
                    ),
                    "Minimum event rate (Hz)": round(
                        float(np.min(avg_evt_rates)), 4
                    ),
                    "Maximum event rate (Hz)": round(
                        float(np.max(avg_evt_rates)), 4
                    ),
                }
            )
            es_output_name = f"{config.ES_REG_PREFIX}.{idx_sess:03g}.isxd"
            es_output_key = Path(es_output_name).stem
            metadata.update({es_output_key: metadata_es_output_dict})

    with open(os.path.join(output_dir, "output_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
