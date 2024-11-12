import os
import isx
import cv2
import numpy as np
from typing import List
from toolbox.utils.previews import generate_caiman_workflow_previews
from toolbox.utils.metadata import generate_caiman_workflow_metadata
from toolbox.utils.utilities import get_file_size
import logging

logger = logging.getLogger()


def write_cell_statuses(
    cell_statuses: List[str], cell_set_filenames: List[str]
):
    """Write cell statuses to cell set files.

    :param cell_statuses: status to assign to each individual cell
    :param cell_set_filenames: list of paths to cell set files
    """
    for f in cell_set_filenames:
        cellset = isx.CellSet.read(f, read_only=False)
        for i in range(len(cell_statuses)):
            cellset.set_cell_status(i, cell_statuses[i])
        cellset.flush()


def convert_caiman_output_to_isxd(
    model,
    caiman_output_filename,
    input_movie_files,
    original_input_movie_indices,
):
    """Convert the output H5 file produced by CaImAn to correspond ISXD files.

    :param model: CaImAn model object
    :param caiman_output_filename: path to the output model file produced by CaImAn
    :param input_movie_files: list of path to the input movies
    :param original_input_movie_indices: original order of the input files prior to sorting
    """
    logger.info("Converting CaImAn data to corresponding ISXD files")

    # cells
    num_cells = len(model.estimates.C)
    if num_cells == 0:
        logger.warning(
            "No cells were identified by the algorithm. The CaImAn output will not be converted to corresponding ISXD files."
        )
        return
    cell_names = ["C{:0>3d}".format(i) for i in range(num_cells)]

    # spacing info and footprints
    height = model.dims[0]
    width = model.dims[1]
    footprints = np.reshape(
        model.estimates.A.toarray(), (height, width, num_cells), order="F"
    ).T.astype(np.float32)

    # get number of frames and timing from input isxd movies
    file_ext = os.path.splitext(input_movie_files[0])[1][1:]
    if file_ext in ["isxd"]:
        movies = [isx.Movie.read(f) for f in input_movie_files]
        num_frames_per_movie = [m.timing.num_samples for m in movies]
        timing_info = [m.timing for m in movies]
        spacing_info = [m.spacing for m in movies]
    elif file_ext in ["tif", "tiff"]:
        movies = [isx.Movie.read(f) for f in input_movie_files]
        num_frames_per_movie = [m.timing.num_samples for m in movies]
        timing_info = [
            isx.Timing(
                num_samples=m.timing.num_samples,
                period=isx.Duration.from_msecs(
                    1.0 / model.params.data["fr"] * 1000
                ),
            )
            for m in movies
        ]
        spacing_info = [m.spacing for m in movies]
    elif file_ext in ["avi", "mp4"]:
        caps = [cv2.VideoCapture(f) for f in input_movie_files]
        num_frames_per_movie = [
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps
        ]
        timing_info = [
            isx.Timing(
                num_samples=n,
                period=isx.Duration.from_msecs(
                    1.0 / model.params.data["fr"] * 1000
                ),
            )
            for n in num_frames_per_movie
        ]
        spacing_info = [
            isx.Spacing(num_pixels=model.params.data["dims"])
            for _ in input_movie_files
        ]
        del caps
    else:
        logger.warning(
            f"Conversion of data with file extension '{file_ext}' to ISXD is not supported"
        )
        return
    frame_index_cutoffs = [0] + list(np.cumsum(num_frames_per_movie))

    # construct output filenames
    cellset_denoised_filenames = [
        f"cellset_denoised.{str(i).zfill(3)}.isxd"
        for i in original_input_movie_indices
    ]
    cellset_raw_filenames = [
        f"cellset_raw.{str(i).zfill(3)}.isxd"
        for i in original_input_movie_indices
    ]
    eventset_filenames = [
        f"neural_events.{str(i).zfill(3)}.isxd"
        for i in original_input_movie_indices
    ]

    # write cell sets and neural events to disk
    for i, (
        cellset_raw_filename,
        cellset_denoised_filename,
        eventset_filename,
    ) in enumerate(
        zip(
            cellset_raw_filenames,
            cellset_denoised_filenames,
            eventset_filenames,
        )
    ):
        denoised_cellset = isx.CellSet.write(
            cellset_denoised_filename, timing_info[i], spacing_info[i]
        )
        raw_cellset = isx.CellSet.write(
            cellset_raw_filename, timing_info[i], spacing_info[i]
        )
        neural_events = isx.EventSet.write(
            eventset_filename, timing_info[i], cell_names
        )

        frame_indices = np.arange(
            frame_index_cutoffs[i], frame_index_cutoffs[i + 1]
        )
        time_offsets = np.array(
            [
                x.to_usecs()
                for x in denoised_cellset.timing.get_offsets_since_start()
            ],
            np.uint64,
        )

        for cell_index in range(num_cells):
            footprint = np.flipud(np.rot90(footprints[cell_index]))

            # denoised cell set
            denoised_trace = np.array(
                model.estimates.C[cell_index, frame_indices], dtype=np.float32
            )
            denoised_cellset.set_cell_data(
                index=cell_index,
                image=footprint,
                trace=denoised_trace,
                name=cell_names[cell_index],
            )

            # raw cell set
            raw_trace = np.array(
                model.estimates.C[cell_index, frame_indices], dtype=np.float32
            ) + np.array(
                model.estimates.YrA[cell_index, frame_indices],
                dtype=np.float32,
            )
            raw_cellset.set_cell_data(
                index=cell_index,
                image=footprint,
                trace=raw_trace,
                name=cell_names[cell_index],
            )

            # neural events
            events_trace = model.estimates.S[cell_index, frame_indices].astype(
                np.float32
            )
            event_indices = np.argwhere(events_trace > 0)
            event_amplitudes = events_trace[event_indices].reshape(-1)
            event_offsets = time_offsets[event_indices]
            neural_events.set_cell_data(
                index=cell_index,
                offsets=event_offsets,
                amplitudes=event_amplitudes,
            )

        denoised_cellset.flush()
        raw_cellset.flush()
        neural_events.flush()

        # log files creates
        logger.info(
            f"Raw cell set saved "
            f"({os.path.basename(cellset_raw_filename)}, "
            f"size: {get_file_size(cellset_raw_filename)})"
        )
        logger.info(
            f"Denoised cell set saved "
            f"({os.path.basename(cellset_denoised_filename)}, "
            f"size: {get_file_size(cellset_denoised_filename)})"
        )
        logger.info(
            f"Neural events saved "
            f"({os.path.basename(eventset_filename)}, "
            f"size: {get_file_size(eventset_filename)})"
        )

    # update cell statuses
    cell_statuses = np.array(["accepted" for _ in range(num_cells)])
    cell_statuses[model.estimates.idx_components_bad] = "rejected"
    write_cell_statuses(
        cell_statuses=cell_statuses,
        cell_set_filenames=cellset_raw_filenames + cellset_denoised_filenames,
    )

    # generate previews
    logger.info("Generating data previews")
    generate_caiman_workflow_previews(
        cellset_raw_filenames=cellset_raw_filenames,
        cellset_denoised_filenames=cellset_denoised_filenames,
        eventset_filenames=eventset_filenames,
        original_input_indices=original_input_movie_indices,
    )

    # generate metadata
    logger.info("Generating output metadata")
    generate_caiman_workflow_metadata(
        caiman_output_filename=caiman_output_filename,
        cellset_raw_filenames=cellset_raw_filenames,
        cellset_denoised_filenames=cellset_denoised_filenames,
        eventset_filenames=eventset_filenames,
        original_input_indices=original_input_movie_indices,
        input_movies_files=input_movie_files,
    )
