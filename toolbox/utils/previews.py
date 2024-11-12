import os
from typing import List
from toolbox.utils.plots import (
    save_neural_traces_preview,
    save_footprints_preview,
    EventSetPreview,
)
from toolbox.utils.utilities import get_num_cells_by_status
from toolbox.utils.utilities import get_file_size
import logging

logger = logging.getLogger()


def generate_cell_set_previews(cellset_filename: str, output_dir: str = None):
    """Generate traces and footprints previews for a single cell set file.

    :param cellset_filename: path to the input cell set
    :param output_dir: path to the output directory
    """
    if output_dir is None:
        output_dir = os.getcwd()

    (
        num_accepted_cells,
        num_undecided_cells,
        num_rejected_cells,
    ) = get_num_cells_by_status(cellset_filename)

    if (num_accepted_cells + num_undecided_cells) > 0:
        # traces preview
        cellset_basename = os.path.splitext(
            os.path.basename(cellset_filename)
        )[0]
        output_traces_preview_filepath = (
            os.path.join(output_dir, "traces_" + cellset_basename) + ".png"
        )
        save_neural_traces_preview(
            cell_set_file=cellset_filename,
            output_preview_filename=output_traces_preview_filepath,
        )
        logger.info(
            f"Cell set traces preview saved "
            f"({os.path.basename(output_traces_preview_filepath)}, "
            f"size: {get_file_size(output_traces_preview_filepath)})"
        )

        # footprints preview
        output_footprints_preview_filepath = (
            os.path.join(output_dir, "footprints_" + cellset_basename) + ".png"
        )
        save_footprints_preview(
            cell_set_file=cellset_filename,
            output_preview_filename=output_footprints_preview_filepath,
        )
        logger.info(
            f"Cell set footprints preview saved "
            f"({os.path.basename(output_footprints_preview_filepath)}, "
            f"size: {get_file_size(output_footprints_preview_filepath)})"
        )


def generate_event_set_preview(eventset_filename: str, output_dir: str = None):
    """Generate previews for a single event set file.

    :param eventset_filename: path to the input event set
    :param output_dir: path to the output directory
    """
    if output_dir is None:
        output_dir = os.getcwd()

    eventset_basename = os.path.splitext(os.path.basename(eventset_filename))[
        0
    ]
    output_events_preview_filepath = (
        os.path.join(output_dir, "preview_" + eventset_basename) + ".png"
    )
    eventset_preview_obj = EventSetPreview(
        input_eventset_filepath=eventset_filename,
        output_png_filepath=output_events_preview_filepath,
    )
    eventset_preview_obj.generate_preview()
    logger.info(
        f"Neural events preview saved "
        f"({os.path.basename(output_events_preview_filepath)}, "
        f"size: {get_file_size(output_events_preview_filepath)})"
    )


def generate_caiman_workflow_previews(
    cellset_raw_filenames: List[str],
    cellset_denoised_filenames: List[str],
    eventset_filenames: List[str],
    original_input_indices: List[int],
    output_dir: str = None,
):
    """Generate previews for files produced by the CaImAn workflow.

    :param cellset_raw_filenames: path to the raw cell set files
    :param cellset_denoised_filenames: path to the denoised cell set files
    :param eventset_filenames: path to the event set files
    :param original_input_indices: original order of the input files prior to sorting
    :param output_dir: path to the output directory
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # raw cell set previews
    for i in original_input_indices:
        try:
            generate_cell_set_previews(cellset_raw_filenames[i], output_dir)
        except Exception as e:
            logger.warning(
                f"Preview could not be generated for file '{os.path.basename(cellset_raw_filenames[i])}': {str(e)}"
            )

    # denoised cell set previews
    for i in original_input_indices:
        try:
            generate_cell_set_previews(
                cellset_denoised_filenames[i], output_dir
            )
        except Exception as e:
            logger.warning(
                f"Preview could not be generated for file '{os.path.basename(cellset_denoised_filenames[i])}': {str(e)}"
            )

    # event set previews
    for i in original_input_indices:
        try:
            generate_event_set_preview(eventset_filenames[i], output_dir)
        except Exception as e:
            logger.warning(
                f"Preview could not be generated for file '{os.path.basename(eventset_filenames[i])}': {str(e)}"
            )
