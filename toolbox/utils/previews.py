import os
import isx
from typing import List
import caiman as cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from toolbox.utils.plots import (
    save_neural_traces_preview,
    save_footprints_preview,
    EventSetPreview,
)
from toolbox.utils.utilities import get_num_cells_by_status
from toolbox.utils.utilities import get_file_size
import logging

logger = logging.getLogger()


def generate_cell_set_previews(
    cellset_filename: str,
    output_dir: str = None,
    vertical_line_indices: list[int] = None,
):
    """Generate traces and footprints previews for a single cell set file.

    :param cellset_filename: path to the input cell set
    :param output_dir: path to the output directory
    :param vertical_line_indices: frame indices along which to draw vertical lines
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
            vertical_line_indices=vertical_line_indices,
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
    else:
        logger.warning(
            f"There are no accepted or undecided cells. No preview will be generated for the cell set file '{os.path.basename(cellset_filename)}'"
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
    global_cellset_filename: str,
    output_dir: str = None,
):
    """Generate previews for files produced by the CaImAn workflow.

    :param cellset_raw_filenames: path to the raw cell set files
    :param cellset_denoised_filenames: path to the denoised cell set files
    :param eventset_filenames: path to the event set files
    :param original_input_indices: original order of the input files prior to sorting
    :param global_cellset_filename: path to the global cell set in which all traces are concatenated
    :param output_dir: path to the output directory
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # raw cell set previews
    for i in original_input_indices:
        try:
            generate_cell_set_previews(
                cellset_filename=cellset_raw_filenames[i],
                output_dir=output_dir,
            )
        except Exception as e:
            logger.warning(
                f"Preview could not be generated for file '{os.path.basename(cellset_raw_filenames[i])}': {str(e)}"
            )

    # denoised cell set previews
    for i in original_input_indices:
        try:
            generate_cell_set_previews(
                cellset_filename=cellset_denoised_filenames[i],
                output_dir=output_dir,
            )
        except Exception as e:
            logger.warning(
                f"Preview could not be generated for file '{os.path.basename(cellset_denoised_filenames[i])}': {str(e)}"
            )

    # event set previews
    for i in original_input_indices:
        try:
            generate_event_set_preview(
                eventset_filename=eventset_filenames[i], output_dir=output_dir
            )
        except Exception as e:
            logger.warning(
                f"Preview could not be generated for file '{os.path.basename(eventset_filenames[i])}': {str(e)}"
            )

    # global cell set
    try:
        if len(cellset_raw_filenames) > 0:
            individual_movie_start_indices = [0]
            for f in cellset_raw_filenames:
                cs = isx.CellSet.read(f)
                individual_movie_start_indices.append(
                    individual_movie_start_indices[-1] + cs.timing.num_samples
                )
            individual_movie_start_indices = individual_movie_start_indices[
                1:-1
            ]
        else:
            individual_movie_start_indices = None

        generate_cell_set_previews(
            cellset_filename=global_cellset_filename,
            output_dir=output_dir,
            vertical_line_indices=individual_movie_start_indices,
        )
    except Exception as e:
        logger.warning(
            f"Footprints and traces previews could not be generated for the CaImAn output data: {str(e)}"
        )
    if os.path.exists(global_cellset_filename):
        os.remove(global_cellset_filename)


def generate_initialization_images_preview(
    images,
    output_dir: str = None,
):
    """Generate previews for files produced by the CaImAn workflow.

    :param images: list of images (num_frames, height, width)
    :param output_dir: path to the output directory
    """
    if output_dir is None:
        output_dir = os.getcwd()

    try:
        logger.info("Generating pnr, correlation, and search images")
        num_frames = len(images)
        correlation_image, pnr_image = cm.summary_images.correlation_pnr(
            images[:: max(num_frames // 1000, 1)], gSig=3, swap_dim=False
        )
        search_image = correlation_image * pnr_image

        fig, ax = plt.subplots(nrows=1, ncols=3)

        # PNR image
        pnr_plot_img = ax[0].imshow(pnr_image)
        ax[0].set_title("PNR Image")

        # Correlation image
        correlation_plot_img = ax[1].imshow(correlation_image)
        ax[1].set_title("Correlation Image")

        # Search image
        search_plot_img = ax[2].imshow(search_image)
        ax[2].set_title("Search Image")

        # add colorbars
        def add_colorbar(target_axis, img):
            divider = make_axes_locatable(target_axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, cax=cax)

        for i, im in enumerate(
            [pnr_plot_img, correlation_plot_img, search_plot_img]
        ):
            add_colorbar(target_axis=ax[i], img=im)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "initialization_images.png"),
            bbox_inches="tight",
            dpi=300,
        )
    except Exception as e:
        logger.warning(
            f"Initialization images (pnr, correlation, search) could not be generated: {str(e)}"
        )
        correlation_image = None

    return correlation_image
