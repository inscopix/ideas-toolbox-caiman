import os
import isx
import math
import cv2
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
from typing import List
import caiman as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from toolbox.utils.plots import (
    save_neural_traces_preview,
    save_footprints_preview,
    EventSetPreview,
)
from toolbox.utils.utilities import (
    get_num_cells_by_status,
    get_file_size,
)
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
            os.path.join(output_dir, "traces_" + cellset_basename) + ".svg"
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
            os.path.join(output_dir, "footprints_" + cellset_basename) + ".svg"
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
        os.path.join(output_dir, "preview_" + eventset_basename) + ".svg"
    )
    eventset_preview_obj = EventSetPreview(
        input_eventset_filepath=eventset_filename,
        output_svg_filepath=output_events_preview_filepath,
    )
    eventset_preview_obj.generate_preview()

    logger.info(
        f"Neural events preview saved "
        f"({os.path.basename(output_events_preview_filepath)}, "
        f"size: {get_file_size(output_events_preview_filepath)})"
    )


def _map_movie_to_preview_frame_ind(
    preview_frame_ind: int,
    preview_sampling_period: float,
    movie_num_frames: int,
    movie_sampling_period: float,
) -> list[int]:
    """Map a sequence of frames in a movie to one frame in the corresponding preview for that movie.
    Movie frames that belong to a preview frame are determined by comparing timestamps from frames in both movies.
    To demonstrate this, consider the following example:
        Movie frame rate: 13 Hz
        Preview frame rate: 10 Hz
        For 1 second of time, frames will be displayed at the following times for the movie and preview:
            Movie frame timestamps = [0.0, 0.083, 0.16, 0.25, 0.33, 0.42, 0.50, 0.58, 0.67, 0.75, 0.83, 0.91, 1.0]
            Preview frame timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        Let fp(i) represent the ith preview frame.
        Let fm(i) represent the ith movie frame.
        fp(0) will be a linear interpolation of all frames in the original movie that
        occurred between 0.0 s - 0.1 s:
            fp(0) = ((0.083 - 0.0) / 0.1) * fm(0) + ((0.1 - 0.083) / 0.1) * fm(1)
            fp(0) = 0.83 * fm(0) + 0.17 * fm(1)

    :param preview_frame_ind: The index of the preview frame.
        The function will determine which frames in the original movie occur during the
        duration of the preview frame.
    :param preview_sampling_period: The sampling period of the preview.
    :param movie_num_frames: The total number of frames in the original movie.
        If no frames from the original movie occur during the duration of the preview frame,
        an empty list is returned.
    :param movie_sampling_period: The sampling period of the original movie.

    :return preview_frame_map: List of tuples. Each tuple consists of a frame index and floating
        point number between 0-1 representing the fraction of time the movie frame was displayed within
        throughout the duration of the preview frame.
    """
    # Step 1: Calculate the start and end timestamps of the preview frame
    # based on the preview sampling period and frame index.
    preview_frame_start_ts = preview_sampling_period * preview_frame_ind
    preview_frame_end_ts = preview_sampling_period * (preview_frame_ind + 1)

    # Step 2: Calculate the first movie frame that starts before the start of the preview frame.
    movie_frame_ind = math.floor(
        preview_frame_start_ts / movie_sampling_period
    )

    # Step 3: Starting from the first movie frame that starts before the start of the preview frame,
    # find all movie frames that occur during the duration of the preview frame.
    preview_frame_map = []
    while movie_frame_ind < movie_num_frames:
        # Step 3.1: Calculate the start and end timestamps of the movie frame
        # based on the movie sampling period and frame index.
        movie_frame_start_ts = movie_sampling_period * movie_frame_ind
        movie_frame_end_ts = movie_sampling_period * (movie_frame_ind + 1)

        # Step 3.2: Determine if movie frame occurs within the preview frame.
        if movie_frame_start_ts >= preview_frame_end_ts:
            # The movie frame starts after the end of the preview frame,
            # so the movie frame does not occur within the preview frame.
            # No more movie frames will occur in the preview frame, exit loop.
            break
        elif movie_frame_end_ts <= preview_frame_end_ts:
            # The movie frame ends before the preview frame,
            # so the movie frame does occur within the preview frame.
            if preview_frame_map:
                # There's already an earlier movie frame that also occurs within the preview frame.
                # Calculate when the previous movie frame ended in order to determine the current
                # movie frame contribution.
                last_frame_end_ts = movie_sampling_period * (
                    preview_frame_map[-1][0] + 1
                )
            else:
                # This is the first movie frame that occurs within the preview frame.
                # Use the start of the preview frame to determine the current movie frame contribution.
                last_frame_end_ts = preview_frame_start_ts

            # Calculate the movie frame contribution as the ratio of time that the movie frame is displayed
            # over the time that the preview frame is displayed.
            movie_frame_contribution = (
                movie_frame_end_ts - last_frame_end_ts
            ) / preview_sampling_period
            preview_frame_map.append(
                (movie_frame_ind, movie_frame_contribution)
            )

            # Move on to next frame in the movie.
            movie_frame_ind += 1
        else:
            # The movie frame ends after the preview frame,
            # so the movie frame does occur within the preview frame, but this will be the last movie frame
            # that occurs within the preview frame, exit loop after adding it to the output.

            # Calculate the movie frame contribution as the ratio of time that the movie frame is displayed
            # over the time that the preview frame is displayed.
            movie_frame_contribution = (
                preview_frame_end_ts - movie_frame_start_ts
            ) / preview_sampling_period
            preview_frame_map.append(
                (movie_frame_ind, movie_frame_contribution)
            )
            break

    # Step 4: Check edge case where sometimes the last preview frame that occurs throughout the duration
    # of the movie ends after the last movie frame. Discard this preview frame as it's incomplete.
    if movie_frame_ind == movie_num_frames and preview_frame_map:
        total_movie_frame_contribution = 0.0
        for data in preview_frame_map:
            _, movie_frame_contribution = data
            total_movie_frame_contribution += movie_frame_contribution
        if not np.allclose(total_movie_frame_contribution, 1.0):
            preview_frame_map = []

    return preview_frame_map


def _transform_movie_to_preview_shape(
    movie_frame_shape: tuple[int, int], preview_frame_shape: tuple[int, int]
) -> tuple[int, int]:
    """Transform movie frame shape to optimally fit within the corresponding preview of the movie.
    Given a desired preview frame shape, determine the largest scaled version of the movie frame shape
    that fits within the preview frame shape, and maintains aspect ratio of the movie frame shape.
    At least one side of the output scaled movie frame shape will be equal to the corresponding
    side in the preview frame shape.

    :param movie_frame_shape: The shape of the movie frame. Represented as (num_rows, num_cols)
        where num_rows is the number of rows in a movie frame, and num_cols is the number of columns
        in a movie frame.
    :param preview_frame_shape: The shape of the preview frame. Represented as (num_rows, num_cols)
        where num_rows is the number of rows in a movie frame, and num_cols is the number of columns
        in a movie frame.

    :return scaled_frame_shape: The scaled shape of the movie frame. Represented as (num_rows, num_cols)
            where num_rows is the number of rows in a movie frame, and num_cols is the number of columns
            in a movie frame. The aspect ratio of scaled_movie_frame_shape should be very close to movie_frame_shape
    """
    # Step 1: Check if the movie frame shape is smaller than the preview frame shape.
    # If so, return the movie frame shape.
    if np.all(
        [movie_frame_shape[i] <= preview_frame_shape[i] for i in range(2)]
    ):
        return movie_frame_shape

    # Step 2: Determine the dimension that needs to be scaled down
    # the most to equal the corresponding preview frame dimension
    # and used this as the scale factor to apply on the movie frame shape
    scale_factor = np.min(
        [preview_frame_shape[i] / movie_frame_shape[i] for i in range(2)]
    )

    # We know that at least one movie frame shape dimension is larger than the
    # corresponding preview frame shape dimension, so at least one scale factor
    # should be less than zero.
    assert scale_factor < 1

    # Step 3: Scale the movie frame shape by the scale factor
    scaled_frame_shape = np.empty(shape=(2,), dtype=int)
    for i in range(2):
        scaled_frame_shape[i] = round(
            float(movie_frame_shape[i]) * scale_factor
        )

    return tuple(scaled_frame_shape)


def generate_movie_preview(
    input_filename: str,
    preview_filename: str,
    preview_max_duration: int = 120,
    preview_max_sampling_rate: float = 10,
    preview_max_resolution: tuple[int, int] = (640, 400),
    preview_crf: int = 23,
    preview_max_size: int = 50,
):
    """Generate a preview for a movie.

    The preview is a compressed version of the movie with a fixed duration, sampling rate, and resolution in
    order to limit the resulting file size. The preview is generated by ffmpeg and compressed using the H.264 standard.

    :param input_filename: The filename of the movie data that is used to generate a preview.
        Supported file formats are: .isxd, .isxb, .mp4, .avi, .tiff/.tif stack, and a list of individual image files (.tiff/.tif).
    :param preview_filename: The filename of the output preview.
    :param preview_max_duration: The maximum duration, in minutes, that the preview can be.
        If the movie duration is less than this value, the preview duration will be equal to the movie duration.
        If the movie duration is greater than this value, the preview duration will be eqaul to this value.
    :param preview_max_sampling_rate: The maximum sampling rate, in Hz, that the preview can be.
        If the movie sampling rate is less than this value, the preview sampling rate will be equal to the movie sampling rate.
        If the movie sampling rate is greater than this value, the preview sampling rate will be eqaul to this value. See
        the function _map_movie_to_preview_frame_ind for more info on how movie frames are temporally downsampled to
        a desired preview sampling rate.
    :param preview_max_resolution: The maximum resolution, in pixels of (width, height), that the preview can be.
        If the movie resolution is smaller than this value, the preview resolution will be equal to the movie resolution.
        If the movie resolution is larger than this value, the preview resolution will be equal to the largest scaled
        movie resolution that fits within the maximum resolution. See the function _transform_movie_to_preview_shape for
        more info on how this resolution is calculated.
    :param preview_crf: Compression factor for preview. CRF stands for constant rate factor used in the H.264 standard
        to control for quality vs file size. The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default,
        and 51 is worst quality possible. A lower value generally leads to higher quality, and a subjectively sane range is 17–28.
        Read more about this parameter here: https://trac.ffmpeg.org/wiki/Encode/H.264
    :param preview_max_size: The max file size for the preview, in MB.
        The determines the max bit rate for the preview.
        Read more about max bit rate in ffmpeg here: https://trac.ffmpeg.org/wiki/Limiting%20the%20output%20bitrate
    """
    # Step 1: Determine file type based on extension
    # Read in properties of movie in order to determine
    # preview duration, sampling rate, and resolution
    if isinstance(input_filename, list):
        # input is a sequence of individual image files
        is_image_sequence = True

        if len(input_filename) <= 1:
            raise ToolException(
                ExitStatus.IDPS_ERROR_0003,
                "A movie preview cannot be generated from a single frame",
            )

        _, file_extension = os.path.splitext(input_filename[0].lower())
        if file_extension not in [".tiff", ".tif"]:
            raise ToolException(
                ExitStatus.IDPS_ERROR_0002,
                "Only sequences of tiff/tif images are supported",
            )

        image = Image.open(input_filename[0])
        frame_width = image.width
        frame_height = image.height
        num_frames = len(input_filename)
        movie_sampling_rate = preview_max_sampling_rate
        movie_sampling_period = 1 / movie_sampling_rate
    else:
        # input is a single movie file
        is_image_sequence = False

        _, file_extension = os.path.splitext(input_filename.lower())
        if file_extension in [".isxd", ".isxb"]:
            # Use isx API to read isxd and isxb movies
            movie = isx.Movie.read(input_filename)
            frame_height, frame_width = movie.spacing.num_pixels
            num_frames = movie.timing.num_samples
            movie_sampling_period = movie.timing.period.secs_float
            movie_sampling_rate = 1 / movie_sampling_period
        elif file_extension in [".mp4", ".avi"]:
            # Use imageio library to read mp4 and avi movies
            # Could also use OpenCV for reading these kinds of files
            # but the read performance of OpenCV is not as good as imageio
            movie = iio.get_reader(input_filename)

            # Read first frame of movie to get frame shape
            first_frame = movie.get_data(0)
            frame_width = first_frame.shape[1]
            frame_height = first_frame.shape[0]
            del first_frame  # Delete so it's not taking up memory space
            num_frames = movie.count_frames()

            # Read metadata to get movie fps, not sure if there's a better
            # way to do this with imageio
            metadata = movie.get_meta_data()
            if "fps" not in metadata:
                raise ToolException(
                    ExitStatus.IDPS_ERROR_0013,
                    f"No fps found in {file_extension} movie metadata",
                )
            movie_sampling_rate = float(metadata["fps"])
            movie_sampling_period = 1 / movie_sampling_rate

            # Set seek point to beginning of file before we start iterating frames
            # for more efficient read time.
            movie.set_image_index(0)
        elif file_extension in [".tiff", ".tif"]:
            image_stack = Image.open(input_filename)

            frame_width = image_stack.width
            frame_height = image_stack.height
            num_frames = image_stack.n_frames

            movie = ImageSequence.all_frames(image_stack)
            movie_sampling_rate = preview_max_sampling_rate
            movie_sampling_period = 1 / movie_sampling_rate
        else:
            raise ToolException(
                ExitStatus.IDPS_ERROR_0002,
                "Only isxd, isxb, mp4, avi, tiff, and tif movies are supported",
            )

    # Step 2: Determine preview duration, sampling rate, and resolution
    # based on movie properties.
    preview_sampling_rate = preview_max_sampling_rate
    if preview_max_sampling_rate > movie_sampling_rate:
        # Set preview sampling rate to movie sampling rate if it's
        # less than max preview sampling rate
        preview_sampling_rate = movie_sampling_rate
    preview_sampling_period = 1 / preview_sampling_rate

    movie_duration = num_frames * movie_sampling_period
    # Preview duration is either max duration or movie duration
    preview_duration = min(preview_max_duration * 60, movie_duration)
    preview_num_frames = math.ceil(preview_sampling_rate * preview_duration)
    # Maximum bit rate for previews in units of Kb/s.
    # This is passed to ffmpeg to ensure all preview file are within a max size limit.
    # The max file size determines the max bit rate.
    preview_max_bit_rate = int((preview_max_size * 1e3 * 8) / preview_duration)

    # Frame shape is represented as (num_rows, num_cols)
    # Resolution is represented as (width, height)
    # num_rows = height and num_cols = width
    # So flip dimensions of resolution to get frame shape
    preview_frame_shape = (
        preview_max_resolution[1],
        preview_max_resolution[0],
    )
    scaled_frame_shape = _transform_movie_to_preview_shape(
        movie_frame_shape=(frame_height, frame_width),
        preview_frame_shape=preview_frame_shape,
    )

    # Step 3: Initialize video writer for preview file
    # Use imageio to write compressed file using H.264 standard
    # https://imageio.readthedocs.io/en/v2.10.0/reference/_backends/imageio.plugins.ffmpeg.html
    writer = iio.get_writer(
        preview_filename,
        format="FFMPEG",  # Use ffmpeg library to write compressed file
        mode="I",  # "I" stands for series of images to write
        fps=preview_sampling_rate,
        codec="h264",  # Use H.264 since it's currently the most widely adopted
        # video compression standard. So it should be compatible with most browsers and user devices
        output_params=[
            "-crf",
            f"{preview_crf}",
            "-maxrate",
            f"{preview_max_bit_rate}K",
            "-bufsize",
            f"{int(preview_max_bit_rate / 2)}K",
        ],
        # Parameter bufsize needs to be specified in order for the max bit rate to be set appropiately.
        # FFMPEG docs suggest that a good general value for this param is half the max bit rate.
        macro_block_size=16,  # Size constraint for video. Width and height, must be divisible by this number.
        # If not divisible by this number imageio will tell ffmpeg to scale the image up to
        # the next closest size divisible by this number.  Most codecs are compatible with a
        # macroblock size of 16 (default). Even though this is the default value for this function
        # I'm leaving this here so others are aware in the future of why the resolution of the preview
        # may not exactly match the resolution of the movie after it's been scaled down to fit within
        # `preview_max_resolution`. It is possible to use a smaller value like 4, 2, or even 1,
        # but many players are not compatible with smaller values so I didn't want to take the risk.
        ffmpeg_log_level="error",  # ffmpeg can be quite chatty with warnings.
        # Setting to error in order to avoid cluttering logs.
    )

    # Step 4: Write frames to preview file
    # Often times one movie frame will appear on the boundary of two consecutive
    # preview frames. In order to prevent reading the same movie frame more than
    # once, keep track of the last movie frame that was read for the previous
    # preview frame that was processed in the loop.
    last_movie_frame_ind = None  # Index of last movie frame that was read for the previous preview frame
    last_movie_frame = None  # Frame data of the last movie frame that was read for the previous preview frame
    for preview_frame_ind in range(preview_num_frames):
        # Step 4.1: Find movie frames that occur within the current preview frame
        preview_frame_map = _map_movie_to_preview_frame_ind(
            preview_frame_ind=preview_frame_ind,
            preview_sampling_period=preview_sampling_period,
            movie_num_frames=num_frames,
            movie_sampling_period=movie_sampling_period,
        )

        # Step 4.2: Iterate through all movie frames that occur within the preview frame
        preview_frame = None  # Initialize preview frame to empty object
        num_mapped_movie_frames = len(
            preview_frame_map
        )  # Number of movie frames that occur within
        # the preview frame
        for mapped_movie_frame_ind, mapped_movie_frame in enumerate(
            preview_frame_map
        ):
            # Step 4.2.1: Unpack data in current entry of the previw frame map.
            # Preview frame map returns a frame index, and a floating point number
            # representng the contribution that the movie frame makes to the preview frame
            movie_frame_ind, movie_frame_contribution = mapped_movie_frame

            # Step 4.2.2: Get movie frame data
            # See if first frame of the movie sequence is equal to the last frame of the last movie sequence
            if (
                mapped_movie_frame_ind == 0
                and last_movie_frame is not None
                and movie_frame_ind == last_movie_frame_ind
            ):
                # Use cached frame data instead of re-reading from movie file
                movie_frame = last_movie_frame
            else:
                # Assert that the next movie frame read into memory is the next
                # successive frame in the movie. That is, no movie frames (within a particular duration)
                # are skipped when generating previews, and movie frames are processed in order.
                # This is important for mp4 and avi files because we read the next frame
                # in the file for those formats since it's more efficient than seeking
                # before reading each frame.
                assert last_movie_frame_ind is None or movie_frame_ind == (
                    last_movie_frame_ind + 1
                )

                # Read movie frame based on file type
                if is_image_sequence:
                    movie_frame = np.array(
                        Image.open(input_filename[movie_frame_ind])
                    )
                elif file_extension in [".isxd", ".isxb"]:
                    movie_frame = movie.get_frame_data(movie_frame_ind)
                elif file_extension in [".mp4", ".avi"]:
                    movie_frame = movie.get_next_data()
                elif file_extension in [".tiff", ".tif"]:
                    movie_frame = np.array(movie[movie_frame_ind])
                else:
                    raise ToolException(
                        ExitStatus.IDPS_ERROR_0002,
                        "Only isxd/isxb/mp4/avi/tiff/tif movies and tiff/tif image sequences are supported.",
                    )

                # Convert the frame to floating point in order to perform linear interpolation later
                movie_frame = movie_frame.astype(np.float64)

                # Keep track of last movie frame that was read from the file
                last_movie_frame_ind = movie_frame_ind

                # If this is the last movie frame in the sequence of movie frames that occur during
                # the preview frame, save a copy of the frame data because it will most likely
                # be used in the next preview frame as well
                if mapped_movie_frame_ind == (num_mapped_movie_frames - 1):
                    last_movie_frame = movie_frame.copy()

            # Step 4.2.3: Add the movie frame to the preview frame
            # Multiply the movie frame by the fraction of time it occurred within the preview frame
            # Preview frame is the linear interpolation of all frames that occurred durin its duration
            movie_frame *= movie_frame_contribution
            if preview_frame is None:
                # Initialize preview frame
                preview_frame = movie_frame
            else:
                # Add movie frame to existing preview frame
                preview_frame += movie_frame

        # Step 4.3: Write final preview frame to file
        if preview_frame is not None:
            # Resize the preview frame if it needs to scaled down to fit within the max resolution
            if scaled_frame_shape != (frame_height, frame_width):
                preview_frame = cv2.resize(
                    preview_frame,
                    (scaled_frame_shape[1], scaled_frame_shape[0]),
                )

            # Normalize the image and convert to 8 bit unsigned int data type
            # This data type is required for compressed video files.
            preview_frame = cv2.normalize(
                preview_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            writer.append_data(preview_frame)
        else:
            # This means no movie frames occurred throughout the duration of the preview frame
            # Exit the loop and close the preview file
            break

    writer.close()

    logger.info(
        f"Movie preview saved "
        f"({os.path.basename(preview_filename)}, "
        f"size: {get_file_size(preview_filename)})"
    )


def generate_caiman_workflow_previews(
    cellset_raw_filenames: List[str],
    cellset_denoised_filenames: List[str],
    eventset_filenames: List[str],
    original_input_indices: List[int],
    global_cellset_filename: str,
    input_movie_files: List[str],
    output_dir: str = None,
):
    """Generate previews for files produced by the CaImAn workflow.

    :param cellset_raw_filenames: path to the raw cell set files
    :param cellset_denoised_filenames: path to the denoised cell set files
    :param eventset_filenames: path to the event set files
    :param original_input_indices: original order of the input files prior to sorting
    :param global_cellset_filename: path to the global cell set in which all traces are concatenated
    :param input_movie_files: list of paths to the input movie files
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
        if len(input_movie_files) > 0:
            individual_movie_start_indices = [0]
            for f in input_movie_files:
                m = isx.Movie.read(f)
                individual_movie_start_indices.append(
                    individual_movie_start_indices[-1] + m.timing.num_samples
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

    # delete global cell set file
    if os.path.exists(global_cellset_filename):
        os.remove(global_cellset_filename)


def _plot_rigid_shifts(mc_obj, vertical_line_indices=None):
    """Plot rigid shifts.

    :param mc_obj: CaImAn motion correction object
    """
    num_frames = len(mc_obj.shifts_rig)
    x_shifts_rig, y_shifts_rig = list(zip(*mc_obj.shifts_rig))

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(x_shifts_rig, c="#1f77b4")
    ax[0].set_ylabel("x shift (pixels)")
    ax[0].set_xlim((0, num_frames))
    ax[0].set_title("Rigid Shifts")

    ax[1].plot(y_shifts_rig, c="#ff7f0e")
    ax[1].set_ylabel("y shift (pixels)")
    ax[1].set_xlim((0, num_frames))

    if vertical_line_indices is not None:
        for x in vertical_line_indices:
            ax[0].axvline(x=x, color="gray", ls="--", lw=1, alpha=0.3)
            ax[1].axvline(x=x, color="gray", ls="--", lw=1, alpha=0.3)

    plt.xlabel("frame")

    plt.tight_layout()
    rigid_shifts_preview_filename = "preview_rigid_shifts.svg"
    fig.savefig(
        rigid_shifts_preview_filename,
        dpi=300,
    )
    plt.close(fig)


def _plot_piecewise_rigid_shifts(mc_obj, vertical_line_indices=None):
    """Plot piecewise rigid shifts.

    :param mc_obj: CaImAn motion correction object
    """
    num_frames = len(mc_obj.shifts_rig)
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(mc_obj.x_shifts_els)
    ax[0].set_ylabel("x shift (pixels)")
    ax[0].set_xlim((0, num_frames))
    ax[0].set_title("Piecewise Rigid Shifts")

    ax[1].plot(mc_obj.y_shifts_els)
    ax[1].set_ylabel("y shift (pixels)")
    ax[1].set_xlim((0, num_frames))

    if vertical_line_indices is not None:
        for x in vertical_line_indices:
            ax[0].axvline(x=x, color="gray", ls="--", lw=1, alpha=0.3)
            ax[1].axvline(x=x, color="gray", ls="--", lw=1, alpha=0.3)

    plt.xlabel("frame")

    plt.tight_layout()
    piecewise_rigid_shifts_preview_filename = (
        "preview_piecewise_rigid_shifts.svg"
    )
    fig.savefig(
        piecewise_rigid_shifts_preview_filename,
        dpi=300,
    )
    plt.close(fig)


def generate_caiman_motion_corrected_previews(
    mc_movie_filenames: List[str],
    mc_obj,
    original_input_indices: List[int],
    frame_index_cutoffs: List[int],
    frame_rate: float,
    output_dir: str = None,
):
    """Generate previews for motion-corrected data produced by the CaImAn motion correction module.

    :param mc_movie_filenames: path to the motion-corrected movie files
    :param mc_obj: CaImAn motion correction object
    :param original_input_indices: original order of the input files prior to sorting
    :param frame_index_cutoffs: frame index cutoffs delineating the items in the series
    :param frame_rate: frame rate to use for the movie previews
    :param output_dir: path to the output directory
    """
    if output_dir is None:
        output_dir = os.getcwd()

    # movie previews
    for i in original_input_indices:
        try:
            movie_basename = os.path.splitext(
                os.path.basename(mc_movie_filenames[i])
            )[0]
            preview_filename = os.path.join(
                output_dir, f"preview_{movie_basename}.mp4"
            )

            file_ext = os.path.splitext(mc_movie_filenames[0])[1][1:]
            if file_ext.lower() in ["tiff", "tif"]:
                generate_movie_preview(
                    input_filename=mc_movie_filenames[i],
                    preview_filename=preview_filename,
                    preview_max_sampling_rate=frame_rate,
                )
            else:
                generate_movie_preview(
                    input_filename=mc_movie_filenames[i],
                    preview_filename=preview_filename,
                )
        except Exception as e:
            logger.warning(
                f"Preview could not be generated for file '{os.path.basename(mc_movie_filenames[i])}': {str(e)}"
            )
            logger.exception(e)

    # motion correction quality assessment previews
    if len(mc_movie_filenames) > 0:
        vertical_line_indices = frame_index_cutoffs[1:-1]
    else:
        vertical_line_indices = None

    _plot_rigid_shifts(
        mc_obj=mc_obj, vertical_line_indices=vertical_line_indices
    )
    if mc_obj.pw_rigid:
        _plot_piecewise_rigid_shifts(
            mc_obj=mc_obj, vertical_line_indices=vertical_line_indices
        )


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

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

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

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "initialization_images.svg"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)
    except Exception as e:
        logger.warning(
            f"Initialization images (pnr, correlation, search) could not be generated: {str(e)}"
        )
        correlation_image = None

    return correlation_image


def generate_local_correlation_image_preview(
    correlation_image,
    output_dir: str = None,
):
    """Generate preview for the local correlation image.
    :param correlation_image: local correlation image computed during CNMF-E initialization
    :param output_dir: path to the output directory
    """
    if output_dir is None:
        output_dir = os.getcwd()

    try:
        logger.info("Generating correlation image preview")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(correlation_image, cmap="gray")
        ax.axis("off")
        fig.tight_layout()
        plt.savefig(
            "local_corr_img_preview.png", bbox_inches="tight", pad_inches=0.1
        )

    except Exception as e:
        logger.warning(
            f"Correlation image preview could not be generated: {str(e)}"
        )
