import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox.utils.utilities import get_file_size
import logging

logger = logging.getLogger()


def generate_motion_correction_quality_assessment_data(
    mc_obj, mc_qc_filename, num_frames_per_movie
):
    """Generate CaImAn motion correction quality assessment.

    :param mc_obj: CaImAn motion correction object
    :param mc_qc_filename: path to the output motion correction quality control file
    :param num_frames_per_movie: list containing the number of frames in each input movie file
    """
    logger.info("Generating motion correction quality assessment data")

    # add x and y rigid shifts (per frame)
    qc_df = pd.DataFrame(
        mc_obj.shifts_rig, columns=["x_shifts_rig", "y_shifts_rig"]
    )

    # add frame number and reorder columns
    movie_indices = []
    movie_frame_indices = []
    series_frame_indices = list(range(len(qc_df)))
    for current_movie_index, n in enumerate(num_frames_per_movie):
        movie_indices.extend([current_movie_index] * n)
        movie_frame_indices.extend(list(range(n)))

    qc_df["movie_index"] = movie_indices
    qc_df["movie_frame_index"] = movie_frame_indices
    qc_df["series_frame_index"] = series_frame_indices
    qc_df = qc_df[
        [
            "movie_index",
            "movie_frame_index",
            "series_frame_index",
            "x_shifts_rig",
            "y_shifts_rig",
        ]
    ]

    # add x and y elastic shifts (per patch per frame)
    if mc_obj.pw_rigid:
        qc_df["x_shifts_els"] = mc_obj.x_shifts_els
        qc_df["y_shifts_els"] = mc_obj.y_shifts_els

    # save to disk
    qc_df.to_csv(mc_qc_filename, index=False)

    logger.info(
        f"Motion correction quality assessment data saved "
        f"({os.path.basename(mc_qc_filename)}, "
        f"size: {get_file_size(mc_qc_filename)})"
    )
