import os
import cv2
import shutil
import pytest
import isx
import json
import numpy as np
import pandas as pd
from PIL import Image
from toolbox.tools.caiman_isx_academic import motion_correction
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from toolbox.utils.exceptions import IdeasError


DATA_DIR = "/ideas/data"


@pytest.fixture(scope="session", autouse=True)
def use_spawn_not_fork():
    """
    Forcibly overrides the start method of the multiprocessing module to use the spawn method. This
    resolves a deadlock when starting the multiprocessing pool in `caiman.cluster.setup_cluster`,
    which is likely to happen under pytest as the fork method copies the entire memory of the
    parent process (pytest), including any possible locks that never get updated (as the child
    threads do not get copied).

    References:
     - https://pythonspeed.com/articles/python-multiprocessing/
    """
    from multiprocessing import set_start_method

    set_start_method("spawn", force=True)


@pytest.fixture(autouse=True)
def clean_up():
    """Clean up output directories and files for each test."""
    shutil.rmtree("caiman_data", ignore_errors=True)

    for f in [
        "output_metadata.json",
        "output_manifest.json",
    ]:
        if os.path.exists(f):
            os.remove(f)

    file_ext = [".isxd", ".png", ".mmap", ".mp4", ".csv"]
    for f in os.listdir("."):
        if os.path.splitext(f)[-1] in file_ext:
            os.remove(f)


def test_caiman_motion_correction_single_2p_tif_file_with_user_specified_params():
    """Verify that the CaImAn motion correction algorithm can be applied to a 2P tif
    movie using user-specified parameters.
    """
    input_movie_files = [os.path.join(DATA_DIR, "demoMovie.tif")]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=True,
        max_shifts=6,
        strides=48,
        overlaps=24,
        max_deviation_rigid=3,
        shifts_opencv=True,
        border_nan="copy",
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD file
    exp_num_frames = 2000
    exp_width = 80
    exp_height = 60
    exp_timing = isx.Timing(
        num_samples=exp_num_frames,
        period=isx.Duration.from_usecs(1.0 / 30.0 * 1e6),
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    mc_movie = isx.Movie.read("mc_movie.000.isxd")
    assert exp_timing == mc_movie.timing
    assert exp_spacing == mc_movie.spacing

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (2000, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata = act_metadata["mc_movie.000"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    exp_mc_movie_metadata = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 2000,
            "period": {"den": 1000000, "num": 33333},
            "start": {
                "secsSinceEpoch": {"den": 1, "num": 0},
                "utcOffset": 0,
            },
            "sampling_rate": 30.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 80, "y": 60},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {
                "x": {"den": 1, "num": 0},
                "y": {"den": 1, "num": 0},
            },
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata == act_mc_movie_metadata

    # motion correction quality control metadata
    exp_mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": -0.3,
            "max_x_rigid_shift": 0.2,
            "min_y_rigid_shift": -0.5,
            "max_y_rigid_shift": 0.5,
            "min_x_pw_rigid_shift": -0.2,
            "max_x_pw_rigid_shift": 0.2,
            "min_y_pw_rigid_shift": -0.8,
            "max_y_pw_rigid_shift": 1.0,
            "num_patches": 2,
        }
    }
    assert exp_mc_qc_metadata == act_mc_qc_metadata


def test_caiman_motion_correction_single_2p_tif_file_with_unused_json_params_file():
    """Verify that the CaImAn motion correction algorithm can be applied to a 2P tif
    movie WITHOUT using parameters obtained from a json file specified by the user.
    """
    input_movie_files = [os.path.join(DATA_DIR, "demoMovie.tif")]
    input_parameters_file = os.path.join(DATA_DIR, "params_normcorre.json")

    motion_correction(
        input_movie_files=input_movie_files,
        parameters_file=input_parameters_file,
        overwrite_analysis_table_params=False,
        pw_rigid=True,
        max_shifts=6,
        strides=48,
        overlaps=24,
        max_deviation_rigid=3,
        shifts_opencv=True,
        border_nan="copy",
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD file
    exp_num_frames = 2000
    exp_width = 80
    exp_height = 60
    exp_timing = isx.Timing(
        num_samples=exp_num_frames,
        period=isx.Duration.from_usecs(1.0 / 30.0 * 1e6),
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    mc_movie = isx.Movie.read("mc_movie.000.isxd")
    assert exp_timing == mc_movie.timing
    assert exp_spacing == mc_movie.spacing

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (2000, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata = act_metadata["mc_movie.000"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    exp_mc_movie_metadata = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 2000,
            "period": {"den": 1000000, "num": 33333},
            "start": {
                "secsSinceEpoch": {"den": 1, "num": 0},
                "utcOffset": 0,
            },
            "sampling_rate": 30.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 80, "y": 60},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {
                "x": {"den": 1, "num": 0},
                "y": {"den": 1, "num": 0},
            },
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata == act_mc_movie_metadata

    # motion correction quality control metadata
    exp_mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": -0.3,
            "max_x_rigid_shift": 0.2,
            "min_y_rigid_shift": -0.5,
            "max_y_rigid_shift": 0.5,
            "min_x_pw_rigid_shift": -0.2,
            "max_x_pw_rigid_shift": 0.2,
            "min_y_pw_rigid_shift": -0.8,
            "max_y_pw_rigid_shift": 1.0,
            "num_patches": 2,
        }
    }
    assert exp_mc_qc_metadata == act_mc_qc_metadata


def test_caiman_motion_correction_single_2p_tif_file_with_json_params_file():
    """Verify that the CaImAn motion correction algorithm can be applied to a 2P tif
    movie using parameters obtained from a json file specified by the user.
    """
    input_movie_files = [os.path.join(DATA_DIR, "demoMovie.tif")]
    input_parameters_file = [os.path.join(DATA_DIR, "params_normcorre.json")]

    motion_correction(
        input_movie_files=input_movie_files,
        parameters_file=input_parameters_file,
        overwrite_analysis_table_params=True,
        # params below should be overwritten by parameters file
        pw_rigid=True,
        max_shifts=6,
        strides=48,
        overlaps=24,
        max_deviation_rigid=3,
        shifts_opencv=True,
        border_nan="copy",
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD file
    exp_num_frames = 2000
    exp_width = 80
    exp_height = 60
    exp_timing = isx.Timing(
        num_samples=exp_num_frames,
        period=isx.Duration.from_usecs(1.0 / 10.0 * 1e6),
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    mc_movie = isx.Movie.read("mc_movie.000.isxd")
    assert exp_timing == mc_movie.timing
    assert exp_spacing == mc_movie.spacing

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (2000, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata = act_metadata["mc_movie.000"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    exp_mc_movie_metadata = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 2000,
            "period": {"den": 1000000, "num": 100000},
            "start": {
                "secsSinceEpoch": {"den": 1, "num": 0},
                "utcOffset": 0,
            },
            "sampling_rate": 10.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 80, "y": 60},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {
                "x": {"den": 1, "num": 0},
                "y": {"den": 1, "num": 0},
            },
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata == act_mc_movie_metadata

    # motion correction quality control metadata
    exp_mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": -0.4,
            "max_x_rigid_shift": 0.2,
            "min_y_rigid_shift": -0.4,
            "max_y_rigid_shift": 0.6,
            "min_x_pw_rigid_shift": -0.4,
            "max_x_pw_rigid_shift": 0.3,
            "min_y_pw_rigid_shift": -0.5,
            "max_y_pw_rigid_shift": 0.6,
            "num_patches": 2,
        }
    }
    assert exp_mc_qc_metadata == act_mc_qc_metadata


def test_caiman_motion_correction_1p_tif_file_pw_rigid_with_user_specified_params():
    """Verify that the CaImAn piecewise rigid motion correction algorithm can be applied
    to a 1P tif movie using user-specified parameters.
    """
    input_movie_files = [os.path.join(DATA_DIR, "data_endoscope.tif")]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=True,
        max_shifts=6,
        strides=30,
        overlaps=15,
        max_deviation_rigid=3,
        shifts_opencv=True,
        border_nan="copy",
        gSig_filt=3,
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD file
    exp_num_frames = 1000
    exp_width = 128
    exp_height = 128
    exp_timing = isx.Timing(
        num_samples=exp_num_frames,
        period=isx.Duration.from_usecs(1.0 / 30.0 * 1e6),
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    mc_movie = isx.Movie.read("mc_movie.000.isxd")
    assert exp_timing == mc_movie.timing
    assert exp_spacing == mc_movie.spacing

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (1000, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata = act_metadata["mc_movie.000"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    assert act_mc_movie_metadata["microscope"]["focus"] is None
    assert act_mc_movie_metadata["timingInfo"]["sampling_rate"] == 30.0

    # motion correction quality control metadata
    assert sorted(act_mc_qc_metadata["metrics"].keys()) == sorted(
        [
            "min_x_rigid_shift",
            "max_x_rigid_shift",
            "min_y_rigid_shift",
            "max_y_rigid_shift",
            "min_x_pw_rigid_shift",
            "max_x_pw_rigid_shift",
            "min_y_pw_rigid_shift",
            "max_y_pw_rigid_shift",
            "num_patches",
        ]
    )


def test_caiman_motion_correction_1p_tif_file_rigid_with_user_specified_params():
    """Verify that the CaImAn rigid motion correction algorithm can be applied
    to a 1P tif movie using user-specified parameters.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.tiff")
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=False,
        gSig_filt=5,
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD file
    exp_num_frames = 100
    exp_width = 108
    exp_height = 123
    exp_timing = isx.Timing(
        num_samples=exp_num_frames,
        period=isx.Duration.from_usecs(1.0 / 30.0 * 1e6),
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    mc_movie = isx.Movie.read("mc_movie.000.isxd")
    assert exp_timing == mc_movie.timing
    assert exp_spacing == mc_movie.spacing

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (100, 5)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata = act_metadata["mc_movie.000"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    assert act_mc_movie_metadata["microscope"]["focus"] is None
    assert act_mc_movie_metadata["timingInfo"]["sampling_rate"] == 30.0

    # motion correction quality control metadata
    assert sorted(act_mc_qc_metadata["metrics"].keys()) == sorted(
        [
            "min_x_rigid_shift",
            "max_x_rigid_shift",
            "min_y_rigid_shift",
            "max_y_rigid_shift",
        ]
    )


def test_caiman_motion_correction_single_isxd_file_with_user_specified_params():
    """Verify that the CaImAn motion correction algorithm can be applied to
    a ISXD movie using user-specified parameters.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd")
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=True,
        max_shifts=4,
        strides=37,
        overlaps=17,
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD file
    exp_num_frames = 100
    exp_width = 108
    exp_height = 123
    exp_timing = isx.Timing(
        num_samples=exp_num_frames,
        period=isx.Duration.from_usecs(1.0 / 10.0 * 1e6),
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    mc_movie = isx.Movie.read("mc_movie.000.isxd")
    assert exp_timing == mc_movie.timing
    assert exp_spacing.num_pixels == mc_movie.spacing.num_pixels

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (100, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )


def test_caiman_motion_correction_single_avi_file_with_user_specified_params():
    """Verify that the CaImAn motion correction algorithm can be applied to
    an AVI movie using user-specified parameters.
    """
    input_movie_files = [os.path.join(DATA_DIR, "movie_part2_108x122x126.avi")]

    motion_correction(
        input_movie_files=input_movie_files, output_movie_format="isxd"
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD file
    exp_num_frames = 126
    exp_width = 108
    exp_height = 122
    exp_timing = isx.Timing(
        num_samples=exp_num_frames,
        period=isx.Duration.from_usecs(1.0 / 20.0 * 1e6),
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    mc_movie = isx.Movie.read("mc_movie.000.isxd")
    assert exp_timing == mc_movie.timing
    assert exp_spacing == mc_movie.spacing

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (126, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )


def test_caiman_motion_correction_isxd_movie_series_piecewise_rigid():
    """Verify that the CaImAn piecewise rigid motion correction algorithm
    can be applied to an ISXD movie series using user-specified parameters.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd"),
        os.path.join(DATA_DIR, "movie_part2_108x123x63.isxd"),
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=True,
        strides=40,
        overlaps=21,
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "mc_movie.001.isxd",
        "preview_mc_movie.000.mp4",
        "preview_mc_movie.001.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD files
    exp_width = 108
    exp_height = 123
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))
    exp_num_frames_part0 = 100
    exp_num_frames_part1 = 63

    mc_movie_part0 = isx.Movie.read("mc_movie.000.isxd")
    assert exp_num_frames_part0 == mc_movie_part0.timing.num_samples
    assert exp_spacing.num_pixels == mc_movie_part0.spacing.num_pixels

    mc_movie_part1 = isx.Movie.read("mc_movie.001.isxd")
    assert exp_num_frames_part1 == mc_movie_part1.timing.num_samples
    assert exp_spacing.num_pixels == mc_movie_part1.spacing.num_pixels

    input_movie_part0 = isx.Movie.read(input_movie_files[0])
    assert input_movie_part0.spacing == mc_movie_part0.spacing
    assert input_movie_part0.timing == mc_movie_part0.timing
    del input_movie_part0

    input_movie_part1 = isx.Movie.read(input_movie_files[1])
    assert input_movie_part1.spacing == mc_movie_part1.spacing
    assert input_movie_part1.timing == mc_movie_part1.timing
    del input_movie_part1

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (163, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * 100 + [1] * 63
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(100)
    ) + list(range(63))
    assert act_qc_data["series_frame_index"].to_list() == list(range(163))

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata0 = act_metadata["mc_movie.000"]
        act_mc_movie_metadata1 = act_metadata["mc_movie.001"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    exp_mc_movie_metadata0 = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 100,
            "period": {"den": 1000, "num": 100},
            "start": {"secsSinceEpoch": {"den": 1, "num": 0}, "utcOffset": 0},
            "sampling_rate": 10.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 108, "y": 123},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {
                "x": {"den": 1, "num": 60},
                "y": {"den": 1, "num": 15},
            },
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata0 == act_mc_movie_metadata0

    exp_mc_movie_metadata1 = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 63,
            "period": {"den": 1000, "num": 100},
            "start": {
                "secsSinceEpoch": {"den": 1000, "num": 30100},
                "utcOffset": 0,
            },
            "sampling_rate": 10.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 108, "y": 123},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {
                "x": {"den": 1, "num": 60},
                "y": {"den": 1, "num": 15},
            },
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata1 == act_mc_movie_metadata1

    # motion correction quality control metadata
    exp_mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": -0.0,
            "max_x_rigid_shift": -0.0,
            "min_y_rigid_shift": -0.1,
            "max_y_rigid_shift": 0.1,
            "min_x_pw_rigid_shift": -0.1,
            "max_x_pw_rigid_shift": 0.1,
            "min_y_pw_rigid_shift": -0.2,
            "max_y_pw_rigid_shift": 0.2,
            "num_patches": 9,
        }
    }
    assert exp_mc_qc_metadata == act_mc_qc_metadata


def test_caiman_motion_correction_isxd_movie_series_rigid():
    """Verify that the CaImAn rigid motion correction algorithm can be applied to
    an ISXD movie series using user-specified parameters.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part2_108x123x63.isxd"),
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd"),
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=False,
        strides=50,
        overlaps=20,
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "mc_movie.001.isxd",
        "preview_mc_movie.000.mp4",
        "preview_mc_movie.001.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD files
    exp_width = 108
    exp_height = 123
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))
    exp_num_frames_part0 = 100
    exp_num_frames_part1 = 63

    mc_movie_part0 = isx.Movie.read("mc_movie.001.isxd")
    assert exp_num_frames_part0 == mc_movie_part0.timing.num_samples
    assert exp_spacing.num_pixels == mc_movie_part0.spacing.num_pixels

    mc_movie_part1 = isx.Movie.read("mc_movie.000.isxd")
    assert exp_num_frames_part1 == mc_movie_part1.timing.num_samples
    assert exp_spacing.num_pixels == mc_movie_part1.spacing.num_pixels

    input_movie_part0 = isx.Movie.read(input_movie_files[1])
    assert input_movie_part0.spacing == mc_movie_part0.spacing
    assert input_movie_part0.timing == mc_movie_part0.timing
    del input_movie_part0

    input_movie_part1 = isx.Movie.read(input_movie_files[0])
    assert input_movie_part1.spacing == mc_movie_part1.spacing
    assert input_movie_part1.timing == mc_movie_part1.timing
    del input_movie_part1

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (163, 5)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * 100 + [1] * 63
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(100)
    ) + list(range(63))
    assert act_qc_data["series_frame_index"].to_list() == list(range(163))

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata0 = act_metadata["mc_movie.001"]
        act_mc_movie_metadata1 = act_metadata["mc_movie.000"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    exp_mc_movie_metadata0 = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 100,
            "period": {"den": 1000, "num": 100},
            "start": {"secsSinceEpoch": {"den": 1, "num": 0}, "utcOffset": 0},
            "sampling_rate": 10.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 108, "y": 123},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {
                "x": {"den": 1, "num": 60},
                "y": {"den": 1, "num": 15},
            },
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata0 == act_mc_movie_metadata0

    exp_mc_movie_metadata1 = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 63,
            "period": {"den": 1000, "num": 100},
            "start": {
                "secsSinceEpoch": {"den": 1000, "num": 30100},
                "utcOffset": 0,
            },
            "sampling_rate": 10.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 108, "y": 123},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {
                "x": {"den": 1, "num": 60},
                "y": {"den": 1, "num": 15},
            },
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata1 == act_mc_movie_metadata1

    # motion correction quality control metadata
    exp_mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": -0.0,
            "max_x_rigid_shift": -0.0,
            "min_y_rigid_shift": -0.1,
            "max_y_rigid_shift": 0.1,
        }
    }
    assert exp_mc_qc_metadata == act_mc_qc_metadata


def test_caiman_motion_correction_tiff_movie_series():
    """Verify that the CaImAn motion correction algorithm can be applied to
    a TIFF movie series using user-specified parameters.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.tiff"),
        os.path.join(DATA_DIR, "movie_part2_108x123x63.tiff"),
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=True,
        fr=15,
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "mc_movie.001.isxd",
        "preview_mc_movie.000.mp4",
        "preview_mc_movie.001.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD files
    exp_width = 108
    exp_height = 123
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))
    exp_num_frames_part0 = 100
    exp_num_frames_part1 = 63

    exp_timing0 = isx.Timing(
        num_samples=exp_num_frames_part0,
        period=isx.Duration.from_usecs(1.0 / 15.0 * 1e6),
    )
    exp_timing1 = isx.Timing(
        num_samples=exp_num_frames_part1,
        period=isx.Duration.from_usecs(1.0 / 15.0 * 1e6),
    )

    mc_movie_part0 = isx.Movie.read("mc_movie.000.isxd")
    assert exp_num_frames_part0 == mc_movie_part0.timing.num_samples
    assert exp_spacing.num_pixels == mc_movie_part0.spacing.num_pixels

    mc_movie_part1 = isx.Movie.read("mc_movie.001.isxd")
    assert exp_num_frames_part1 == mc_movie_part1.timing.num_samples
    assert exp_spacing.num_pixels == mc_movie_part1.spacing.num_pixels

    input_movie_part0 = isx.Movie.read(input_movie_files[0])
    assert input_movie_part0.spacing == mc_movie_part0.spacing
    assert exp_timing0 == mc_movie_part0.timing
    del input_movie_part0

    input_movie_part1 = isx.Movie.read(input_movie_files[1])
    assert input_movie_part1.spacing == mc_movie_part1.spacing
    assert exp_timing1 == mc_movie_part1.timing
    del input_movie_part1

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (163, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * 100 + [1] * 63
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(100)
    ) + list(range(63))
    assert act_qc_data["series_frame_index"].to_list() == list(range(163))

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata0 = act_metadata["mc_movie.000"]
        act_mc_movie_metadata1 = act_metadata["mc_movie.001"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    exp_mc_movie_metadata0 = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 100,
            "period": {"den": 1000000, "num": 66666},
            "start": {"secsSinceEpoch": {"den": 1, "num": 0}, "utcOffset": 0},
            "sampling_rate": 15.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 108, "y": 123},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata0 == act_mc_movie_metadata0

    exp_mc_movie_metadata1 = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 63,
            "period": {"den": 1000000, "num": 66666},
            "start": {"secsSinceEpoch": {"den": 1, "num": 0}, "utcOffset": 0},
            "sampling_rate": 15.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 108, "y": 123},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata1 == act_mc_movie_metadata1

    # motion correction quality control metadata
    exp_mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": -0.0,
            "max_x_rigid_shift": -0.0,
            "min_y_rigid_shift": -0.1,
            "max_y_rigid_shift": 0.1,
            "min_x_pw_rigid_shift": -0.1,
            "max_x_pw_rigid_shift": 0.1,
            "min_y_pw_rigid_shift": -0.1,
            "max_y_pw_rigid_shift": 0.2,
            "num_patches": 6,
        }
    }
    assert exp_mc_qc_metadata == act_mc_qc_metadata


def test_caiman_motion_correction_avi_movie_series():
    """Verify that the CaImAn motion correction algorithm can be applied to
    an AVI movie series using user-specified parameters.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x122x200.avi"),
        os.path.join(DATA_DIR, "movie_part2_108x122x126.avi"),
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=True,
        fr=10.0,
        output_movie_format="isxd",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.isxd",
        "mc_movie.001.isxd",
        "preview_mc_movie.000.mp4",
        "preview_mc_movie.001.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output ISXD files
    exp_width = 108
    exp_height = 122
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))
    exp_num_frames_part0 = 200
    exp_num_frames_part1 = 126

    exp_timing0 = isx.Timing(
        num_samples=exp_num_frames_part0,
        period=isx.Duration.from_usecs(1.0 / 10.0 * 1e6),
    )
    exp_timing1 = isx.Timing(
        num_samples=exp_num_frames_part1,
        period=isx.Duration.from_usecs(1.0 / 10.0 * 1e6),
    )

    mc_movie_part0 = isx.Movie.read("mc_movie.000.isxd")
    assert exp_num_frames_part0 == mc_movie_part0.timing.num_samples
    assert exp_spacing.num_pixels == mc_movie_part0.spacing.num_pixels

    mc_movie_part1 = isx.Movie.read("mc_movie.001.isxd")
    assert exp_num_frames_part1 == mc_movie_part1.timing.num_samples
    assert exp_spacing.num_pixels == mc_movie_part1.spacing.num_pixels

    # compare input/output movies - part 0
    input_movie_cap0 = cv2.VideoCapture(input_movie_files[0])
    input_movie_fps0 = input_movie_cap0.get(cv2.CAP_PROP_FPS)
    output_movie_fps0 = 1.0 / mc_movie_part0.timing.period.secs_float
    assert input_movie_fps0 == output_movie_fps0

    input_movie_frame_count0 = int(
        input_movie_cap0.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    output_movie_frame_count0 = mc_movie_part0.timing.num_samples
    assert input_movie_frame_count0 == output_movie_frame_count0

    input_movie_frame_height0 = int(
        input_movie_cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    output_movie_frame_height0 = mc_movie_part0.spacing.num_pixels[0]
    assert input_movie_frame_height0 == output_movie_frame_height0

    input_movie_frame_width0 = int(
        input_movie_cap0.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    output_movie_frame_width0 = mc_movie_part0.spacing.num_pixels[1]
    assert input_movie_frame_width0 == output_movie_frame_width0

    del input_movie_cap0

    # compare input/output movies - part 1
    input_movie_cap1 = cv2.VideoCapture(input_movie_files[1])
    input_movie_fps1 = input_movie_cap1.get(cv2.CAP_PROP_FPS)
    output_movie_fps1 = 1.0 / mc_movie_part1.timing.period.secs_float
    assert input_movie_fps1 == output_movie_fps1

    input_movie_frame_count1 = int(
        input_movie_cap1.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    output_movie_frame_count1 = mc_movie_part1.timing.num_samples
    assert input_movie_frame_count1 == output_movie_frame_count1

    input_movie_frame_height1 = int(
        input_movie_cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    output_movie_frame_height1 = mc_movie_part1.spacing.num_pixels[0]
    assert input_movie_frame_height1 == output_movie_frame_height1

    input_movie_frame_width1 = int(
        input_movie_cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    output_movie_frame_width1 = mc_movie_part1.spacing.num_pixels[1]
    assert input_movie_frame_width1 == output_movie_frame_width1

    del input_movie_cap1

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (326, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * 200 + [1] * 126
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(200)
    ) + list(range(126))
    assert act_qc_data["series_frame_index"].to_list() == list(range(326))

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata0 = act_metadata["mc_movie.000"]
        act_mc_movie_metadata1 = act_metadata["mc_movie.001"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    exp_mc_movie_metadata0 = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 200,
            "period": {"den": 1000, "num": 50},
            "start": {"secsSinceEpoch": {"den": 1, "num": 0}, "utcOffset": 0},
            "sampling_rate": 20.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 108, "y": 122},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata0 == act_mc_movie_metadata0

    exp_mc_movie_metadata1 = {
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 126,
            "period": {"den": 1000, "num": 50},
            "start": {"secsSinceEpoch": {"den": 1, "num": 0}, "utcOffset": 0},
            "sampling_rate": 20.0,
        },
        "spacingInfo": {
            "numPixels": {"x": 108, "y": 122},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata1 == act_mc_movie_metadata1

    # motion correction quality control metadata
    exp_mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": -0.0,
            "max_x_rigid_shift": 0.7,
            "min_y_rigid_shift": -0.1,
            "max_y_rigid_shift": 0.7,
            "min_x_pw_rigid_shift": -0.1,
            "max_x_pw_rigid_shift": 0.7,
            "min_y_pw_rigid_shift": -0.1,
            "max_y_pw_rigid_shift": 0.7,
            "num_patches": 6,
        }
    }
    assert exp_mc_qc_metadata == act_mc_qc_metadata


def test_caiman_motion_correction_single_input_isxd_file_single_output_tiff_file():
    """Verify that the CaImAn motion correction algorithm can be applied to
    a ISXD movie and produce a corresponding TIFF output file.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd")
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=True,
        max_shifts=4,
        strides=37,
        overlaps=17,
        output_movie_format="tiff",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.tiff",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        "preview_piecewise_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output TIFF file
    act_image_stack = Image.open("mc_movie.000.tiff")
    assert act_image_stack.width == 108
    assert act_image_stack.height == 123
    assert act_image_stack.n_frames == 100

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (100, 7)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
        "x_shifts_els",
        "y_shifts_els",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )


def test_caiman_motion_correction_single_input_isxd_file_single_output_avi_file():
    """Verify that the CaImAn motion correction algorithm can be applied to
    a ISXD movie and produce a corresponding AVI output file.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd")
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=False,
        max_shifts=4,
        strides=37,
        overlaps=17,
        output_movie_format="avi",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.avi",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output AVI file
    cap = cv2.VideoCapture("mc_movie.000.avi")
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 108
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 122
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 100
    del cap

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (100, 5)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )


def test_caiman_motion_correction_single_tiff_file_single_output_avi_file():
    """Verify that the CaImAn motion correction algorithm can be applied to
    a TIFF movie and produce a corresponding AVI output file.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.tiff")
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=False,
        max_shifts=4,
        strides=37,
        overlaps=17,
        output_movie_format="avi",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.avi",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output AVI file
    cap = cv2.VideoCapture("mc_movie.000.avi")
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 108
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 122
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 100
    del cap

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (100, 5)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )


def test_caiman_motion_correction_single_avi_file_single_output_tiff_file():
    """Verify that the CaImAn motion correction algorithm can be applied to
    a AVI movie and produce a corresponding TIFF output file.
    """
    input_movie_files = [os.path.join(DATA_DIR, "movie_part1_108x122x200.avi")]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=False,
        max_shifts=4,
        strides=37,
        overlaps=17,
        output_movie_format="tiff",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.tiff",
        "preview_mc_movie.000.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output TIFF file
    act_image_stack = Image.open("mc_movie.000.tiff")
    assert act_image_stack.width == 108
    assert act_image_stack.height == 122
    assert act_image_stack.n_frames == 200

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (200, 5)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * len(act_qc_data)
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )
    assert act_qc_data["series_frame_index"].to_list() == list(
        range(len(act_qc_data))
    )


def test_caiman_motion_correction_tiff_movie_series_to_avi_output():
    """Verify that the CaImAn motion correction algorithm can be applied to
    a TIFF movie series and produce corresponding AVI output movies.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.tiff"),
        os.path.join(DATA_DIR, "movie_part2_108x123x63.tiff"),
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=False,
        fr=15,
        output_movie_format="avi",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.avi",
        "mc_movie.001.avi",
        "preview_mc_movie.000.mp4",
        "preview_mc_movie.001.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output AVI files
    cap0 = cv2.VideoCapture("mc_movie.000.avi")
    assert int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)) == 108
    assert int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 122
    assert int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) == 100
    del cap0

    cap1 = cv2.VideoCapture("mc_movie.001.avi")
    assert int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)) == 108
    assert int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 122
    assert int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) == 63
    del cap1

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (163, 5)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * 100 + [1] * 63
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(100)
    ) + list(range(63))
    assert act_qc_data["series_frame_index"].to_list() == list(range(163))

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata0 = act_metadata["mc_movie.000"]
        act_mc_movie_metadata1 = act_metadata["mc_movie.001"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    exp_mc_movie_metadata0 = {
        "timingInfo": {"numTimes": 100, "sampling_rate": 15},
        "spacingInfo": {"numPixels": {"x": 122, "y": 108}},
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata0 == act_mc_movie_metadata0

    exp_mc_movie_metadata1 = {
        "timingInfo": {"numTimes": 63, "sampling_rate": 15},
        "spacingInfo": {"numPixels": {"x": 122, "y": 108}},
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata1 == act_mc_movie_metadata1

    # motion correction quality control metadata
    exp_mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": -0.0,
            "max_x_rigid_shift": -0.0,
            "min_y_rigid_shift": -0.1,
            "max_y_rigid_shift": 0.1,
        }
    }
    assert exp_mc_qc_metadata == act_mc_qc_metadata


def test_caiman_motion_correction_avi_movie_series_to_tiff_output():
    """Verify that the CaImAn motion correction algorithm can be applied to
    an AVI movie series and produce corresponding TIFF output movies.
    """
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x122x200.avi"),
        os.path.join(DATA_DIR, "movie_part2_108x122x126.avi"),
    ]

    motion_correction(
        input_movie_files=input_movie_files,
        pw_rigid=False,
        fr=10,  # will be overriden by actual file frame rate (20 fps)
        output_movie_format="tiff",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # motion-corrected movie
        "mc_movie.000.tiff",
        "mc_movie.001.tiff",
        "preview_mc_movie.000.mp4",
        "preview_mc_movie.001.mp4",
        # motion correction quality control data
        "mc_qc_data.csv",
        "preview_rigid_shifts.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate output AVI files
    act_image_stack0 = Image.open("mc_movie.000.tiff")
    assert act_image_stack0.width == 108
    assert act_image_stack0.height == 122
    assert act_image_stack0.n_frames == 200

    act_image_stack1 = Image.open("mc_movie.001.tiff")
    assert act_image_stack1.width == 108
    assert act_image_stack1.height == 122
    assert act_image_stack1.n_frames == 126

    # validate output quality control data
    act_qc_data = pd.read_csv("mc_qc_data.csv")
    assert act_qc_data.shape == (326, 5)
    assert list(act_qc_data.columns) == [
        "movie_index",
        "movie_frame_index",
        "series_frame_index",
        "x_shifts_rig",
        "y_shifts_rig",
    ]
    assert act_qc_data["movie_index"].to_list() == [0] * 200 + [1] * 126
    assert act_qc_data["movie_frame_index"].to_list() == list(
        range(200)
    ) + list(range(126))
    assert act_qc_data["series_frame_index"].to_list() == list(range(326))

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_mc_movie_metadata0 = act_metadata["mc_movie.000"]
        act_mc_movie_metadata1 = act_metadata["mc_movie.001"]
        act_mc_qc_metadata = act_metadata["mc_qc_data"]

    # motion-corrected movie metadata
    exp_mc_movie_metadata0 = {
        "timingInfo": {"numTimes": 200, "sampling_rate": 20.0},
        "spacingInfo": {"numPixels": {"x": 108, "y": 122, "z": 200}},
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata0 == act_mc_movie_metadata0

    exp_mc_movie_metadata1 = {
        "timingInfo": {"numTimes": 126, "sampling_rate": 20.0},
        "spacingInfo": {"numPixels": {"x": 108, "y": 122, "z": 126}},
        "microscope": {"focus": None},
    }
    assert exp_mc_movie_metadata1 == act_mc_movie_metadata1

    # motion correction quality control metadata
    exp_mc_qc_metadata = {
        "metrics": {
            "min_x_rigid_shift": -0.0,
            "max_x_rigid_shift": 0.7,
            "min_y_rigid_shift": -0.1,
            "max_y_rigid_shift": 0.7,
        }
    }
    assert exp_mc_qc_metadata == act_mc_qc_metadata
