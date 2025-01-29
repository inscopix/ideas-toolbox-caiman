import os
import cv2
import shutil
import pytest
import isx
import json
import numpy as np
import pandas as pd
from PIL import Image
from toolbox.tools.caiman_isx_academic import spike_extraction
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


def test_caiman_spike_extraction_single_isxd_file():
    """Verify that the CaImAn spike extraction correctly identifies spikes in a single ISXD cellset."""
    input_cellset_files = [os.path.join(DATA_DIR, "cellset_series_part1.isxd")]

    spike_extraction(
        input_cellset_files=input_cellset_files,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.png",
        "footprints_cellset_denoised.000.png",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # define expected values
    exp_num_frames = 100
    exp_width = 21
    exp_height = 21
    exp_num_cells = 6

    input_cellset = isx.CellSet.read(input_cellset_files[0])
    exp_timing = input_cellset.timing
    exp_spacing = input_cellset.spacing
    exp_cell_statuses = [
        input_cellset.get_cell_status(i)
        for i in range(input_cellset.num_cells)
    ]
    exp_cell_names = [
        input_cellset.get_cell_name(i) for i in range(input_cellset.num_cells)
    ]
    assert input_cellset.num_cells == exp_num_cells

    # validate output denoised cell set
    denoised_cellset = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset.num_cells == exp_num_cells
    assert denoised_cellset.timing.num_samples == exp_num_frames
    assert denoised_cellset.spacing.num_pixels[0] == exp_height
    assert denoised_cellset.spacing.num_pixels[1] == exp_width

    act_denoised_cellset_names = np.array(
        [
            denoised_cellset.get_cell_name(j)
            for j in range(denoised_cellset.num_cells)
        ]
    )
    assert (exp_cell_names == act_denoised_cellset_names).all()

    act_denoised_cellset_statuses = np.array(
        [
            denoised_cellset.get_cell_status(i)
            for i in range(denoised_cellset.num_cells)
        ]
    )
    assert (exp_cell_statuses == act_denoised_cellset_statuses).all()

    assert exp_timing == denoised_cellset.timing
    assert exp_spacing == denoised_cellset.spacing

    # validate output event set
    eventset = isx.EventSet.read("neural_events.000.isxd")
    assert eventset.num_cells == exp_num_cells
    assert eventset.timing.num_samples == exp_num_frames
    assert exp_timing == eventset.timing

    act_eventset_names = np.array(
        [eventset.get_cell_name(j) for j in range(eventset.num_cells)]
    )
    assert (exp_cell_names == act_eventset_names).all()


def test_caiman_spike_extraction_isxd_movie_series():
    """Verify that the CaImAn spike extraction correctly identifies spikes in an ISXD cellset series."""
    input_cellset_files = [
        os.path.join(DATA_DIR, "cellset_series_part1.isxd"),
        os.path.join(DATA_DIR, "cellset_series_part2.isxd"),
    ]

    spike_extraction(
        input_cellset_files=input_cellset_files,
        bl="auto",
        c1="auto",
        g="auto",
        sn="auto",
        p=2,
        method_deconvolution="oasis",
        bas_nonneg=True,
        noise_method="logmexp",
        noise_range="0.25,0.5",
        s_min=None,
        optimize_g=False,
        fudge_factor=0.96,
        lags=5,
        solvers="ECOS,SCS",
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # denoised cell sets
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.png",
        "footprints_cellset_denoised.000.png",
        "cellset_denoised.001.isxd",
        "traces_cellset_denoised.001.png",
        "footprints_cellset_denoised.001.png",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.png",
        "neural_events.001.isxd",
        "preview_neural_events.001.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # define expected values
    exp_num_frames_part0 = 100
    exp_num_frames_part1 = 95
    exp_width = 21
    exp_height = 21
    exp_num_cells = 6

    input_cellset0 = isx.CellSet.read(input_cellset_files[0])
    exp_timing0 = input_cellset0.timing
    exp_spacing0 = input_cellset0.spacing
    exp_cell_statuses = [
        input_cellset0.get_cell_status(i)
        for i in range(input_cellset0.num_cells)
    ]
    exp_cell_names = [
        input_cellset0.get_cell_name(i)
        for i in range(input_cellset0.num_cells)
    ]
    assert input_cellset0.num_cells == exp_num_cells

    input_cellset1 = isx.CellSet.read(input_cellset_files[1])
    exp_timing1 = input_cellset1.timing
    exp_spacing1 = input_cellset1.spacing
    assert input_cellset1.num_cells == exp_num_cells
    input_cell_statuses1 = np.array(
        [
            input_cellset1.get_cell_status(i)
            for i in range(input_cellset1.num_cells)
        ]
    )
    input_cell_names1 = np.array(
        [
            input_cellset1.get_cell_name(i)
            for i in range(input_cellset1.num_cells)
        ]
    )
    assert (exp_cell_statuses == input_cell_statuses1).all()
    assert (exp_cell_names == input_cell_names1).all()

    # validate output denoised cell sets
    denoised_cellset0 = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset0.num_cells == exp_num_cells
    assert denoised_cellset0.timing.num_samples == exp_num_frames_part0
    assert denoised_cellset0.spacing.num_pixels[0] == exp_height
    assert denoised_cellset0.spacing.num_pixels[1] == exp_width
    act_denoised_cellset_names0 = np.array(
        [
            denoised_cellset0.get_cell_name(j)
            for j in range(denoised_cellset0.num_cells)
        ]
    )
    assert (exp_cell_names == act_denoised_cellset_names0).all()
    act_denoised_cellset_statuses0 = np.array(
        [
            denoised_cellset0.get_cell_status(i)
            for i in range(denoised_cellset0.num_cells)
        ]
    )
    assert (exp_cell_statuses == act_denoised_cellset_statuses0).all()
    assert exp_timing0 == denoised_cellset0.timing
    assert exp_spacing0 == denoised_cellset0.spacing

    denoised_cellset1 = isx.CellSet.read("cellset_denoised.001.isxd")
    assert denoised_cellset1.num_cells == exp_num_cells
    assert denoised_cellset1.timing.num_samples == exp_num_frames_part1
    assert denoised_cellset1.spacing.num_pixels[0] == exp_height
    assert denoised_cellset1.spacing.num_pixels[1] == exp_width
    act_denoised_cellset_names1 = np.array(
        [
            denoised_cellset1.get_cell_name(j)
            for j in range(denoised_cellset1.num_cells)
        ]
    )
    assert (exp_cell_names == act_denoised_cellset_names1).all()
    act_denoised_cellset_statuses1 = np.array(
        [
            denoised_cellset1.get_cell_status(i)
            for i in range(denoised_cellset1.num_cells)
        ]
    )
    assert (exp_cell_statuses == act_denoised_cellset_statuses1).all()
    assert exp_timing1 == denoised_cellset1.timing
    assert exp_spacing1 == denoised_cellset1.spacing

    # validate output event sets
    eventset0 = isx.EventSet.read("neural_events.000.isxd")
    assert eventset0.num_cells == exp_num_cells
    assert eventset0.timing.num_samples == exp_num_frames_part0
    assert exp_timing0 == eventset0.timing
    act_eventset_names0 = np.array(
        [eventset0.get_cell_name(j) for j in range(eventset0.num_cells)]
    )
    assert (exp_cell_names == act_eventset_names0).all()

    eventset1 = isx.EventSet.read("neural_events.001.isxd")
    assert eventset1.num_cells == exp_num_cells
    assert eventset1.timing.num_samples == exp_num_frames_part1
    assert exp_timing1 == eventset1.timing
    act_eventset_names1 = np.array(
        [eventset1.get_cell_name(j) for j in range(eventset1.num_cells)]
    )
    assert (exp_cell_names == act_eventset_names1).all()


def test_caiman_spike_extraction_isxd_movie_series_with_unordered_inputs():
    """Verify that the CaImAn spike extraction correctly identifies spikes in an ISXD cellset series
    when the input files are not chronologically sorted.
    """
    input_cellset_files = [
        os.path.join(DATA_DIR, "cellset_series_part2.isxd"),
        os.path.join(DATA_DIR, "cellset_series_part1.isxd"),
    ]

    spike_extraction(
        input_cellset_files=input_cellset_files,
        p=1,
        method_deconvolution="oasis",
        bas_nonneg=False,
        noise_method="logmexp",
        noise_range="0.25,0.5",
        optimize_g=False,
        fudge_factor=1.0,
        lags=4,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # denoised cell sets
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.png",
        "footprints_cellset_denoised.000.png",
        "cellset_denoised.001.isxd",
        "traces_cellset_denoised.001.png",
        "footprints_cellset_denoised.001.png",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.png",
        "neural_events.001.isxd",
        "preview_neural_events.001.png",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # define expected values
    exp_num_frames_part0 = 95
    exp_num_frames_part1 = 100
    exp_width = 21
    exp_height = 21
    exp_num_cells = 6

    input_cellset0 = isx.CellSet.read(input_cellset_files[0])
    exp_timing0 = input_cellset0.timing
    exp_spacing0 = input_cellset0.spacing
    exp_cell_statuses = [
        input_cellset0.get_cell_status(i)
        for i in range(input_cellset0.num_cells)
    ]
    exp_cell_names = [
        input_cellset0.get_cell_name(i)
        for i in range(input_cellset0.num_cells)
    ]
    assert input_cellset0.num_cells == exp_num_cells

    input_cellset1 = isx.CellSet.read(input_cellset_files[1])
    exp_timing1 = input_cellset1.timing
    exp_spacing1 = input_cellset1.spacing
    assert input_cellset1.num_cells == exp_num_cells
    input_cell_statuses1 = np.array(
        [
            input_cellset1.get_cell_status(i)
            for i in range(input_cellset1.num_cells)
        ]
    )
    input_cell_names1 = np.array(
        [
            input_cellset1.get_cell_name(i)
            for i in range(input_cellset1.num_cells)
        ]
    )
    assert (exp_cell_statuses == input_cell_statuses1).all()
    assert (exp_cell_names == input_cell_names1).all()

    # validate output denoised cell sets
    denoised_cellset0 = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset0.num_cells == exp_num_cells
    assert denoised_cellset0.timing.num_samples == exp_num_frames_part0
    assert denoised_cellset0.spacing.num_pixels[0] == exp_height
    assert denoised_cellset0.spacing.num_pixels[1] == exp_width
    act_denoised_cellset_names0 = np.array(
        [
            denoised_cellset0.get_cell_name(j)
            for j in range(denoised_cellset0.num_cells)
        ]
    )
    assert (exp_cell_names == act_denoised_cellset_names0).all()
    act_denoised_cellset_statuses0 = np.array(
        [
            denoised_cellset0.get_cell_status(i)
            for i in range(denoised_cellset0.num_cells)
        ]
    )
    assert (exp_cell_statuses == act_denoised_cellset_statuses0).all()
    assert exp_timing0 == denoised_cellset0.timing
    assert exp_spacing0 == denoised_cellset0.spacing

    denoised_cellset1 = isx.CellSet.read("cellset_denoised.001.isxd")
    assert denoised_cellset1.num_cells == exp_num_cells
    assert denoised_cellset1.timing.num_samples == exp_num_frames_part1
    assert denoised_cellset1.spacing.num_pixels[0] == exp_height
    assert denoised_cellset1.spacing.num_pixels[1] == exp_width
    act_denoised_cellset_names1 = np.array(
        [
            denoised_cellset1.get_cell_name(j)
            for j in range(denoised_cellset1.num_cells)
        ]
    )
    assert (exp_cell_names == act_denoised_cellset_names1).all()
    act_denoised_cellset_statuses1 = np.array(
        [
            denoised_cellset1.get_cell_status(i)
            for i in range(denoised_cellset1.num_cells)
        ]
    )
    assert (exp_cell_statuses == act_denoised_cellset_statuses1).all()
    assert exp_timing1 == denoised_cellset1.timing
    assert exp_spacing1 == denoised_cellset1.spacing

    # validate output event sets
    eventset0 = isx.EventSet.read("neural_events.000.isxd")
    assert eventset0.num_cells == exp_num_cells
    assert eventset0.timing.num_samples == exp_num_frames_part0
    assert exp_timing0 == eventset0.timing
    act_eventset_names0 = np.array(
        [eventset0.get_cell_name(j) for j in range(eventset0.num_cells)]
    )
    assert (exp_cell_names == act_eventset_names0).all()

    eventset1 = isx.EventSet.read("neural_events.001.isxd")
    assert eventset1.num_cells == exp_num_cells
    assert eventset1.timing.num_samples == exp_num_frames_part1
    assert exp_timing1 == eventset1.timing
    act_eventset_names1 = np.array(
        [eventset1.get_cell_name(j) for j in range(eventset1.num_cells)]
    )
    assert (exp_cell_names == act_eventset_names1).all()
