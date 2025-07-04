import os
import shutil
import pytest
import isx
import json
import numpy as np
from toolbox.tools.caiman_isx_academic import caiman_workflow
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from toolbox.utils.exceptions import IdeasError
from toolbox.utils.utilities import read_isxd_metadata


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
        "caiman_output.hdf5",
        "output_metadata.json",
        "output_manifest.json",
    ]:
        if os.path.exists(f):
            os.remove(f)

    file_ext = [".isxd", ".svg"]
    for f in os.listdir("."):
        if os.path.splitext(f)[-1] in file_ext:
            os.remove(f)


def test_caiman_cnmfe_workflow_single_1p_tif_movie_params_from_file():
    """Verify that the CaImAn CNMF-E workflow can be applied to a 1P tif movie using parameters obtained from a json file."""
    input_movie_files = [os.path.join(DATA_DIR, "data_endoscope.tif")]
    input_parameters_file = [
        os.path.join(DATA_DIR, "params_demo_pipeline_cnmfE.json")
    ]

    caiman_workflow(
        input_movie_files=input_movie_files,
        parameters_file=input_parameters_file,
        overwrite_analysis_table_params=True,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 159
    exp_num_accepted_cells = 118
    exp_num_rejected_cells = 41
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames = 1000
    exp_width = 128
    exp_height = 128
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["accepted"] * act_num_cells)
    exp_cell_statuses[model.estimates.idx_components_bad] = "rejected"
    exp_timing = isx.Timing(
        num_samples=exp_num_frames, period=isx.Duration.from_msecs(100)
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    raw_cellset = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset.num_cells == exp_num_cells
    assert raw_cellset.timing.num_samples == exp_num_frames
    assert raw_cellset.spacing.num_pixels[0] == exp_height
    assert raw_cellset.spacing.num_pixels[1] == exp_width
    assert raw_cellset.get_cell_name(0) == exp_cell0_name
    assert raw_cellset.get_cell_name(3) == exp_cell3_name
    raw_cellset_statuses = np.array(
        [raw_cellset.get_cell_status(i) for i in range(raw_cellset.num_cells)]
    )
    assert (exp_cell_statuses == raw_cellset_statuses).all()
    assert exp_timing == raw_cellset.timing
    assert exp_spacing == raw_cellset.spacing

    denoised_cellset = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset.num_cells == exp_num_cells
    assert denoised_cellset.timing.num_samples == exp_num_frames
    assert denoised_cellset.spacing.num_pixels[0] == exp_height
    assert denoised_cellset.spacing.num_pixels[1] == exp_width
    assert denoised_cellset.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset.get_cell_name(3) == exp_cell3_name
    denoised_cellset_statuses = np.array(
        [
            denoised_cellset.get_cell_status(i)
            for i in range(denoised_cellset.num_cells)
        ]
    )
    assert (exp_cell_statuses == denoised_cellset_statuses).all()
    assert exp_timing == denoised_cellset.timing
    assert exp_spacing == denoised_cellset.spacing

    eventset = isx.EventSet.read("neural_events.000.isxd")
    assert eventset.num_cells == exp_num_cells
    assert eventset.timing.num_samples == exp_num_frames
    assert eventset.get_cell_name(0) == exp_cell0_name
    assert eventset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == eventset.timing


def test_caiman_cnmfe_workflow_single_1p_tif_movie_params_from_default_user_specified():
    """Verify that the CaImAn CNMF-E workflow can be applied to a 1P tif movie using parameters specified by the user or default."""
    input_movie_files = [os.path.join(DATA_DIR, "data_endoscope.tif")]

    caiman_workflow(input_movie_files=input_movie_files, fr=10, gSig_filt=3)

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 94
    exp_num_accepted_cells = 78
    exp_num_rejected_cells = 16
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames = 1000
    exp_width = 128
    exp_height = 128
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["accepted"] * act_num_cells)
    exp_cell_statuses[model.estimates.idx_components_bad] = "rejected"
    exp_timing = isx.Timing(
        num_samples=exp_num_frames, period=isx.Duration.from_msecs(100)
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    raw_cellset = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset.num_cells == exp_num_cells
    assert raw_cellset.timing.num_samples == exp_num_frames
    assert raw_cellset.spacing.num_pixels[0] == exp_height
    assert raw_cellset.spacing.num_pixels[1] == exp_width
    assert raw_cellset.get_cell_name(0) == exp_cell0_name
    assert raw_cellset.get_cell_name(3) == exp_cell3_name
    raw_cellset_statuses = np.array(
        [raw_cellset.get_cell_status(i) for i in range(raw_cellset.num_cells)]
    )
    assert (exp_cell_statuses == raw_cellset_statuses).all()
    assert exp_timing == raw_cellset.timing
    assert exp_spacing == raw_cellset.spacing

    denoised_cellset = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset.num_cells == exp_num_cells
    assert denoised_cellset.timing.num_samples == exp_num_frames
    assert denoised_cellset.spacing.num_pixels[0] == exp_height
    assert denoised_cellset.spacing.num_pixels[1] == exp_width
    assert denoised_cellset.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset.get_cell_name(3) == exp_cell3_name
    denoised_cellset_statuses = np.array(
        [
            denoised_cellset.get_cell_status(i)
            for i in range(denoised_cellset.num_cells)
        ]
    )
    assert (exp_cell_statuses == denoised_cellset_statuses).all()
    assert exp_timing == denoised_cellset.timing
    assert exp_spacing == denoised_cellset.spacing

    eventset = isx.EventSet.read("neural_events.000.isxd")
    assert eventset.num_cells == exp_num_cells
    assert eventset.timing.num_samples == exp_num_frames
    assert eventset.get_cell_name(0) == exp_cell0_name
    assert eventset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == eventset.timing


def test_caiman_cnmfe_workflow_no_cells_identified():
    """Verify that the CaImAn CNMF-E workflow correctly handle scenario where no cells are identified."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd")
    ]

    with pytest.raises(IdeasError) as e:
        caiman_workflow(
            input_movie_files=input_movie_files,
            overwrite_analysis_table_params=False,
            min_corr=0.8,
            min_pnr=10,
        )
    assert str(e.value) == "No cells were identified"


def test_caiman_cnmfe_workflow_single_isxd_file():
    """Verify that the CaImAn CNMF-E workflow correctly identifies cells in a single ISXD movie."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd")
    ]

    caiman_workflow(
        input_movie_files=input_movie_files,
        overwrite_analysis_table_params=False,
        min_corr=0.5,
        min_pnr=5,
        gSig_filt=3,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 70
    exp_num_accepted_cells = 43
    exp_num_rejected_cells = 27
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames = 100
    exp_width = 108
    exp_height = 123
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["accepted"] * act_num_cells)
    exp_cell_statuses[model.estimates.idx_components_bad] = "rejected"
    input_movie = isx.Movie.read(input_movie_files[0])
    exp_timing = input_movie.timing
    exp_spacing = input_movie.spacing

    raw_cellset = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset.num_cells == exp_num_cells
    assert raw_cellset.timing.num_samples == exp_num_frames
    assert raw_cellset.spacing.num_pixels[0] == exp_height
    assert raw_cellset.spacing.num_pixels[1] == exp_width
    assert raw_cellset.get_cell_name(0) == exp_cell0_name
    assert raw_cellset.get_cell_name(3) == exp_cell3_name
    raw_cellset_statuses = np.array(
        [raw_cellset.get_cell_status(i) for i in range(raw_cellset.num_cells)]
    )
    assert (exp_cell_statuses == raw_cellset_statuses).all()
    assert exp_timing == raw_cellset.timing
    assert exp_spacing == raw_cellset.spacing

    denoised_cellset = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset.num_cells == exp_num_cells
    assert denoised_cellset.timing.num_samples == exp_num_frames
    assert denoised_cellset.spacing.num_pixels[0] == exp_height
    assert denoised_cellset.spacing.num_pixels[1] == exp_width
    assert denoised_cellset.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset.get_cell_name(3) == exp_cell3_name
    denoised_cellset_statuses = np.array(
        [
            denoised_cellset.get_cell_status(i)
            for i in range(denoised_cellset.num_cells)
        ]
    )
    assert (exp_cell_statuses == denoised_cellset_statuses).all()
    assert exp_timing == denoised_cellset.timing
    assert exp_spacing == denoised_cellset.spacing

    eventset = isx.EventSet.read("neural_events.000.isxd")
    assert eventset.num_cells == exp_num_cells
    assert eventset.timing.num_samples == exp_num_frames
    assert eventset.get_cell_name(0) == exp_cell0_name
    assert eventset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == eventset.timing


def test_caiman_cnmfe_workflow_isxd_movie_series():
    """Verify that the CaImAn CNMF-E workflow correctly processes an ISXD movie series."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd"),
        os.path.join(DATA_DIR, "movie_part2_108x123x63.isxd"),
    ]

    caiman_workflow(
        input_movie_files=input_movie_files,
        overwrite_analysis_table_params=False,
        min_corr=0.5,
        min_pnr=5,
        gSig_filt=3,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        "cellset_raw.001.isxd",
        "traces_cellset_raw.001.svg",
        "footprints_cellset_raw.001.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        "cellset_denoised.001.isxd",
        "traces_cellset_denoised.001.svg",
        "footprints_cellset_denoised.001.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        "neural_events.001.isxd",
        "preview_neural_events.001.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 150
    exp_num_accepted_cells = 63
    exp_num_rejected_cells = 87
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames_part0 = 100
    exp_num_frames_part1 = 63
    exp_width = 108
    exp_height = 123
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["accepted"] * act_num_cells)
    exp_cell_statuses[model.estimates.idx_components_bad] = "rejected"

    input_movie0 = isx.Movie.read(input_movie_files[0])
    exp_timing0 = input_movie0.timing
    exp_spacing0 = input_movie0.spacing

    input_movie1 = isx.Movie.read(input_movie_files[1])
    exp_timing1 = input_movie1.timing
    exp_spacing1 = input_movie1.spacing

    # first part of output series
    raw_cellset0 = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset0.num_cells == exp_num_cells
    assert raw_cellset0.timing.num_samples == exp_num_frames_part0
    assert raw_cellset0.spacing.num_pixels[0] == exp_height
    assert raw_cellset0.spacing.num_pixels[1] == exp_width
    assert raw_cellset0.get_cell_name(0) == exp_cell0_name
    assert raw_cellset0.get_cell_name(3) == exp_cell3_name
    raw_cellset0_statuses = np.array(
        [
            raw_cellset0.get_cell_status(i)
            for i in range(raw_cellset0.num_cells)
        ]
    )
    assert (exp_cell_statuses == raw_cellset0_statuses).all()
    assert exp_timing0 == raw_cellset0.timing
    assert exp_spacing0 == raw_cellset0.spacing

    denoised_cellset0 = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset0.num_cells == exp_num_cells
    assert denoised_cellset0.timing.num_samples == exp_num_frames_part0
    assert denoised_cellset0.spacing.num_pixels[0] == exp_height
    assert denoised_cellset0.spacing.num_pixels[1] == exp_width
    assert denoised_cellset0.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset0.get_cell_name(3) == exp_cell3_name
    denoised_cellset0_statuses = np.array(
        [
            denoised_cellset0.get_cell_status(i)
            for i in range(denoised_cellset0.num_cells)
        ]
    )
    assert (exp_cell_statuses == denoised_cellset0_statuses).all()
    assert exp_timing0 == denoised_cellset0.timing
    assert exp_spacing0 == denoised_cellset0.spacing

    eventset0 = isx.EventSet.read("neural_events.000.isxd")
    assert eventset0.num_cells == exp_num_cells
    assert eventset0.timing.num_samples == exp_num_frames_part0
    assert eventset0.get_cell_name(0) == exp_cell0_name
    assert eventset0.get_cell_name(3) == exp_cell3_name
    assert exp_timing0 == eventset0.timing

    # second part of output series
    raw_cellset1 = isx.CellSet.read("cellset_raw.001.isxd")
    assert raw_cellset1.num_cells == exp_num_cells
    assert raw_cellset1.timing.num_samples == exp_num_frames_part1
    assert raw_cellset1.spacing.num_pixels[0] == exp_height
    assert raw_cellset1.spacing.num_pixels[1] == exp_width
    assert raw_cellset1.get_cell_name(0) == exp_cell0_name
    assert raw_cellset1.get_cell_name(3) == exp_cell3_name
    raw_cellset1_statuses = np.array(
        [
            raw_cellset1.get_cell_status(i)
            for i in range(raw_cellset1.num_cells)
        ]
    )
    assert (exp_cell_statuses == raw_cellset1_statuses).all()
    assert exp_timing1 == raw_cellset1.timing
    assert exp_spacing1 == raw_cellset1.spacing

    denoised_cellset1 = isx.CellSet.read("cellset_denoised.001.isxd")
    assert denoised_cellset1.num_cells == exp_num_cells
    assert denoised_cellset1.timing.num_samples == exp_num_frames_part1
    assert denoised_cellset1.spacing.num_pixels[0] == exp_height
    assert denoised_cellset1.spacing.num_pixels[1] == exp_width
    assert denoised_cellset1.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset1.get_cell_name(3) == exp_cell3_name
    denoised_cellset1_statuses = np.array(
        [
            denoised_cellset1.get_cell_status(i)
            for i in range(denoised_cellset1.num_cells)
        ]
    )
    assert (exp_cell_statuses == denoised_cellset1_statuses).all()
    assert exp_timing1 == denoised_cellset1.timing
    assert exp_spacing1 == denoised_cellset1.spacing

    eventset1 = isx.EventSet.read("neural_events.001.isxd")
    assert eventset1.num_cells == exp_num_cells
    assert eventset1.timing.num_samples == exp_num_frames_part1
    assert eventset1.get_cell_name(0) == exp_cell0_name
    assert eventset1.get_cell_name(3) == exp_cell3_name
    assert exp_timing1 == eventset1.timing


def test_caiman_cnmfe_workflow_single_tiff_file():
    """Verify that the CaImAn CNMF-E workflow correctly identifies cells in a single TIFF movie."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.tiff")
    ]

    caiman_workflow(
        input_movie_files=input_movie_files,
        overwrite_analysis_table_params=False,
        min_corr=0.5,
        min_pnr=5,
        fr=10,
        gSig_filt=3,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 70
    exp_num_accepted_cells = 43
    exp_num_rejected_cells = 27
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames = 100
    exp_width = 108
    exp_height = 123
    exp_cell0_status = "accepted"
    exp_cell0_name = "C000"
    exp_cell3_status = "rejected"
    exp_cell3_name = "C003"
    exp_timing = isx.Timing(
        num_samples=exp_num_frames, period=isx.Duration.from_msecs(100)
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    raw_cellset = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset.num_cells == exp_num_cells
    assert raw_cellset.timing.num_samples == exp_num_frames
    assert raw_cellset.spacing.num_pixels[0] == exp_height
    assert raw_cellset.spacing.num_pixels[1] == exp_width
    assert raw_cellset.get_cell_status(0) == exp_cell0_status
    assert raw_cellset.get_cell_name(0) == exp_cell0_name
    assert raw_cellset.get_cell_status(3) == exp_cell3_status
    assert raw_cellset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == raw_cellset.timing
    assert exp_spacing == raw_cellset.spacing

    denoised_cellset = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset.num_cells == exp_num_cells
    assert denoised_cellset.timing.num_samples == exp_num_frames
    assert denoised_cellset.spacing.num_pixels[0] == exp_height
    assert denoised_cellset.spacing.num_pixels[1] == exp_width
    assert denoised_cellset.get_cell_status(0) == exp_cell0_status
    assert denoised_cellset.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset.get_cell_status(3) == exp_cell3_status
    assert denoised_cellset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == denoised_cellset.timing
    assert exp_spacing == denoised_cellset.spacing

    eventset = isx.EventSet.read("neural_events.000.isxd")
    assert eventset.num_cells == exp_num_cells
    assert eventset.timing.num_samples == exp_num_frames
    assert eventset.get_cell_name(0) == exp_cell0_name
    assert eventset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == eventset.timing


def test_caiman_cnmfe_workflow_tiff_movie_series():
    """Verify that the CaImAn CNMF-E workflow correctly processes a TIFF movie series."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.tiff"),
        os.path.join(DATA_DIR, "movie_part2_108x123x63.tiff"),
    ]

    caiman_workflow(
        input_movie_files=input_movie_files,
        overwrite_analysis_table_params=False,
        min_corr=0.5,
        min_pnr=5,
        fr=10,
        gSig_filt=3,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        "cellset_raw.001.isxd",
        "traces_cellset_raw.001.svg",
        "footprints_cellset_raw.001.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        "cellset_denoised.001.isxd",
        "traces_cellset_denoised.001.svg",
        "footprints_cellset_denoised.001.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        "neural_events.001.isxd",
        "preview_neural_events.001.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 150
    exp_num_accepted_cells = 63
    exp_num_rejected_cells = 87
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames_part0 = 100
    exp_num_frames_part1 = 63
    exp_width = 108
    exp_height = 123
    exp_cell0_status = "accepted"
    exp_cell0_name = "C000"
    exp_cell3_status = "rejected"
    exp_cell3_name = "C003"

    exp_timing0 = isx.Timing(
        num_samples=exp_num_frames_part0, period=isx.Duration.from_msecs(100)
    )
    exp_spacing0 = isx.Spacing(num_pixels=(exp_height, exp_width))

    exp_timing1 = isx.Timing(
        num_samples=exp_num_frames_part1, period=isx.Duration.from_msecs(100)
    )
    exp_spacing1 = isx.Spacing(num_pixels=(exp_height, exp_width))

    # first part of output series
    raw_cellset0 = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset0.num_cells == exp_num_cells
    assert raw_cellset0.timing.num_samples == exp_num_frames_part0
    assert raw_cellset0.spacing.num_pixels[0] == exp_height
    assert raw_cellset0.spacing.num_pixels[1] == exp_width
    assert raw_cellset0.get_cell_status(0) == exp_cell0_status
    assert raw_cellset0.get_cell_name(0) == exp_cell0_name
    assert raw_cellset0.get_cell_status(3) == exp_cell3_status
    assert raw_cellset0.get_cell_name(3) == exp_cell3_name
    assert exp_timing0 == raw_cellset0.timing
    assert exp_spacing0 == raw_cellset0.spacing

    denoised_cellset0 = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset0.num_cells == exp_num_cells
    assert denoised_cellset0.timing.num_samples == exp_num_frames_part0
    assert denoised_cellset0.spacing.num_pixels[0] == exp_height
    assert denoised_cellset0.spacing.num_pixels[1] == exp_width
    assert denoised_cellset0.get_cell_status(0) == exp_cell0_status
    assert denoised_cellset0.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset0.get_cell_status(3) == exp_cell3_status
    assert denoised_cellset0.get_cell_name(3) == exp_cell3_name
    assert exp_timing0 == denoised_cellset0.timing
    assert exp_spacing0 == denoised_cellset0.spacing

    eventset0 = isx.EventSet.read("neural_events.000.isxd")
    assert eventset0.num_cells == exp_num_cells
    assert eventset0.timing.num_samples == exp_num_frames_part0
    assert eventset0.get_cell_name(0) == exp_cell0_name
    assert eventset0.get_cell_name(3) == exp_cell3_name
    assert exp_timing0 == eventset0.timing

    # second part of output series
    raw_cellset1 = isx.CellSet.read("cellset_raw.001.isxd")
    assert raw_cellset1.num_cells == exp_num_cells
    assert raw_cellset1.timing.num_samples == exp_num_frames_part1
    assert raw_cellset1.spacing.num_pixels[0] == exp_height
    assert raw_cellset1.spacing.num_pixels[1] == exp_width
    assert raw_cellset1.get_cell_status(0) == exp_cell0_status
    assert raw_cellset1.get_cell_name(0) == exp_cell0_name
    assert raw_cellset1.get_cell_status(3) == exp_cell3_status
    assert raw_cellset1.get_cell_name(3) == exp_cell3_name
    assert exp_timing1 == raw_cellset1.timing
    assert exp_spacing1 == raw_cellset1.spacing

    denoised_cellset1 = isx.CellSet.read("cellset_denoised.001.isxd")
    assert denoised_cellset1.num_cells == exp_num_cells
    assert denoised_cellset1.timing.num_samples == exp_num_frames_part1
    assert denoised_cellset1.spacing.num_pixels[0] == exp_height
    assert denoised_cellset1.spacing.num_pixels[1] == exp_width
    assert denoised_cellset1.get_cell_status(0) == exp_cell0_status
    assert denoised_cellset1.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset1.get_cell_status(3) == exp_cell3_status
    assert denoised_cellset1.get_cell_name(3) == exp_cell3_name
    assert exp_timing1 == denoised_cellset1.timing
    assert exp_spacing1 == denoised_cellset1.spacing

    eventset1 = isx.EventSet.read("neural_events.001.isxd")
    assert eventset1.num_cells == exp_num_cells
    assert eventset1.timing.num_samples == exp_num_frames_part1
    assert eventset1.get_cell_name(0) == exp_cell0_name
    assert eventset1.get_cell_name(3) == exp_cell3_name
    assert exp_timing1 == eventset1.timing


def test_caiman_cnmf_workflow_single_2p_tif_movie_params_from_default_user_specified():
    """Verify that the CaImAn CNMF workflow can be applied to a 2P tif movie using parameters specified by the user or default."""
    input_movie_files = [os.path.join(DATA_DIR, "demoMovie.tif")]

    caiman_workflow(
        input_movie_files=input_movie_files,
        K=4,
        dxy=2.0,
        center_psf=False,
        gSig=4,
        gSiz=9,
        method_init="greedy_roi",
        nb=2,
        normalize_init=True,
        tsub=1,
        merge_thr=0.85,
        gSig_filt=None,
        max_shifts=6,
        pw_rigid=True,
        min_SNR=2.0,
        del_duplicates=False,
        low_rank_background=True,
        nb_patch=1,
        rf=15,
        stride=10,
        min_cnn_thr=0.99,
        use_cnn=True,
        bas_nonneg=True,
        min_pnr=20,
        ring_size_factor=1.5,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 18
    exp_num_accepted_cells = 16
    exp_num_rejected_cells = 2
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames = 2000
    exp_width = 80
    exp_height = 60
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["accepted"] * act_num_cells)
    exp_cell_statuses[model.estimates.idx_components_bad] = "rejected"
    exp_timing = isx.Timing(
        num_samples=exp_num_frames,
        period=isx.Duration.from_msecs(1.0 / 30.0 * 1000),
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    raw_cellset = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset.num_cells == exp_num_cells
    assert raw_cellset.timing.num_samples == exp_num_frames
    assert raw_cellset.spacing.num_pixels[0] == exp_height
    assert raw_cellset.spacing.num_pixels[1] == exp_width
    assert raw_cellset.get_cell_name(0) == exp_cell0_name
    assert raw_cellset.get_cell_name(3) == exp_cell3_name
    raw_cellset_statuses = np.array(
        [raw_cellset.get_cell_status(i) for i in range(raw_cellset.num_cells)]
    )
    assert (exp_cell_statuses == raw_cellset_statuses).all()
    assert exp_timing == raw_cellset.timing
    assert exp_spacing == raw_cellset.spacing

    denoised_cellset = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset.num_cells == exp_num_cells
    assert denoised_cellset.timing.num_samples == exp_num_frames
    assert denoised_cellset.spacing.num_pixels[0] == exp_height
    assert denoised_cellset.spacing.num_pixels[1] == exp_width
    assert denoised_cellset.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset.get_cell_name(3) == exp_cell3_name
    denoised_cellset_statuses = np.array(
        [
            denoised_cellset.get_cell_status(i)
            for i in range(denoised_cellset.num_cells)
        ]
    )
    assert (exp_cell_statuses == denoised_cellset_statuses).all()
    assert exp_timing == denoised_cellset.timing
    assert exp_spacing == denoised_cellset.spacing

    eventset = isx.EventSet.read("neural_events.000.isxd")
    assert eventset.num_cells == exp_num_cells
    assert eventset.timing.num_samples == exp_num_frames
    assert eventset.get_cell_name(0) == exp_cell0_name
    assert eventset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == eventset.timing


def test_caiman_cnmf_workflow_single_2p_tif_movie_params_from_file():
    """Verify that the CaImAn CNMF workflow can be applied to a 2P tif movie using parameters obtained from a json file."""
    input_movie_files = [os.path.join(DATA_DIR, "demoMovie.tif")]
    input_parameters_file = [
        os.path.join(DATA_DIR, "params_demo_pipeline.json")
    ]

    caiman_workflow(
        input_movie_files=input_movie_files,
        parameters_file=input_parameters_file,
        overwrite_analysis_table_params=True,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 18
    exp_num_accepted_cells = 16
    exp_num_rejected_cells = 2
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames = 2000
    exp_width = 80
    exp_height = 60
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["accepted"] * act_num_cells)
    exp_cell_statuses[model.estimates.idx_components_bad] = "rejected"
    exp_timing = isx.Timing(
        num_samples=exp_num_frames,
        period=isx.Duration.from_msecs(1.0 / 30.0 * 1000),
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    raw_cellset = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset.num_cells == exp_num_cells
    assert raw_cellset.timing.num_samples == exp_num_frames
    assert raw_cellset.spacing.num_pixels[0] == exp_height
    assert raw_cellset.spacing.num_pixels[1] == exp_width
    assert raw_cellset.get_cell_name(0) == exp_cell0_name
    assert raw_cellset.get_cell_name(3) == exp_cell3_name
    raw_cellset_statuses = np.array(
        [raw_cellset.get_cell_status(i) for i in range(raw_cellset.num_cells)]
    )
    assert (exp_cell_statuses == raw_cellset_statuses).all()
    assert exp_timing == raw_cellset.timing
    assert exp_spacing == raw_cellset.spacing

    denoised_cellset = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset.num_cells == exp_num_cells
    assert denoised_cellset.timing.num_samples == exp_num_frames
    assert denoised_cellset.spacing.num_pixels[0] == exp_height
    assert denoised_cellset.spacing.num_pixels[1] == exp_width
    assert denoised_cellset.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset.get_cell_name(3) == exp_cell3_name
    denoised_cellset_statuses = np.array(
        [
            denoised_cellset.get_cell_status(i)
            for i in range(denoised_cellset.num_cells)
        ]
    )
    assert (exp_cell_statuses == denoised_cellset_statuses).all()
    assert exp_timing == denoised_cellset.timing
    assert exp_spacing == denoised_cellset.spacing

    eventset = isx.EventSet.read("neural_events.000.isxd")
    assert eventset.num_cells == exp_num_cells
    assert eventset.timing.num_samples == exp_num_frames
    assert eventset.get_cell_name(0) == exp_cell0_name
    assert eventset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == eventset.timing


def test_caiman_cnmfe_workflow_single_avi_movie():
    """Verify that the CaImAn CNMF-E workflow can be applied to a 1P avi movie."""
    input_movie_files = [os.path.join(DATA_DIR, "movie_part1_108x122x200.avi")]

    caiman_workflow(
        input_movie_files=input_movie_files,
        min_corr=0.6,
        min_pnr=7,
        gSig_filt=3,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 45
    exp_num_accepted_cells = 33
    exp_num_rejected_cells = 12
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames = 200
    exp_width = 108
    exp_height = 122
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["accepted"] * act_num_cells)
    exp_cell_statuses[model.estimates.idx_components_bad] = "rejected"
    exp_timing = isx.Timing(
        num_samples=exp_num_frames, period=isx.Duration.from_msecs(50)
    )
    exp_spacing = isx.Spacing(num_pixels=(exp_height, exp_width))

    raw_cellset = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset.num_cells == exp_num_cells
    assert raw_cellset.timing.num_samples == exp_num_frames
    assert raw_cellset.spacing.num_pixels[0] == exp_height
    assert raw_cellset.spacing.num_pixels[1] == exp_width
    assert raw_cellset.get_cell_name(0) == exp_cell0_name
    assert raw_cellset.get_cell_name(3) == exp_cell3_name
    raw_cellset_statuses = np.array(
        [raw_cellset.get_cell_status(i) for i in range(raw_cellset.num_cells)]
    )
    assert (exp_cell_statuses == raw_cellset_statuses).all()
    assert exp_timing == raw_cellset.timing
    assert exp_spacing == raw_cellset.spacing

    denoised_cellset = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset.num_cells == exp_num_cells
    assert denoised_cellset.timing.num_samples == exp_num_frames
    assert denoised_cellset.spacing.num_pixels[0] == exp_height
    assert denoised_cellset.spacing.num_pixels[1] == exp_width
    assert denoised_cellset.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset.get_cell_name(3) == exp_cell3_name
    denoised_cellset_statuses = np.array(
        [
            denoised_cellset.get_cell_status(i)
            for i in range(denoised_cellset.num_cells)
        ]
    )
    assert (exp_cell_statuses == denoised_cellset_statuses).all()
    assert exp_timing == denoised_cellset.timing
    assert exp_spacing == denoised_cellset.spacing

    eventset = isx.EventSet.read("neural_events.000.isxd")
    assert eventset.num_cells == exp_num_cells
    assert eventset.timing.num_samples == exp_num_frames
    assert eventset.get_cell_name(0) == exp_cell0_name
    assert eventset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == eventset.timing


def test_caiman_cnmfe_workflow_avi_movie_series():
    """Verify that the CaImAn CNMF-E workflow correctly processes an AVI movie series."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x122x200.avi"),
        os.path.join(DATA_DIR, "movie_part2_108x122x126.avi"),
    ]

    caiman_workflow(
        input_movie_files=input_movie_files,
        min_corr=0.6,
        min_pnr=7,
        gSig_filt=3,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        "cellset_raw.001.isxd",
        "traces_cellset_raw.001.svg",
        "footprints_cellset_raw.001.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        "cellset_denoised.001.isxd",
        "traces_cellset_denoised.001.svg",
        "footprints_cellset_denoised.001.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        "neural_events.001.isxd",
        "preview_neural_events.001.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 66
    exp_num_accepted_cells = 39
    exp_num_rejected_cells = 27
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    exp_num_frames_part0 = 200
    exp_num_frames_part1 = 126
    exp_width = 108
    exp_height = 122
    exp_cell0_status = "accepted"
    exp_cell0_name = "C000"
    exp_cell3_status = "accepted"
    exp_cell3_name = "C003"

    exp_timing0 = isx.Timing(
        num_samples=exp_num_frames_part0, period=isx.Duration.from_msecs(50)
    )
    exp_spacing0 = isx.Spacing(num_pixels=(exp_height, exp_width))

    exp_timing1 = isx.Timing(
        num_samples=exp_num_frames_part1, period=isx.Duration.from_msecs(50)
    )
    exp_spacing1 = isx.Spacing(num_pixels=(exp_height, exp_width))

    # first part of output series
    raw_cellset0 = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset0.num_cells == exp_num_cells
    assert raw_cellset0.timing.num_samples == exp_num_frames_part0
    assert raw_cellset0.spacing.num_pixels[0] == exp_height
    assert raw_cellset0.spacing.num_pixels[1] == exp_width
    assert raw_cellset0.get_cell_status(0) == exp_cell0_status
    assert raw_cellset0.get_cell_name(0) == exp_cell0_name
    assert raw_cellset0.get_cell_status(3) == exp_cell3_status
    assert raw_cellset0.get_cell_name(3) == exp_cell3_name
    assert exp_timing0 == raw_cellset0.timing
    assert exp_spacing0 == raw_cellset0.spacing

    denoised_cellset0 = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset0.num_cells == exp_num_cells
    assert denoised_cellset0.timing.num_samples == exp_num_frames_part0
    assert denoised_cellset0.spacing.num_pixels[0] == exp_height
    assert denoised_cellset0.spacing.num_pixels[1] == exp_width
    assert denoised_cellset0.get_cell_status(0) == exp_cell0_status
    assert denoised_cellset0.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset0.get_cell_status(3) == exp_cell3_status
    assert denoised_cellset0.get_cell_name(3) == exp_cell3_name
    assert exp_timing0 == denoised_cellset0.timing
    assert exp_spacing0 == denoised_cellset0.spacing

    eventset0 = isx.EventSet.read("neural_events.000.isxd")
    assert eventset0.num_cells == exp_num_cells
    assert eventset0.timing.num_samples == exp_num_frames_part0
    assert eventset0.get_cell_name(0) == exp_cell0_name
    assert eventset0.get_cell_name(3) == exp_cell3_name
    assert exp_timing0 == eventset0.timing

    # second part of output series
    raw_cellset1 = isx.CellSet.read("cellset_raw.001.isxd")
    assert raw_cellset1.num_cells == exp_num_cells
    assert raw_cellset1.timing.num_samples == exp_num_frames_part1
    assert raw_cellset1.spacing.num_pixels[0] == exp_height
    assert raw_cellset1.spacing.num_pixels[1] == exp_width
    assert raw_cellset1.get_cell_status(0) == exp_cell0_status
    assert raw_cellset1.get_cell_name(0) == exp_cell0_name
    assert raw_cellset1.get_cell_status(3) == exp_cell3_status
    assert raw_cellset1.get_cell_name(3) == exp_cell3_name
    assert exp_timing1 == raw_cellset1.timing
    assert exp_spacing1 == raw_cellset1.spacing

    denoised_cellset1 = isx.CellSet.read("cellset_denoised.001.isxd")
    assert denoised_cellset1.num_cells == exp_num_cells
    assert denoised_cellset1.timing.num_samples == exp_num_frames_part1
    assert denoised_cellset1.spacing.num_pixels[0] == exp_height
    assert denoised_cellset1.spacing.num_pixels[1] == exp_width
    assert denoised_cellset1.get_cell_status(0) == exp_cell0_status
    assert denoised_cellset1.get_cell_name(0) == exp_cell0_name
    assert denoised_cellset1.get_cell_status(3) == exp_cell3_status
    assert denoised_cellset1.get_cell_name(3) == exp_cell3_name
    assert exp_timing1 == denoised_cellset1.timing
    assert exp_spacing1 == denoised_cellset1.spacing

    eventset1 = isx.EventSet.read("neural_events.001.isxd")
    assert eventset1.num_cells == exp_num_cells
    assert eventset1.timing.num_samples == exp_num_frames_part1
    assert eventset1.get_cell_name(0) == exp_cell0_name
    assert eventset1.get_cell_name(3) == exp_cell3_name
    assert exp_timing1 == eventset1.timing


def test_caiman_cnmf_workflow_on_2p_data_with_minimal_params():
    """Verify that the CaImAn CNMF workflow correctly processes 2P data with minimal 2P parameters specified."""
    input_movie_files = [os.path.join(DATA_DIR, "demoMovie.tif")]

    caiman_workflow(
        input_movie_files=input_movie_files,
        fr=30,
        dxy=2.0,
        K=4,
        gSig=4,
        gSiz=9,
        method_init="greedy_roi",
        tsub=1,
        center_psf=False,
        normalize_init=True,
        nb=2,
        max_shifts=6,
        nb_patch=1,
        rf=15,
        stride=10,
        low_rank_background=True,
        del_duplicates=False,
    )

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files


def test_caiman_cnmfe_workflow_unordered_isxd_movie_series():
    """Verify that the CaImAn CNMF-E workflow correctly processes an UNORDERED ISXD movie series."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part2.isxd"),
        os.path.join(DATA_DIR, "movie_part3.isxd"),
        os.path.join(DATA_DIR, "movie_part1.isxd"),
    ]

    caiman_workflow(input_movie_files=input_movie_files)

    # validate existence of output files
    output_dir = os.getcwd()
    act_output_files = os.listdir(output_dir)
    for f in [
        # caiman output
        "caiman_output.hdf5",
        "initialization_images.svg",
        # raw cell set
        "cellset_raw.000.isxd",
        "traces_cellset_raw.000.svg",
        "footprints_cellset_raw.000.svg",
        "cellset_raw.001.isxd",
        "traces_cellset_raw.001.svg",
        "footprints_cellset_raw.001.svg",
        "cellset_raw.002.isxd",
        "traces_cellset_raw.002.svg",
        "footprints_cellset_raw.002.svg",
        # denoised cell set
        "cellset_denoised.000.isxd",
        "traces_cellset_denoised.000.svg",
        "footprints_cellset_denoised.000.svg",
        "cellset_denoised.001.isxd",
        "traces_cellset_denoised.001.svg",
        "footprints_cellset_denoised.001.svg",
        "cellset_denoised.002.isxd",
        "traces_cellset_denoised.002.svg",
        "footprints_cellset_denoised.002.svg",
        # neural events
        "neural_events.000.isxd",
        "preview_neural_events.000.svg",
        "neural_events.001.isxd",
        "preview_neural_events.001.svg",
        "neural_events.002.isxd",
        "preview_neural_events.002.svg",
        # metadata
        "output_metadata.json",
    ]:
        assert f in act_output_files

    # validate CaImAn output
    model_file = os.path.join(output_dir, "caiman_output.hdf5")
    model = load_CNMF(model_file)
    act_num_cells = len(model.estimates.C)
    act_num_accepted_cells = len(model.estimates.idx_components)
    act_num_rejected_cells = len(model.estimates.idx_components_bad)
    assert act_num_cells == act_num_accepted_cells + act_num_rejected_cells

    exp_num_cells = 6
    exp_num_accepted_cells = 6
    exp_num_rejected_cells = 0
    assert act_num_cells == exp_num_cells
    assert act_num_accepted_cells == exp_num_accepted_cells
    assert act_num_rejected_cells == exp_num_rejected_cells

    # validate output ISXD files
    input_movie_part0 = isx.Movie.read(input_movie_files[0])
    input_movie_part1 = isx.Movie.read(input_movie_files[1])
    input_movie_part2 = isx.Movie.read(input_movie_files[2])

    # validate raw cell sets
    raw_cellset_part0 = isx.CellSet.read("cellset_raw.000.isxd")
    assert raw_cellset_part0.timing.num_samples == 95
    assert raw_cellset_part0.spacing.num_pixels == (128, 128)
    assert input_movie_part0.spacing == raw_cellset_part0.spacing
    assert input_movie_part0.timing == raw_cellset_part0.timing

    raw_cellset_part1 = isx.CellSet.read("cellset_raw.001.isxd")
    assert raw_cellset_part1.timing.num_samples == 34
    assert raw_cellset_part1.spacing.num_pixels == (128, 128)
    assert input_movie_part1.spacing == raw_cellset_part1.spacing
    assert input_movie_part1.timing == raw_cellset_part1.timing

    raw_cellset_part2 = isx.CellSet.read("cellset_raw.002.isxd")
    assert raw_cellset_part2.timing.num_samples == 100
    assert raw_cellset_part2.spacing.num_pixels == (128, 128)
    assert input_movie_part2.spacing == raw_cellset_part2.spacing
    assert input_movie_part2.timing == raw_cellset_part2.timing

    # validate denoised cell sets
    denoised_cellset_part0 = isx.CellSet.read("cellset_denoised.000.isxd")
    assert denoised_cellset_part0.timing.num_samples == 95
    assert denoised_cellset_part0.spacing.num_pixels == (128, 128)
    assert input_movie_part0.spacing == denoised_cellset_part0.spacing
    assert input_movie_part0.timing == denoised_cellset_part0.timing

    denoised_cellset_part1 = isx.CellSet.read("cellset_denoised.001.isxd")
    assert denoised_cellset_part1.timing.num_samples == 34
    assert denoised_cellset_part1.spacing.num_pixels == (128, 128)
    assert input_movie_part1.spacing == denoised_cellset_part1.spacing
    assert input_movie_part1.timing == denoised_cellset_part1.timing

    denoised_cellset_part2 = isx.CellSet.read("cellset_denoised.002.isxd")
    assert denoised_cellset_part2.timing.num_samples == 100
    assert denoised_cellset_part2.spacing.num_pixels == (128, 128)
    assert input_movie_part2.spacing == denoised_cellset_part2.spacing
    assert input_movie_part2.timing == denoised_cellset_part2.timing

    # validate event sets
    eventset_part0 = isx.EventSet.read("neural_events.000.isxd")
    assert eventset_part0.timing.num_samples == 95
    assert input_movie_part0.timing == eventset_part0.timing

    eventset_part1 = isx.EventSet.read("neural_events.001.isxd")
    assert eventset_part1.timing.num_samples == 34
    assert input_movie_part1.timing == eventset_part1.timing

    eventset_part2 = isx.EventSet.read("neural_events.002.isxd")
    assert eventset_part2.timing.num_samples == 100
    assert input_movie_part2.timing == eventset_part2.timing

    # validate output metadata
    with open("output_metadata.json") as f:
        act_metadata = json.load(f)
        act_caiman_output_metadata = act_metadata["caiman_output"]
        act_raw_cellset_metadata0 = act_metadata["cellset_raw.000"]
        act_raw_cellset_metadata1 = act_metadata["cellset_raw.001"]
        act_raw_cellset_metadata2 = act_metadata["cellset_raw.002"]
        act_denoised_cellset_metadata0 = act_metadata["cellset_denoised.000"]
        act_denoised_cellset_metadata1 = act_metadata["cellset_denoised.001"]
        act_denoised_cellset_metadata2 = act_metadata["cellset_denoised.002"]
        act_eventset_metadata0 = act_metadata["neural_events.000"]
        act_eventset_metadata1 = act_metadata["neural_events.001"]
        act_eventset_metadata2 = act_metadata["neural_events.002"]

    # caiman metadata
    exp_caiman_output_metadata = {
        "metrics": {
            "num_accepted_cells": 6,
            "num_undecided_cells": 0,
            "num_rejected_cells": 0,
            "total_num_cells": 6,
        }
    }
    assert exp_caiman_output_metadata == act_caiman_output_metadata

    # raw cell sets metadata
    exp_raw_cellset_metadata0 = {
        "metrics": {
            "num_accepted_cells": 6,
            "num_undecided_cells": 0,
            "num_rejected_cells": 0,
            "total_num_cells": 6,
        },
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 95,
            "period": {"den": 1000, "num": 100},
            "start": {
                "secsSinceEpoch": {"den": 1000, "num": 30100},
                "utcOffset": 0,
            },
            "sampling_rate": 10.0,
            "end": {
                "secsSinceEpoch": {"den": 1000, "num": 39600},
                "utcOffset": 0,
            },
        },
        "spacingInfo": {
            "numPixels": {"x": 128, "y": 128},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_raw_cellset_metadata0 == act_raw_cellset_metadata0

    exp_raw_cellset_metadata1 = {
        "metrics": {
            "num_accepted_cells": 6,
            "num_undecided_cells": 0,
            "num_rejected_cells": 0,
            "total_num_cells": 6,
        },
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 34,
            "period": {"den": 1000, "num": 100},
            "start": {
                "secsSinceEpoch": {"den": 1000, "num": 62300},
                "utcOffset": 0,
            },
            "sampling_rate": 10.0,
            "end": {
                "secsSinceEpoch": {"den": 1000, "num": 65700},
                "utcOffset": 0,
            },
        },
        "spacingInfo": {
            "numPixels": {"x": 128, "y": 128},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_raw_cellset_metadata1 == act_raw_cellset_metadata1

    exp_raw_cellset_metadata2 = {
        "metrics": {
            "num_accepted_cells": 6,
            "num_undecided_cells": 0,
            "num_rejected_cells": 0,
            "total_num_cells": 6,
        },
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 100,
            "period": {"den": 1000, "num": 100},
            "start": {"secsSinceEpoch": {"den": 1, "num": 0}, "utcOffset": 0},
            "sampling_rate": 10.0,
            "end": {"secsSinceEpoch": {"den": 1, "num": 10}, "utcOffset": 0},
        },
        "spacingInfo": {
            "numPixels": {"x": 128, "y": 128},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_raw_cellset_metadata2 == act_raw_cellset_metadata2

    # denoised cell sets metadata
    exp_denoised_cellset_metadata0 = {
        "metrics": {
            "num_accepted_cells": 6,
            "num_undecided_cells": 0,
            "num_rejected_cells": 0,
            "total_num_cells": 6,
        },
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 95,
            "period": {"den": 1000, "num": 100},
            "start": {
                "secsSinceEpoch": {"den": 1000, "num": 30100},
                "utcOffset": 0,
            },
            "sampling_rate": 10.0,
            "end": {
                "secsSinceEpoch": {"den": 1000, "num": 39600},
                "utcOffset": 0,
            },
        },
        "spacingInfo": {
            "numPixels": {"x": 128, "y": 128},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_denoised_cellset_metadata0 == act_denoised_cellset_metadata0

    exp_denoised_cellset_metadata1 = {
        "metrics": {
            "num_accepted_cells": 6,
            "num_undecided_cells": 0,
            "num_rejected_cells": 0,
            "total_num_cells": 6,
        },
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 34,
            "period": {"den": 1000, "num": 100},
            "start": {
                "secsSinceEpoch": {"den": 1000, "num": 62300},
                "utcOffset": 0,
            },
            "sampling_rate": 10.0,
            "end": {
                "secsSinceEpoch": {"den": 1000, "num": 65700},
                "utcOffset": 0,
            },
        },
        "spacingInfo": {
            "numPixels": {"x": 128, "y": 128},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_denoised_cellset_metadata1 == act_denoised_cellset_metadata1

    exp_denoised_cellset_metadata2 = {
        "metrics": {
            "num_accepted_cells": 6,
            "num_undecided_cells": 0,
            "num_rejected_cells": 0,
            "total_num_cells": 6,
        },
        "timingInfo": {
            "blank": [],
            "cropped": [],
            "dropped": [],
            "numTimes": 100,
            "period": {"den": 1000, "num": 100},
            "start": {"secsSinceEpoch": {"den": 1, "num": 0}, "utcOffset": 0},
            "sampling_rate": 10.0,
            "end": {"secsSinceEpoch": {"den": 1, "num": 10}, "utcOffset": 0},
        },
        "spacingInfo": {
            "numPixels": {"x": 128, "y": 128},
            "pixelSize": {
                "x": {"den": 1, "num": 3},
                "y": {"den": 1, "num": 3},
            },
            "topLeft": {"x": {"den": 1, "num": 0}, "y": {"den": 1, "num": 0}},
        },
        "microscope": {"focus": None},
    }
    assert exp_denoised_cellset_metadata2 == act_denoised_cellset_metadata2

    # event sets metadata
    exp_eventset_metadata0 = {
        "metrics": {"total_num_cells": 6},
        "timingInfo": {
            "start": {
                "secsSinceEpoch": {"den": 1000, "num": 30100},
                "utcOffset": 0,
            },
            "end": {
                "secsSinceEpoch": {"den": 1000, "num": 39600},
                "utcOffset": 0,
            },
            "numSamples": [32, 19, 24, 25, 20, 18],
            "numFrames": 95,
            "period": {"den": 1000, "num": 100},
            "sampling_rate": 10.0,
            "dropped": [],
            "cropped": [],
        },
        "microscope": {"focus": None},
    }
    assert exp_eventset_metadata0 == act_eventset_metadata0

    exp_eventset_metadata1 = {
        "metrics": {"total_num_cells": 6},
        "timingInfo": {
            "start": {
                "secsSinceEpoch": {"den": 1000, "num": 62300},
                "utcOffset": 0,
            },
            "end": {
                "secsSinceEpoch": {"den": 1000, "num": 65700},
                "utcOffset": 0,
            },
            "numSamples": [8, 4, 5, 4, 10, 5],
            "numFrames": 34,
            "period": {"den": 1000, "num": 100},
            "sampling_rate": 10.0,
            "dropped": [],
            "cropped": [],
        },
        "microscope": {"focus": None},
    }
    assert exp_eventset_metadata1 == act_eventset_metadata1

    exp_eventset_metadata2 = {
        "metrics": {"total_num_cells": 6},
        "timingInfo": {
            "start": {"secsSinceEpoch": {"den": 1, "num": 0}, "utcOffset": 0},
            "end": {
                "secsSinceEpoch": {"den": 1000, "num": 10000},
                "utcOffset": 0,
            },
            "numSamples": [40, 11, 13, 11, 22, 12],
            "numFrames": 100,
            "period": {"den": 1000, "num": 100},
            "sampling_rate": 10.0,
            "dropped": [],
            "cropped": [],
        },
        "microscope": {"focus": None},
    }
    assert exp_eventset_metadata2 == act_eventset_metadata2


def test_caiman_cnmfe_workflow_isxd_extra_properties():
    """Verify that the CaImAn CNMF-E workflow correctly copies the extra properties
    in the json metadata of input isxd files to output isxd files"""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd"),
        os.path.join(DATA_DIR, "movie_part2_108x123x63.isxd"),
    ]

    caiman_workflow(
        input_movie_files=input_movie_files,
        overwrite_analysis_table_params=False,
        min_corr=0.5,
        min_pnr=5,
        gSig_filt=3,
    )

    input_movie0_metadata = read_isxd_metadata(input_movie_files[0])
    input_movie1_metadata = read_isxd_metadata(input_movie_files[1])

    # first part of output series
    output0_files = [
        "cellset_raw.000.isxd",
        "cellset_denoised.000.isxd",
        "neural_events.000.isxd",
    ]
    for output0_file in output0_files:
        output0_file_metadata = read_isxd_metadata(output0_file)
        assert (
            "extraProperties" in output0_file_metadata
            and output0_file_metadata["extraProperties"]
            == input_movie0_metadata["extraProperties"]
        )

    # second part of output series
    output1_files = [
        "cellset_raw.001.isxd",
        "cellset_denoised.001.isxd",
        "neural_events.001.isxd",
    ]
    for output1_file in output1_files:
        output1_file_metadata = read_isxd_metadata(output1_file)
        assert (
            "extraProperties" in output1_file_metadata
            and output1_file_metadata["extraProperties"]
            == input_movie1_metadata["extraProperties"]
        )
