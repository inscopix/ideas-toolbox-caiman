import os
import shutil
import pytest
import isx
import numpy as np
from toolbox.tools.caiman_isx_academic import source_extraction
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
        "output_metadata.json",
        "output_manifest.json",
    ]:
        if os.path.exists(f):
            os.remove(f)

    file_ext = [".isxd", ".svg", ".mmap", ".mp4", ".csv"]
    for f in os.listdir("."):
        if os.path.splitext(f)[-1] in file_ext:
            os.remove(f)


def test_caiman_source_extraction_no_cells_identified():
    """Verify that the CaImAn source extraction algorithm correctly handle scenario where no cells are identified."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd")
    ]

    with pytest.raises(IdeasError) as e:
        source_extraction(
            input_movie_files=input_movie_files,
            overwrite_analysis_table_params=False,
            min_corr=0.85,
            min_pnr=11,
        )
    assert str(e.value) == "No cells were identified"


def test_caiman_source_extraction_single_isxd_file():
    """Verify that the CaImAn source extraction correctly identifies cells in a single ISXD movie."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd")
    ]

    source_extraction(
        input_movie_files=input_movie_files,
        min_corr=0.5,
        min_pnr=5,
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
    exp_num_cells = 67
    act_num_cells = len(model.estimates.C)
    assert act_num_cells == exp_num_cells
    assert model.estimates.idx_components is None
    assert model.estimates.idx_components_bad is None

    # validate output ISXD files
    exp_num_frames = 100
    exp_width = 108
    exp_height = 123
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["undecided"] * act_num_cells)
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


def test_caiman_source_extraction_isxd_movie_series():
    """Verify that the CaImAn source extraction correctly processes an ISXD movie series."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd"),
        os.path.join(DATA_DIR, "movie_part2_108x123x63.isxd"),
    ]

    source_extraction(
        input_movie_files=input_movie_files, min_corr=0.5, min_pnr=5
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
    exp_num_cells = 146
    act_num_cells = len(model.estimates.C)
    assert act_num_cells == exp_num_cells
    assert model.estimates.idx_components is None
    assert model.estimates.idx_components_bad is None

    # validate output ISXD files
    exp_num_frames_part0 = 100
    exp_num_frames_part1 = 63
    exp_width = 108
    exp_height = 123
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["undecided"] * act_num_cells)

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


def test_caiman_source_extraction_single_tiff_file():
    """Verify that the CaImAn source extraction correctly identifies cells in a single TIFF movie."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.tiff")
    ]

    source_extraction(
        input_movie_files=input_movie_files,
        overwrite_analysis_table_params=False,
        min_corr=0.6,
        min_pnr=6,
        fr=10,
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
    exp_num_cells = 32
    act_num_cells = len(model.estimates.C)
    assert act_num_cells == exp_num_cells
    assert model.estimates.idx_components is None
    assert model.estimates.idx_components_bad is None

    # validate output ISXD files
    exp_num_frames = 100
    exp_width = 108
    exp_height = 123
    exp_cell0_status = "undecided"
    exp_cell0_name = "C000"
    exp_cell3_status = "undecided"
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


def test_caiman_source_extraction_single_avi_movie():
    """Verify that the CaImAn source extraction can be applied to a 1P avi movie."""
    input_movie_files = [os.path.join(DATA_DIR, "movie_part1_108x122x200.avi")]

    source_extraction(
        input_movie_files=input_movie_files,
        min_corr=0.6,
        min_pnr=7,
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
    exp_num_cells = 50
    act_num_cells = len(model.estimates.C)
    assert act_num_cells == exp_num_cells
    assert model.estimates.idx_components is None
    assert model.estimates.idx_components_bad is None

    # validate output ISXD files
    exp_num_frames = 200
    exp_width = 108
    exp_height = 122
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["undecided"] * act_num_cells)
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


def test_caiman_source_extraction_params_from_file():
    """Verify that the CaImAn source extraction can be applied to a movie using parameters obtained from a json file."""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd")
    ]
    input_parameters_file = [
        os.path.join(DATA_DIR, "params_source_extraction.json")
    ]

    source_extraction(
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
    exp_num_cells = 10
    act_num_cells = len(model.estimates.C)
    assert act_num_cells == exp_num_cells
    assert model.estimates.idx_components is None
    assert model.estimates.idx_components_bad is None

    # validate output ISXD files
    exp_num_frames = 100
    exp_width = 108
    exp_height = 123
    exp_cell0_name = "C000"
    exp_cell3_name = "C003"
    exp_cell_statuses = np.array(["undecided"] * act_num_cells)
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
    assert exp_spacing.num_pixels == raw_cellset.spacing.num_pixels

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
    assert exp_spacing.num_pixels == denoised_cellset.spacing.num_pixels

    eventset = isx.EventSet.read("neural_events.000.isxd")
    assert eventset.num_cells == exp_num_cells
    assert eventset.timing.num_samples == exp_num_frames
    assert eventset.get_cell_name(0) == exp_cell0_name
    assert eventset.get_cell_name(3) == exp_cell3_name
    assert exp_timing == eventset.timing


def test_caiman_source_extraction_isxd_extra_properties():
    """Verify that the CaImAn source extraction correctly copies the extra properties
    in the json metadata of input isxd files to output isxd files"""
    input_movie_files = [
        os.path.join(DATA_DIR, "movie_part1_108x123x100.isxd"),
        os.path.join(DATA_DIR, "movie_part2_108x123x63.isxd"),
    ]

    source_extraction(
        input_movie_files=input_movie_files, min_corr=0.5, min_pnr=5
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
