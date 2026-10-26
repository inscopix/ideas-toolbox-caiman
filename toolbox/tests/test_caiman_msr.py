import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ideas.exceptions import IdeasError as CoreIdeasError
from scipy.sparse import csc_matrix

import toolbox.tools.caiman_msr as caiman_msr_tool
import toolbox.utils.io as io_utils
from toolbox.tools.caiman_msr import (
    _pad_footprints_to_dims,
    _pad_template_image_to_dims,
    caiman_msr,
)
from toolbox.utils.exceptions import IdeasError

data_dir = "/ideas/data"


@pytest.mark.parametrize(
    "cellset_paths,template_paths,eventset_paths,contains_series,align_flag,"
    "use_opt_flow,max_shifts,max_thr,max_dist,thresh_cost,enclosed_thr,"
    "use_cell_status,min_n_regist_sess,fp_thr_method,fp_thr,cmap_div,cmap,"
    "show_grid,ticks_step,n_sample_cells,",
    [
        [
            [
                "sample_cellset_msr_1.isxd",
                "sample_cellset_msr_2.isxd",
            ],
            [
                "sample_template_image_msr_1.tif",
                "sample_template_image_msr_2.tif",
            ],
            [
                "sample_eventset_msr_1.isxd",
                "sample_eventset_msr_2.isxd",
            ],
            False,
            True,
            False,
            80,
            0.0,
            20.0,
            0.8,
            0.0,
            "accepted",
            1,
            "nrg",
            0.9,
            "bwr",
            "gray",
            True,
            100,
            20,
        ],
        [
            [
                "sample_cellset_msr_1.isxd",
                "sample_cellset_msr_2.isxd",
            ],
            [
                "sample_template_image_msr_1.tif",
                "sample_template_image_msr_2.tif",
            ],
            [""],
            False,
            True,
            True,
            80,
            0.0,
            20.0,
            0.8,
            0.0,
            "accepted",
            1,
            "nrg",
            0.9,
            "bwr",
            "gray",
            True,
            100,
            20,
        ],
        [
            [
                "sample_cellset_msr_1.hdf5",
                "sample_cellset_msr_2.hdf5",
            ],
            [
                "sample_template_image_msr_1.tif",
                "sample_template_image_msr_2.tif",
            ],
            [""],
            False,
            True,
            False,
            80,
            0.0,
            20.0,
            0.8,
            0.0,
            "accepted",
            2,
            "nrg",
            0.8,
            "seismic",
            "gray",
            True,
            128,
            20,
        ],
        [
            [
                "sample_cellset_msr_1a.isxd",
                "sample_cellset_msr_1b.isxd",
                "sample_cellset_msr_2.isxd",
            ],
            [
                "sample_template_image_msr_1.tif",
                "sample_template_image_msr_2.tif",
            ],
            [""],
            True,
            True,
            False,
            80,
            0.0,
            20.0,
            0.8,
            0.0,
            "accepted",
            2,
            "nrg",
            0.8,
            "seismic",
            "gray",
            True,
            128,
            20,
        ],
        [
            [
                "sample_cellset_msr_1a.isxd",
                "sample_cellset_msr_1b.isxd",
                "sample_cellset_msr_2.isxd",
            ],
            [
                "sample_template_image_msr_1.tif",
                "sample_template_image_msr_2.tif",
            ],
            [
                "sample_eventset_msr_1a.isxd",
                "sample_eventset_msr_1b.isxd",
                "sample_eventset_msr_2a.isxd",
            ],
            True,
            True,
            False,
            80,
            0.0,
            20.0,
            0.8,
            0.0,
            "accepted",
            2,
            "nrg",
            0.8,
            "seismic",
            "gray",
            True,
            128,
            20,
        ],
    ],
)
def test_caiman_msr(
    cellset_paths,
    template_paths,
    eventset_paths,
    contains_series,
    align_flag,
    use_opt_flow,
    max_shifts,
    max_thr,
    max_dist,
    thresh_cost,
    enclosed_thr,
    use_cell_status,
    min_n_regist_sess,
    fp_thr_method,
    fp_thr,
    cmap_div,
    cmap,
    show_grid,
    ticks_step,
    n_sample_cells,
    output_dir,
):
    """
    Test that caiman_msr() runs properly for various configurations of
    input files and parameters.
    """
    cellset_paths = [f"{data_dir}/{x}" for x in cellset_paths]
    for idx, f in enumerate(cellset_paths):
        dest = str(Path.cwd() / Path(f).name)
        shutil.copy(f, dest)
        cellset_paths[idx] = dest

    template_paths = [f"{data_dir}/{x}" for x in template_paths]
    for idx, f in enumerate(template_paths):
        dest = str(Path.cwd() / Path(f).name)
        shutil.copy(f, dest)
        template_paths[idx] = dest

    if len(eventset_paths) > 0 and len(eventset_paths[0]) > 0:
        eventset_paths = [f"{data_dir}/{x}" for x in eventset_paths]
        for idx, f in enumerate(eventset_paths):
            dest = str(Path.cwd() / Path(f).name)
            shutil.copy(f, dest)
            eventset_paths[idx] = dest

    caiman_msr(
        cellset_paths=cellset_paths,
        template_paths=template_paths,
        eventset_paths=eventset_paths,
        contains_series=contains_series,
        align_flag=align_flag,
        use_opt_flow=use_opt_flow,
        max_shifts=max_shifts,
        max_thr=max_thr,
        max_dist=max_dist,
        thresh_cost=thresh_cost,
        enclosed_thr=enclosed_thr,
        use_cell_status=use_cell_status,
        min_n_regist_sess=min_n_regist_sess,
        fp_thr_method=fp_thr_method,
        fp_thr=fp_thr,
        cmap_div=cmap_div,
        cmap=cmap,
        show_grid=show_grid,
        ticks_step=ticks_step,
        n_sample_cells=n_sample_cells,
        output_dir=output_dir,
    )

    fname_out_list = [
        "CaImAn_MSR_output.h5",
        "alignment_qc_metrics.csv",
        "CaImAn_MSR_metrics.csv",
        "preview_registered_cells_on_proj_images.svg",
        "preview_registered_cells_on_unified_image.svg",
        "preview_number_registered_cells_stacked_bars.svg",
        "preview_distance_overlap_histograms.svg",
        "preview_registered_traces.svg",
        "preview_template_images_alignment.svg",
        "output_metadata.json",
    ]
    fname_prefix_list = [
        "preview_footprints_cellset_registered",
        "preview_traces_cellset_registered",
    ]
    if len(eventset_paths) > 0 and len(eventset_paths[0]) > 0:
        fname_out_list.append("preview_registered_events.svg")
        fname_prefix_list.append("preview_eventset_registered")
    n_sessions = len(cellset_paths)
    for fname_prefix in fname_prefix_list:
        for idx in range(n_sessions):
            fname_out_list.append(f"{fname_prefix}.{idx:03g}.svg")

    for fname_out in fname_out_list:
        assert (
            Path(output_dir) / fname_out
        ).exists(), f"Could not find {fname_out} in {output_dir}"


def test_pad_fov_helpers():
    """Verify template and footprint FoV padding for each corner anchor."""
    template = np.array([[1, 2], [3, 4]], dtype=float)
    target_dims = (3, 4)
    fp = csc_matrix(np.array([[1.0], [2.0], [3.0], [4.0]]))

    anchor_to_expected_template = {
        "top-left": ((0, 1), (0, 2)),
        "top-right": ((1, 0), (0, 2)),
        "bottom-left": ((0, 1), (2, 0)),
        "bottom-right": ((1, 0), (2, 0)),
    }
    anchor_to_expected_fp = {
        "top-left": ((0, 2), (0, 1), (0, 0)),
        "top-right": ((0, 2), (1, 0), (0, 0)),
        "bottom-left": ((2, 0), (0, 1), (0, 0)),
        "bottom-right": ((2, 0), (1, 0), (0, 0)),
    }

    for padding_anchor in anchor_to_expected_template:
        padded_template = _pad_template_image_to_dims(
            template_img=template,
            target_dims=target_dims,
            padding_anchor=padding_anchor,
        )
        assert padded_template.shape == target_dims
        expected_template = np.pad(
            template,
            anchor_to_expected_template[padding_anchor],
            mode="edge",
        )
        assert np.array_equal(padded_template, expected_template)

        padded_fp = _pad_footprints_to_dims(
            footprints_mat=fp,
            source_dims=(2, 2),
            target_dims=target_dims,
            padding_anchor=padding_anchor,
        )
        fp_arr = padded_fp.toarray().reshape(target_dims[1], target_dims[0], 1)
        assert fp_arr.shape[:2] == (4, 3)
        expected_fp = np.pad(
            fp.toarray().reshape(2, 2, 1),
            anchor_to_expected_fp[padding_anchor],
            mode="constant",
        )
        assert np.array_equal(fp_arr, expected_fp)


def test_padding_helper_rejects_disabled_anchor():
    """Reject direct helper calls using padding_anchor='disabled'."""
    with pytest.raises(IdeasError, match="not valid for direct padding"):
        caiman_msr_tool._compute_padding_by_anchor(
            source_dims=(2, 2),
            target_dims=(3, 4),
            padding_anchor="disabled",
        )


def test_save_registered_rejects_disabled_output_anchor_with_output_fov_dims():
    """Fail fast when output canvas is requested but output anchor is disabled."""
    with pytest.raises(
        CoreIdeasError, match="When `output_fov_dims` is provided"
    ):
        io_utils.save_registered_cellsets_eventsets(
            cellset_paths=[],
            eventset_paths=[],
            df_assignments=pd.DataFrame(),
            output_dir="/tmp",
            output_fov_dims=(3, 4),
            output_padding_anchor="disabled",
        )


def test_series_fov_consistency_across_all_recordings(monkeypatch, output_dir):
    """Fail when any recording in a series mismatches its matched template FoV."""
    cellset_paths = [
        "series_day1_part1.isxd",
        "series_day1_part2.isxd",
        "series_day2_part1.isxd",
    ]
    template_paths = [
        "template_day1.isxd",
        "template_day2.isxd",
    ]

    def _fake_validate_caiman_msr_inputs(**kwargs):
        return (
            [cellset_paths[0], cellset_paths[2]],
            [None, None, None],
            None,
            1,
            {0: 0, 1: 0, 2: 1},
        )

    monkeypatch.setattr(
        caiman_msr_tool.utilities,
        "validate_caiman_msr_inputs",
        _fake_validate_caiman_msr_inputs,
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "get_cellset_fov_dims",
        lambda *, cellset_path: {
            cellset_paths[0]: (50, 40),
            cellset_paths[1]: (49, 40),
            cellset_paths[2]: (55, 42),
        }[cellset_path],
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "load_template_image",
        lambda *, template_path: np.zeros(
            {
                template_paths[0]: (50, 40),
                template_paths[1]: (55, 42),
            }[template_path]
        ),
    )

    with pytest.raises(
        IdeasError,
        match="within each session for all series recordings",
    ):
        caiman_msr(
            cellset_paths=cellset_paths,
            template_paths=template_paths,
            eventset_paths=[""],
            contains_series=True,
            output_dir=output_dir,
        )


def test_disabled_padding_anchor_raises_on_cross_session_mismatch(
    monkeypatch, output_dir
):
    """Fail on cross-session FoV mismatch when padding_anchor is disabled."""
    cellset_paths = ["day1.isxd", "day2.isxd"]
    template_paths = ["temp1.isxd", "temp2.isxd"]

    monkeypatch.setattr(
        caiman_msr_tool.utilities,
        "validate_caiman_msr_inputs",
        lambda **kwargs: (
            cellset_paths,
            [None, None],
            None,
            1,
            None,
        ),
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "get_footprints_from_cellset",
        lambda **kwargs: (
            csc_matrix(np.array([[1.0], [0.0], [0.0], [1.0]])),
            ["C0"],
            {"day1.isxd": (2, 2), "day2.isxd": (3, 2)}[kwargs["cellset_path"]],
        ),
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "load_template_image",
        lambda **kwargs: np.zeros(
            {
                "temp1.isxd": (2, 2),
                "temp2.isxd": (3, 2),
            }[kwargs["template_path"]]
        ),
    )

    with pytest.raises(IdeasError, match="padding_anchor=disabled"):
        caiman_msr(
            cellset_paths=cellset_paths,
            template_paths=template_paths,
            eventset_paths=[""],
            contains_series=False,
            output_dir=output_dir,
        )


def test_padding_tolerance_raises_on_large_cross_session_mismatch(
    monkeypatch, output_dir
):
    """Fail when required cross-session padding exceeds tolerance."""
    cellset_paths = ["day1.isxd", "day2.isxd"]
    template_paths = ["temp1.isxd", "temp2.isxd"]

    monkeypatch.setattr(
        caiman_msr_tool.utilities,
        "validate_caiman_msr_inputs",
        lambda **kwargs: (
            cellset_paths,
            [None, None],
            None,
            1,
            None,
        ),
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "get_footprints_from_cellset",
        lambda **kwargs: (
            csc_matrix(np.array([[1.0], [0.0], [0.0], [1.0]])),
            ["C0"],
            {"day1.isxd": (2, 2), "day2.isxd": (20, 2)}[
                kwargs["cellset_path"]
            ],
        ),
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "load_template_image",
        lambda **kwargs: np.zeros(
            {
                "temp1.isxd": (2, 2),
                "temp2.isxd": (20, 2),
            }[kwargs["template_path"]]
        ),
    )

    with pytest.raises(IdeasError, match="padding_tolerance"):
        caiman_msr(
            cellset_paths=cellset_paths,
            template_paths=template_paths,
            eventset_paths=[""],
            contains_series=False,
            padding_anchor="top-left",
            output_dir=output_dir,
        )


@pytest.mark.parametrize(
    "padding_anchor,template_pad,fp_pad",
    [
        ("top-left", ((0, 1), (0, 2)), ((0, 2), (0, 1), (0, 0))),
        ("top-right", ((1, 0), (0, 2)), ((0, 2), (1, 0), (0, 0))),
        ("bottom-left", ((0, 1), (2, 0)), ((2, 0), (0, 1), (0, 0))),
        ("bottom-right", ((1, 0), (2, 0)), ((2, 0), (1, 0), (0, 0))),
    ],
)
def test_padding_end_to_end_consistency_for_all_anchors(
    monkeypatch, output_dir, padding_anchor, template_pad, fp_pad
):
    """Verify input padding and output canvas settings stay consistent per anchor."""
    calls = {"register_called": False, "save_registered_called": False}

    monkeypatch.setattr(
        caiman_msr_tool.utilities,
        "validate_caiman_msr_inputs",
        lambda **kwargs: (
            ["day1.isxd", "day2.isxd"],
            [None, None],
            None,
            1,
            None,
        ),
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "get_footprints_from_cellset",
        lambda **kwargs: (
            csc_matrix(
                np.array([[1.0], [0.0], [0.0], [2.0]])
                if kwargs["cellset_path"] == "day1.isxd"
                else np.array(
                    [
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [2.0],
                    ]
                )
            ),
            ["C0"],
            {"day1.isxd": (2, 2), "day2.isxd": (3, 4)}[kwargs["cellset_path"]],
        ),
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "load_template_image",
        lambda **kwargs: (
            np.array([[1.0, 2.0], [3.0, 4.0]])
            if kwargs["template_path"] == "temp1.isxd"
            else np.array(
                [
                    [10.0, 20.0, 30.0, 40.0],
                    [50.0, 60.0, 70.0, 80.0],
                    [90.0, 100.0, 110.0, 120.0],
                ]
            )
        ),
    )

    def _fake_register_multisession(**kwargs):
        calls["register_called"] = True
        assert kwargs["dims"] == (3, 4)
        assert kwargs["templates"][0].shape == (3, 4)
        assert kwargs["templates"][1].shape == (3, 4)
        assert kwargs["A"][0].shape[0] == 12
        assert kwargs["A"][1].shape[0] == 12

        expected_template0 = np.pad(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            template_pad,
            mode="edge",
        )
        assert np.array_equal(kwargs["templates"][0], expected_template0)

        expected_fp0 = np.pad(
            np.array([[1.0], [0.0], [0.0], [2.0]]).reshape(2, 2, 1),
            fp_pad,
            mode="constant",
        )
        fp0 = kwargs["A"][0].toarray().reshape(4, 3, 1)
        assert np.array_equal(fp0, expected_fp0)
        return (
            np.array([[0.0]]),
            np.array([[0, 0]], dtype=float),
            [[0]],
            None,
            [[np.array([[0.0]])]],
            [[np.array([[0.0]])]],
            kwargs["templates"][:-1],
            [np.array([[0.0, 0.0]])],
        )

    monkeypatch.setattr(
        caiman_msr_tool, "register_multisession", _fake_register_multisession
    )
    monkeypatch.setattr(
        caiman_msr_tool.io, "save_msr_output", lambda **kwargs: None
    )
    monkeypatch.setattr(
        caiman_msr_tool.utilities,
        "transform_assignments_matrix",
        lambda **kwargs: (np.empty((0, 0)), 0),
    )

    def _fake_save_registered_cellsets_eventsets(**kwargs):
        calls["save_registered_called"] = True
        assert kwargs["output_fov_dims"] == (3, 4)
        assert kwargs["output_padding_anchor"] == padding_anchor
        return ([], [])

    monkeypatch.setattr(
        caiman_msr_tool.io,
        "save_registered_cellsets_eventsets",
        _fake_save_registered_cellsets_eventsets,
    )
    monkeypatch.setattr(
        caiman_msr_tool.io, "save_alignment_metrics_csv", lambda **kwargs: None
    )
    monkeypatch.setattr(
        caiman_msr_tool.io, "save_msr_metrics_csv", lambda **kwargs: None
    )
    monkeypatch.setattr(
        caiman_msr_tool.metadata,
        "create_msr_output_metadata",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        caiman_msr_tool.previews,
        "create_CaImAn_MSR_preview_figures",
        lambda **kwargs: None,
    )

    caiman_msr(
        cellset_paths=["day1.isxd", "day2.isxd"],
        template_paths=["temp1.isxd", "temp2.isxd"],
        eventset_paths=[""],
        padding_anchor=padding_anchor,
        output_dir=output_dir,
    )
    assert calls["register_called"]
    assert calls["save_registered_called"]


def test_no_padding_keeps_original_output_canvas(monkeypatch, output_dir):
    """Verify output canvas override is disabled when padding is not applied."""
    calls = {"save_registered_called": False}

    monkeypatch.setattr(
        caiman_msr_tool.utilities,
        "validate_caiman_msr_inputs",
        lambda **kwargs: (
            ["day1.isxd", "day2.isxd"],
            [None, None],
            None,
            1,
            None,
        ),
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "get_footprints_from_cellset",
        lambda **kwargs: (
            csc_matrix(np.array([[1.0], [0.0], [0.0], [2.0]])),
            ["C0"],
            (2, 2),
        ),
    )
    monkeypatch.setattr(
        caiman_msr_tool.io,
        "load_template_image",
        lambda **kwargs: np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    monkeypatch.setattr(
        caiman_msr_tool,
        "register_multisession",
        lambda **kwargs: (
            np.array([[0.0]]),
            np.array([[0, 0]], dtype=float),
            [[0]],
            None,
            [[np.array([[0.0]])]],
            [[np.array([[0.0]])]],
            kwargs["templates"][:-1],
            [np.array([[0.0, 0.0]])],
        ),
    )
    monkeypatch.setattr(
        caiman_msr_tool.io, "save_msr_output", lambda **kwargs: None
    )
    monkeypatch.setattr(
        caiman_msr_tool.utilities,
        "transform_assignments_matrix",
        lambda **kwargs: (np.empty((0, 0)), 0),
    )

    def _fake_save_registered_cellsets_eventsets(**kwargs):
        calls["save_registered_called"] = True
        assert kwargs["output_fov_dims"] is None
        assert kwargs["output_padding_anchor"] == "disabled"
        return ([], [])

    monkeypatch.setattr(
        caiman_msr_tool.io,
        "save_registered_cellsets_eventsets",
        _fake_save_registered_cellsets_eventsets,
    )
    monkeypatch.setattr(
        caiman_msr_tool.io, "save_alignment_metrics_csv", lambda **kwargs: None
    )
    monkeypatch.setattr(
        caiman_msr_tool.io, "save_msr_metrics_csv", lambda **kwargs: None
    )
    monkeypatch.setattr(
        caiman_msr_tool.metadata,
        "create_msr_output_metadata",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        caiman_msr_tool.previews,
        "create_CaImAn_MSR_preview_figures",
        lambda **kwargs: None,
    )

    caiman_msr(
        cellset_paths=["day1.isxd", "day2.isxd"],
        template_paths=["temp1.isxd", "temp2.isxd"],
        eventset_paths=[""],
        padding_anchor="disabled",
        output_dir=output_dir,
    )
    assert calls["save_registered_called"]
