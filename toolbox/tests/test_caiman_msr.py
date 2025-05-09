# from glob import glob
import os
import pytest
import shutil
from toolbox.tools.caiman_msr import caiman_msr

data_dir = "/ideas/data"


@pytest.mark.parametrize(
    "cellset_paths,template_paths,eventset_paths,align_flag,use_opt_flow,"
    "max_shifts,max_thr,max_dist,thresh_cost,enclosed_thr,use_cell_status,"
    "min_n_regist_sess,fp_thr_method,fp_thr,cmap_div,cmap,show_grid,"
    "ticks_step,n_sample_cells,",
    [
        [
            [
                "IM13_session1_2024-10-07_cellset_10cells-10s-subset.isxd",
                "IM13_session2_2024-10-10_cellset_10cells-10s-subset.isxd",
                "IM13_session4_2024-10-28_cellset_10cells-10s-subset.isxd",
                "IM13_session5_2024-10-31_cellset_10cells-10s-subset.isxd",
            ],
            [
                "IM13_session1_2024-10-07_local_corr_img.isxd",
                "IM13_session2_2024-10-10_local_corr_img.isxd",
                "IM13_session4_2024-10-28_local_corr_img.isxd",
                "IM13_session5_2024-10-31_local_corr_img.isxd",
            ],
            [
                "IM13_session1_2024-10-07_eventset_10cells-10s-subset.isxd",
                "IM13_session2_2024-10-10_eventset_10cells-10s-subset.isxd",
                "IM13_session4_2024-10-28_eventset_10cells-10s-subset.isxd",
                "IM13_session5_2024-10-31_eventset_10cells-10s-subset.isxd",
            ],
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
                "IM13_session1_2024-10-07_cellset_10cells-10s-subset.isxd",
                "IM13_session2_2024-10-10_cellset_10cells-10s-subset.isxd",
                "IM13_session4_2024-10-28_cellset_10cells-10s-subset.isxd",
                "IM13_session5_2024-10-31_cellset_10cells-10s-subset.isxd",
            ],
            [
                "IM13_session1_2024-10-07_local_corr_img.isxd",
                "IM13_session2_2024-10-10_local_corr_img.isxd",
                "IM13_session4_2024-10-28_local_corr_img.isxd",
                "IM13_session5_2024-10-31_local_corr_img.isxd",
            ],
            [
                "IM13_session1_2024-10-07_eventset_10cells-10s-subset.isxd",
                "IM13_session2_2024-10-10_eventset_10cells-10s-subset.isxd",
                "IM13_session4_2024-10-28_eventset_10cells-10s-subset.isxd",
                "IM13_session5_2024-10-31_eventset_10cells-10s-subset.isxd",
            ],
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
                "IM13_session1_2024-10-07_cellset_10cells-10s-subset.isxd",
                "IM13_session2_2024-10-10_cellset_10cells-10s-subset.isxd",
                "IM13_session4_2024-10-28_cellset_10cells-10s-subset.isxd",
                "IM13_session5_2024-10-31_cellset_10cells-10s-subset.isxd",
            ],
            [
                "IM13_session1_2024-10-07_local_corr_img.isxd",
                "IM13_session2_2024-10-10_local_corr_img.isxd",
                "IM13_session4_2024-10-28_local_corr_img.isxd",
                "IM13_session5_2024-10-31_local_corr_img.isxd",
            ],
            [
                "IM13_session1_2024-10-07_eventset_10cells-10s-subset.isxd",
                "IM13_session2_2024-10-10_eventset_10cells-10s-subset.isxd",
                "IM13_session4_2024-10-28_eventset_10cells-10s-subset.isxd",
                "IM13_session5_2024-10-31_eventset_10cells-10s-subset.isxd",
            ],
            True,
            False,
            80,
            0.0,
            20.0,
            0.8,
            0.0,
            "accepted",
            4,
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
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        cellset_paths[idx] = dest

    template_paths = [f"{data_dir}/{x}" for x in template_paths]
    for idx, f in enumerate(template_paths):
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        template_paths[idx] = dest

    if len(eventset_paths) > 0 and len(eventset_paths[0]) > 0:
        eventset_paths = [f"{data_dir}/{x}" for x in eventset_paths]
        for idx, f in enumerate(eventset_paths):
            dest = os.path.join(os.getcwd(), os.path.basename(f))
            shutil.copy(f, dest)
            eventset_paths[idx] = dest

    caiman_msr(
        cellset_paths=cellset_paths,
        template_paths=template_paths,
        eventset_paths=eventset_paths,
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
        "preview_registered_events.svg",
        "preview_template_images_alignment.svg",
        "output_metadata.json",
    ]
    fname_prefix_list = [
        "preview_footprints_cellset_registered",
        "preview_traces_cellset_registered",
    ]
    if len(eventset_paths) > 0 and len(eventset_paths[0]) > 0:
        fname_prefix_list.append("preview_eventset_registered")
    n_sessions = len(cellset_paths)
    for fname_prefix in fname_prefix_list:
        for idx in range(n_sessions):
            fname_out_list.append(f"{fname_prefix}.{idx:03g}.svg")

    for fname_out in fname_out_list:
        assert os.path.exists(
            os.path.join(output_dir, fname_out)
        ), f"Could not find {fname_out} in {output_dir}"
