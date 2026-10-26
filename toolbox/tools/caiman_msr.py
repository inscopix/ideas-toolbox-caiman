import logging
from pathlib import Path

import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Union
from caiman.base.rois import register_multisession
from scipy.sparse import csc_matrix

from toolbox.utils import io, metadata, previews, utilities
from toolbox.utils.exceptions import IdeasError

logger = logging.getLogger()


def _compute_padding_by_anchor(source_dims, target_dims, padding_anchor):
    """Compute per-side padding based on a corner anchor."""
    source_w, source_h = source_dims
    target_w, target_h = target_dims
    pad_w = target_w - source_w
    pad_h = target_h - source_h
    if pad_w < 0 or pad_h < 0:
        raise IdeasError(
            f"Target FoV dimensions {target_dims} must be >= source dimensions "
            f"{source_dims}."
        )

    if padding_anchor == "disabled":
        raise IdeasError(
            "`padding_anchor='disabled'` is not valid for direct padding "
            "operations. Use one of ['top-left', 'top-right', "
            "'bottom-left', 'bottom-right']."
        )
    if padding_anchor == "top-left":
        return 0, pad_w, 0, pad_h
    if padding_anchor == "top-right":
        return pad_w, 0, 0, pad_h
    if padding_anchor == "bottom-left":
        return 0, pad_w, pad_h, 0
    if padding_anchor == "bottom-right":
        return pad_w, 0, pad_h, 0
    raise IdeasError(
        f"`padding_anchor` {padding_anchor} not recognized. "
        "Expected one of ['top-left', 'top-right', 'bottom-left', "
        "'bottom-right']."
    )


def _pad_template_image_to_dims(template_img, target_dims, padding_anchor):
    """Pad a template image to target FoV dimensions (width, height)."""
    source_dims = (template_img.shape[0], template_img.shape[1])
    pad_left, pad_right, pad_top, pad_bottom = _compute_padding_by_anchor(
        source_dims=source_dims,
        target_dims=target_dims,
        padding_anchor=padding_anchor,
    )
    if all(x == 0 for x in [pad_left, pad_right, pad_top, pad_bottom]):
        return template_img
    return np.pad(
        template_img,
        ((pad_left, pad_right), (pad_top, pad_bottom)),
        mode="edge",
    )


def _pad_footprints_to_dims(
    footprints_mat, source_dims, target_dims, padding_anchor
):
    """Pad cell footprints to target FoV dimensions (width, height)."""
    source_w, source_h = source_dims
    target_w, target_h = target_dims
    pad_left, pad_right, pad_top, pad_bottom = _compute_padding_by_anchor(
        source_dims=(source_w, source_h),
        target_dims=(target_w, target_h),
        padding_anchor=padding_anchor,
    )
    if all(x == 0 for x in [pad_left, pad_right, pad_top, pad_bottom]):
        return footprints_mat

    n_px_expected = source_w * source_h
    if footprints_mat.shape[0] != n_px_expected:
        raise IdeasError(
            f"Footprints row count {footprints_mat.shape[0]} does not match "
            f"source FoV dimensions {source_dims} ({n_px_expected} pixels)."
        )

    n_cells = footprints_mat.shape[1]
    fp_coo = footprints_mat.tocoo(copy=False)
    old_y = fp_coo.row // source_w
    old_x = fp_coo.row % source_w
    new_y = old_y + pad_top
    new_x = old_x + pad_left
    new_row = new_y * target_w + new_x
    return csc_matrix(
        (fp_coo.data, (new_row, fp_coo.col)),
        shape=(target_h * target_w, n_cells),
    )


@beartype
def caiman_msr(
    *,
    cellset_paths: List[str],
    template_paths: List[str],
    eventset_paths: List[str] = [""],
    contains_series: bool = False,
    align_flag: bool = True,
    use_opt_flow: bool = False,
    max_shifts: int = 10,
    max_thr: Union[float, int] = 0.0,
    max_dist: Union[float, int] = 10.0,
    thresh_cost: Union[float, int] = 0.7,
    enclosed_thr: Optional[Union[float, int]] = None,
    use_cell_status: str = "accepted",
    min_n_regist_sess: int = 1,
    fp_thr_method: str = "nrg",
    fp_thr: Union[float, int] = 0.9,
    cmap_div: str = "bwr",
    cmap: str = "gray",
    show_grid: bool = True,
    ticks_step: int = 100,
    n_sample_cells: int = 20,
    padding_tolerance: int = 10,
    padding_anchor: str = "disabled",
    output_dir: str = "",
) -> None:
    """
    Perform CaImAn Multi-Session Registration on a set of cellsets and related template images.

    :param cellset_paths: Cellsets containing the footprints for MSR computation, as well as the traces for outputting registered cellsets. [.h5/.hdf5, .isxd]
    :param template_paths: Template images corresponding to the input cellsets. [.tif/.tiff, .isxd]. If FoV dimensions differ across sessions and `padding_anchor` is not "disabled", smaller sessions are edge-padded to the largest FoV.
    :param eventset_paths: (optional) Eventsets containing the neural events for outputting registered eventsets. If cellsets were provided as .h5/.hdf5, then they already contain Eventsets. [.isxd]
    :param contains_series: Indicate whether at least some of the input cellsets and optional eventsets are series. If True, cellsets will be matched to template images based on filename. [bool]
    :param align_flag: [from CaImAn docs for `align_flag`] "Align the templates before matching." [bool]
    :param use_opt_flow: [from CaImAn docs for `use_opt_flow`] "Use dense optical flow to align templates." [bool]
    :param max_shifts: [from CaImAn docs for `max_shifts`] "Max shifts in x and y." [int]
    :param max_thr: [from CaImAn docs for `max_thr`] "Max threshold parameter before binarization." For each footprint, pixels below `max_thr` * footprint maximum amplitude will be set to 0. [float]
    :param max_dist: [from CaImAn docs for `max_dist`] "Max distance between centroids." Any pair of footprints with centroids more distant than `max_dist` will be discarded from potential matches. [float]
    :param thresh_cost: [from CaImAn docs for `thresh_cost`] "Maximum distance considered." Cost is defined as 1 - Jaccard Index (degree of overlap). Higher values of `thresh_cost` will thus relax overlap requirement for footprints to be considered a match. [float]
    :param enclosed_thr: [from CaImAn docs for `enclosed_thr`] "If not None set distance to at most the specified value when ground truth is a subset of inferred." Effectively caps cost to 0.5 when one footprint is contained within another one. [None, float]
    :param use_cell_status: Cell status to use (either "accepted", "accepted & undecided", or "all").
    :param min_n_regist_sess: Minimum number of sessions an ROI must be registered across to be outputted in the registered cellsets (and optional eventsets). [int]
    :param fp_thr_method: Method for thresholding footprint contours: either "nrg" to keep the pixels that contribute up to a specified fraction of the footprint energy, or "max" to set to 0 pixels that have value less than a fraction of the maximum footprint value.
    :param fp_thr: Threshold for footprint contours, in terms of either footprint energy or maximum value, depending of the chosen Footprint Thresholding Method. Set to a value between 0.0 and 1.0, for rawer or smoother footprints, respectively. [float]
    :param cmap_div: Name of the Matplotlib diverging colormap to be used in the template image alignment preview figure. [str]
    :param cmap: Name of the Matplotlib colormap to be used in the template images in preview figures. [str]
    :param show_grid: Show grid on the template images of preview figures. [bool]
    :param ticks_step: Step for the x- and y-axis ticks. [int]
    :param n_sample_cells: Number of sample cells to be plotted in the registered traces preview figure. [int]
    :param padding_tolerance: Maximum allowed FoV padding in pixels along each axis when cross-session harmonization is enabled. If required padding exceeds this value, the tool raises an error. [int]
    :param padding_anchor: Corner used to anchor each session when padding FoV dimensions. Padding is added on the opposite sides to reach the largest FoV across sessions. Set to "disabled" to keep legacy behavior and fail if FoV dimensions differ across sessions. Must be one of "disabled", "top-left", "top-right", "bottom-left", or "bottom-right". [str]
    :param output_dir: [for testing purpose only] Directory where output files should be saved. [str]
    """
    logger.info("Starting CaImAn Multi-Session Registration")

    if not output_dir:
        output_dir = str(Path.cwd())
    if padding_tolerance < 0:
        raise IdeasError("`padding_tolerance` must be >= 0.")

    # validate inputs
    logger.info("Validating input files and parameters")
    if contains_series:
        # if series are provided, keep a copy of the original inputs
        cellset_paths_orig = cellset_paths.copy()
    (
        cellset_paths,
        eventset_paths,
        enclosed_thr,
        min_n_regist_sess,
        rec_to_sess_lookup,
    ) = utilities.validate_caiman_msr_inputs(
        cellset_paths=cellset_paths,
        template_paths=template_paths,
        eventset_paths=eventset_paths,
        contains_series=contains_series,
        enclosed_thr=enclosed_thr,
        use_cell_status=use_cell_status,
        min_n_regist_sess=min_n_regist_sess,
    )
    if contains_series:
        logger.info("Checking FoV consistency across all series recordings")
        template_dims_lookup = {}
        for idx_rec, cellset_path in enumerate(cellset_paths_orig):
            idx_sess = rec_to_sess_lookup[idx_rec]
            template_path = template_paths[idx_sess]

            if template_path not in template_dims_lookup:
                template_dims_lookup[template_path] = io.load_template_image(
                    template_path=template_path
                ).shape
            template_dims = template_dims_lookup[template_path]

            fp_dims = io.get_cellset_fov_dims(cellset_path=cellset_path)
            if fp_dims != template_dims:
                raise IdeasError(
                    "Cellset and template image FoV dimensions must match "
                    "within each session for all series recordings. "
                    f"Got cellset dims {fp_dims} and template dims "
                    f"{template_dims} for `{Path(cellset_path).name}` and "
                    f"`{Path(template_path).name}`."
                )

    # write input info to log
    n_sessions = len(cellset_paths)
    logger.info(
        f"CaImAn MSR will register cells across {n_sessions} sessions, "
        "based on the following files:"
    )
    logger.info(
        "- input cellsets: " f"{[Path(x).name for x in cellset_paths]}"
    )
    logger.info(
        "- input template images: " f"{[Path(x).name for x in template_paths]}"
    )
    if isinstance(eventset_paths[0], str):
        logger.info(
            "- input eventsets: " f"{[Path(x).name for x in eventset_paths]}"
        )

    # load footprints from cellsets and template images
    logger.info("Loading footprints and template images")
    spatial = []
    cell_names_list = []
    templates = []
    fov_dims = []
    for cellset_path, temp in zip(cellset_paths, template_paths):
        footprints_mat, cell_names, fp_dims = io.get_footprints_from_cellset(
            cellset_path=cellset_path,
            use_cell_status=use_cell_status,
        )
        template_img = io.load_template_image(template_path=temp)
        template_dims = template_img.shape
        if fp_dims != template_dims:
            raise IdeasError(
                "Cellset and template image FoV dimensions must match within "
                "each session. "
                f"Got cellset dims {fp_dims} and template dims {template_dims} "
                f"for `{Path(cellset_path).name}` and `{Path(temp).name}`."
            )

        spatial.append(footprints_mat)
        cell_names_list.append(cell_names)
        templates.append(template_img)
        fov_dims.append(template_dims)

    # harmonize FoV dimensions across sessions
    dims = (max(x[0] for x in fov_dims), max(x[1] for x in fov_dims))
    if any(x != dims for x in fov_dims):
        if padding_anchor == "disabled":
            raise IdeasError(
                "Input FoV dimensions differ across sessions and "
                "`padding_anchor=disabled`. Please use matching FoV dimensions "
                f"across sessions or set `padding_anchor` to one of "
                "['top-left', 'top-right', 'bottom-left', 'bottom-right'] "
                f"to enable padding. Here shapes were {fov_dims}."
            )
        pad_deltas = [(dims[0] - d[0], dims[1] - d[1]) for d in fov_dims]
        max_required_padding = max([max(x) for x in pad_deltas])
        if max_required_padding > padding_tolerance:
            raise IdeasError(
                "Input FoV dimensions differ across sessions beyond "
                f"`padding_tolerance={padding_tolerance}` pixels. Maximum "
                f"required padding is {max_required_padding} pixels. "
                f"Session shapes were {fov_dims}."
            )
        logger.warning(
            "Input FoV dimensions differ across sessions "
            f"({fov_dims}). Padding to common FoV {dims} using "
            f"`padding_anchor={padding_anchor}`."
        )
        templates = [
            _pad_template_image_to_dims(
                template_img=t,
                target_dims=dims,
                padding_anchor=padding_anchor,
            )
            for t in templates
        ]
        spatial = [
            _pad_footprints_to_dims(
                footprints_mat=s,
                source_dims=d_source,
                target_dims=dims,
                padding_anchor=padding_anchor,
            )
            for s, d_source in zip(spatial, fov_dims)
        ]

    # run CaImAn MSR
    logger.info("Running CaImAn alignment and registration algorithm")
    (
        spatial_union,
        assignments,
        matchings,
        _,
        D,
        D_cm,
        aligned_templates,
        xy_shifts,
    ) = register_multisession(
        A=spatial,
        dims=dims,
        templates=templates,
        align_flag=align_flag,
        use_opt_flow=use_opt_flow,
        max_shifts=max_shifts,
        max_thr=max_thr,
        max_dist=max_dist,
        thresh_cost=thresh_cost,
        enclosed_thr=enclosed_thr,
    )
    # D and D_cm, outputted by the function above and used as inputs in
    # several functions below, refer to the list of cost (1 - Jaccard Index)
    # and Euclidean distance matrices, respectively, as defined in CaImAn

    # save MSR outputs
    logger.info("Saving CaImAn MSR outputs")
    io.save_msr_output(
        spatial_union=spatial_union,
        assignments=assignments,
        matchings=matchings,
        dims=dims,
        output_dir=output_dir,
    )

    # transform and filter assignments matrix into a pandas DataFrame
    logger.info("Transforming assignments matrix")
    df_assignments, n_reg_cells_all = utilities.transform_assignments_matrix(
        assignments=assignments,
        min_n_regist_sess=min_n_regist_sess,
        cell_names_list=cell_names_list,
    )
    logger.info(
        f"Found {n_reg_cells_all} cells registered "
        f"across all {n_sessions} sessions!"
    )

    # save registered cellsets and eventsets
    logger.info("Saving registered cellsets and optional eventsets")
    if contains_series:
        # if series are provided, apply assignments to all provided inputs
        # cellsets using the recording-to-session lookup table
        cellset_paths = cellset_paths_orig.copy()
    if any(x != dims for x in fov_dims) and padding_anchor != "disabled":
        output_fov_dims = dims
        output_padding_anchor = padding_anchor
    else:
        output_fov_dims = None
        output_padding_anchor = "disabled"
    cs_out_paths, es_out_paths = io.save_registered_cellsets_eventsets(
        cellset_paths=cellset_paths,
        eventset_paths=eventset_paths,
        df_assignments=df_assignments,
        output_dir=output_dir,
        rec_to_sess_lookup=rec_to_sess_lookup,
        output_fov_dims=output_fov_dims,
        output_padding_anchor=output_padding_anchor,
    )

    # save output csv containing alignment metrics
    logger.info("Saving alignment metrics")
    io.save_alignment_metrics_csv(
        templates=templates,
        aligned_templates=aligned_templates,
        xy_shifts=xy_shifts,
        output_dir=output_dir,
    )

    # save output csv containing MSR metrics for all matches
    logger.info("Saving registration metrics")
    io.save_msr_metrics_csv(
        df_assignments=df_assignments,
        D=D,
        D_cm=D_cm,
        output_dir=output_dir,
    )

    # create metadata
    logger.info("Generating output metadata")
    metadata.create_msr_output_metadata(
        spatial=spatial,
        dims=dims,
        df_assignments=df_assignments,
        n_reg_cells_all=n_reg_cells_all,
        templates=templates,
        aligned_templates=aligned_templates,
        xy_shifts=xy_shifts,
        D=D,
        D_cm=D_cm,
        cs_out_paths=cs_out_paths,
        es_out_paths=es_out_paths,
        output_dir=output_dir,
    )

    # create preview figures
    logger.info("Generating preview figures")
    previews.create_CaImAn_MSR_preview_figures(
        spatial_union=spatial_union,
        spatial=spatial,
        dims=dims,
        templates=templates,
        aligned_templates=aligned_templates,
        xy_shifts=xy_shifts,
        assignments=assignments,
        df_assignments=df_assignments,
        D=D,
        D_cm=D_cm,
        max_dist=max_dist,
        cs_out_paths=cs_out_paths,
        es_out_paths=es_out_paths,
        fp_thr_method=fp_thr_method,
        fp_thr=fp_thr,
        cmap_div=cmap_div,
        cmap=cmap,
        show_grid=show_grid,
        ticks_step=ticks_step,
        n_sample_cells=n_sample_cells,
        output_dir=output_dir,
        rec_to_sess_lookup=rec_to_sess_lookup,
    )

    logger.info("CaImAn Multi-Session Registration successfully completed!")
