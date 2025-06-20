from beartype import beartype
from beartype.typing import List, Optional, Union
from caiman.base.rois import register_multisession
import logging
import os
from toolbox.utils import io, metadata, previews, utilities

logger = logging.getLogger()


@beartype
def caiman_msr(
    *,
    cellset_paths: List[str],
    template_paths: List[str],
    eventset_paths: List[str] = [""],
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
    output_dir: str = "",
) -> None:
    """
    Perform CaImAn Multi-Session Registration on a set of cellsets and related template images.

    :param cellset_paths: Cellsets containing the footprints for MSR computation, as well as the traces for outputting registered cellsets. [.h5/.hdf5, .isxd]
    :param template_paths: Template images corresponding to the input cellsets. [.tif/.tiff, .isxd]
    :param eventset_paths: (optional) Eventsets containing the neural events for outputting registered eventsets. If Cellsets were provided as .h5/.hdf5, then they already contain Eventsets. [.isxd]
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
    :param output_dir: [for testing purpose only] Directory where output files should be saved. [str]
    """
    logger.info("Starting CaImAn Multi-Session Registration")

    if not output_dir:
        output_dir = os.getcwd()

    # validate inputs
    logger.info("Validating input files and parameters")
    eventset_paths, enclosed_thr, min_n_regist_sess = (
        utilities.validate_caiman_msr_inputs(
            cellset_paths=cellset_paths,
            template_paths=template_paths,
            eventset_paths=eventset_paths,
            enclosed_thr=enclosed_thr,
            use_cell_status=use_cell_status,
            min_n_regist_sess=min_n_regist_sess,
        )
    )

    # write input info to log
    n_sessions = len(cellset_paths)
    logger.info(
        f"CaImAn MSR will register cells across {n_sessions} sessions, "
        "based on the following files:"
    )
    logger.info(
        "- input cellsets: " f"{[os.path.basename(x) for x in cellset_paths]}"
    )
    logger.info(
        "- input template images: "
        f"{[os.path.basename(x) for x in template_paths]}"
    )
    if isinstance(eventset_paths[0], str):
        logger.info(
            "- input eventsets: "
            f"{[os.path.basename(x) for x in eventset_paths]}"
        )

    # load footprints from cellsets and template images
    logger.info("Loading footprints and template images")
    spatial = []
    cell_names_list = []
    templates = []
    for cellset_path, temp in zip(cellset_paths, template_paths):
        footprints_mat, cell_names = io.get_footprints_from_cellset(
            cellset_path=cellset_path,
            use_cell_status=use_cell_status,
        )
        spatial.append(footprints_mat)
        cell_names_list.append(cell_names)

        template_img = io.load_template_image(template_path=temp)
        templates.append(template_img)

    # get FoV dimensions
    dims = templates[0].shape
    assert all([x.shape == dims for x in templates]), (
        "All template images must have the same shape. "
        f"Here shapes were {[x.shape for x in templates]}."
    )

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
    cs_out_paths, es_out_paths = io.save_registered_cellsets_eventsets(
        cellset_paths=cellset_paths,
        eventset_paths=eventset_paths,
        df_assignments=df_assignments,
        output_dir=output_dir,
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
    )

    logger.info("CaImAn Multi-Session Registration successfully completed!")
