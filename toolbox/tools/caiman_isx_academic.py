from multiprocessing import set_start_method

set_start_method("spawn", force=True)

from typing import Optional, List
import numpy as np
import isx
import logging
import shutil
import os
import ast
import cv2
import psutil
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
from toolbox.utils.data_conversion import (
    convert_caiman_output_to_isxd,
    convert_memmap_data_to_output_files,
    write_cell_statuses,
    save_local_correlation_image,
)
from toolbox.utils.exceptions import IdeasError
from toolbox.utils.metadata import (
    generate_caiman_motion_correction_metadata,
    generate_caiman_spike_extraction_metadata,
)
from toolbox.utils.previews import (
    generate_caiman_motion_corrected_previews,
    generate_initialization_images_preview,
    generate_cell_set_previews,
    generate_event_set_preview,
    generate_local_correlation_image_preview,
)
from toolbox.utils.qc import generate_motion_correction_quality_assessment_data
from toolbox.utils.utilities import (
    movie_series,
    cell_set_series,
    get_file_size,
    copy_isxd_extra_properties,
    read_isxd_metadata,
)

logger = logging.getLogger()


def _run_caiman_workflow(
    *,
    # Input Files
    input_movie_files: List[str],
    parameters_file: Optional[List[str]] = None,
    # Settings
    overwrite_analysis_table_params: bool = False,
    save_img: bool = True,
    # Dataset
    fr: str = "auto",
    decay_time: float = 0.4,
    dxy: float = 1.0,
    # Motion Correction
    motion_correct: bool = True,
    pw_rigid: bool = False,
    gSig_filt: Optional[int] = None,
    max_shifts: int = 5,
    strides: int = 48,
    overlaps: int = 24,
    max_deviation_rigid: int = 3,
    border_nan: str = "copy",
    # Cell Extraction
    p: int = 1,
    K: str = "auto",
    gSig: int = 3,
    gSiz: int = 7,
    merge_thr: float = 0.7,
    rf: int = 40,
    stride: int = 20,
    tsub: int = 2,
    ssub: int = 1,
    method_init: str = "corr_pnr",
    min_corr: float = 0.85,
    min_pnr: float = 12,
    ssub_B: int = 2,
    ring_size_factor: float = 1.4,
    method_deconvolution: str = "oasis",
    update_background_components: bool = True,
    del_duplicates: bool = True,
    nb: int = 0,
    low_rank_background: bool = False,
    nb_patch: int = 0,
    rolling_sum: bool = True,
    only_init: bool = True,
    normalize_init: bool = False,
    center_psf: bool = True,
    bas_nonneg: bool = False,
    border_pix: int = 0,
    refit: bool = False,
    # Component Evaluation
    evaluate_components: bool = True,
    SNR_lowest: float = 0.5,
    min_SNR: float = 3,
    rval_lowest: float = -1,
    rval_thr: float = 0.85,
    use_cnn: bool = False,
    cnn_lowest: float = 0.1,
    gSig_range: str = None,
    min_cnn_thr: float = 0.9,
    # Patches
    n_processes: int = 7,
):
    """Run CaImAn cell extraction workflow.

     Steps:
        1. Motion correction using the NorMCorre algorithm
        2. Cell identification using the CNMF/CNMF-E algorithm
        3. Automated cell classification

    INPUT FILES
    :param input_movie_files: list of paths to the input movie files (isxd, tif, tiff, avi)
    :param parameters_file: path to the json parameters file

    SETTINGS
    :param overwrite_analysis_table_params: if True and a parameters file is provided, the analysis table columns
                                            will be overwritten by the values specified in the parameters file
    :param save_img: if True, local correlation image will be saved as a standalone .tif file

    DATASET
    :param fr: imaging rate in frames per second (If set to 'auto', the frame rate will be set based on file metadata if available. Otherwise, it will use CaImAn's default frame rate of 30)
    :param decay_time: length of typical transient in seconds
    :param dxy: spatial resolution of FOV in pixels per um

    MOTION CORRECTION
    :param motion_correct: If True, motion correction will be performed
    :param pw_rigid: If True, piecewise-rigid motion correction will be performed
    :param gSig_filt: size of kernel for high pass spatial filtering in 1p data.
                      If None no spatial filtering is performed
    :param max_shifts: maximum shifts during motion correction
    :param strides: how often to start a new patch in pw-rigid registration
    :param overlaps: overlap between patches in pixels in pw-rigid motion correction
    :param max_deviation_rigid: maximum deviation in pixels between rigid shifts and
                                shifts of individual patches
    :param border_nan: flag for allowing NaN in the boundaries. True allows NaN, whereas 'copy' copies
                       the value of the nearest data point

    CELL EXTRACTION
    :param p: order of AR indicator dynamics
    :param K: number of components to be found (per patch or whole FOV depending on whether rf=None) (If set to 'auto', it will be automatically estimated)
    :param gSig: radius of average neurons (in pixels)
    :param gSiz: half-size of bounding box for each neuron
    :param merge_thr: Trace correlation threshold for merging two components
    :param rf: Half-size of patch in pixels
    :param stride: Overlap between neighboring patches in pixels
    :param tsub: temporal downsampling factor
    :param ssub: spatial downsampling factor
    :param method_init: initialization method
    :param min_corr: minimum value of correlation image for determining a candidate component during corr_pnr
    :param min_pnr: minimum value of pnr image for determining a candidate component during corr_pnr
    :param ssub_B: downsampling factor for background during corr_pnr
    :param ring_size_factor: radius of ring (*gSig) for computing background during corr_pnr
    :param method_deconvolution: method for solving the constrained deconvolution problem
    :param update_background_components: whether to update the spatial background components
    :param del_duplicates: Delete duplicate components in the overlapping regions between neighboring patches. If False, then merging is used.
    :param nb: number of background components
    :param low_rank_background: Whether to update the background using a low rank approximation. If False all the nonzero elements of the background components are updated using hal (to be used with one background per patch)
    :param nb_patch: Number of (local) background components per patch
    :param rolling_sum: use rolling sum (as opposed to full sum) for determining candidate centroids during greedy_roi
    :param only_init: whether to run only the initialization
    :param normalize_init: whether to equalize the movies during initialization
    :param center_psf: whether to use 1p data processing mode. Set to true for 1p
    :param bas_nonneg: whether to set a non-negative baseline (otherwise b >= min(y))
    :param border_pix: Number of pixels to exclude around each border
    :param refit: if True, the initial estimates will be refined by re-running the CNMF algorithm seeded just on the spatial estimates from the previous step

    COMPONENT EVALUATION
    :param SNR_lowest: minimum required trace SNR. Traces with SNR below this will get rejected
    :param min_SNR: trace SNR threshold. Traces with SNR above this will get accepted
    :param rval_lowest: minimum required space correlation. Components with correlation below this will get rejected
    :param rval_thr: space correlation threshold. Components with correlation higher than this will get accepted
    :param use_cnn: flag for using the CNN classifier
    :param cnn_lowest: minimum required CNN threshold. Components with score lower than this will get rejected
    :param gSig_range: gSig scale values for CNN classifier. In not None, multiple values are tested in the CNN classifier.
    :param min_cnn_thr: CNN classifier threshold. Components with score higher than this will get accepted

    PATCHES
    :param n_processes: Number of processes used for processing patches in parallel
    """
    # set output directory
    output_dir = os.getcwd()

    # copy caiman cell classification cnn model
    caiman_cnn_model_path = "/ideas/caiman_data/model"
    if not os.path.exists(caiman_cnn_model_path):
        os.makedirs(caiman_cnn_model_path)
        for f in ["cnn_model.h5", "cnn_model.h5.pb", "cnn_model.json"]:
            shutil.copyfile(
                f"/ideas/{f}", os.path.join(caiman_cnn_model_path, f)
            )

    # update n_processes to match available resources
    cpu_count = psutil.cpu_count()
    new_n_processes = np.maximum(
        np.minimum(n_processes, int(cpu_count - 1)), 1
    )
    if n_processes != new_n_processes:
        logger.info(
            f"'n_processes' changed from {n_processes} to {new_n_processes} based on a CPU count of {cpu_count}"
        )
        n_processes = new_n_processes

    # adjust parameters that can be automatically estimated
    K = None if K in ["auto", None] else int(K)
    fr = 30 if fr in ["auto", None] else float(fr)

    # initialize parameters
    params_dict = {
        # input files
        "fnames": input_movie_files,
        # dataset
        "fr": fr,
        "decay_time": decay_time,
        "dxy": (dxy, dxy) if dxy is not None else dxy,
        # motion correction
        "motion_correct": motion_correct,
        "pw_rigid": pw_rigid,
        "gSig_filt": (
            (gSig_filt, gSig_filt) if gSig_filt is not None else gSig_filt
        ),
        "max_shifts": (
            (max_shifts, max_shifts) if max_shifts is not None else max_shifts
        ),
        "strides": (strides, strides) if strides is not None else strides,
        "overlaps": (overlaps, overlaps) if overlaps is not None else overlaps,
        "max_deviation_rigid": max_deviation_rigid,
        "border_nan": (
            border_nan
            if border_nan not in ["True", "False"]
            else ast.literal_eval(border_nan)
        ),
        # cell extraction
        "p": p,
        "K": K,
        "gSig": (gSig, gSig) if gSig is not None else gSig,
        "gSiz": (gSiz, gSiz) if gSiz is not None else gSiz,
        "merge_thr": merge_thr,
        "rf": rf,
        "stride": stride,
        "tsub": tsub,
        "ssub": ssub,
        "method_init": method_init,
        "min_corr": min_corr,
        "min_pnr": min_pnr,
        "ssub_B": ssub_B,
        "ring_size_factor": ring_size_factor,
        "method_deconvolution": method_deconvolution,
        "update_background_components": update_background_components,
        "del_duplicates": del_duplicates,
        "nb": nb,
        "low_rank_background": low_rank_background,
        "nb_patch": nb_patch,
        "rolling_sum": rolling_sum,
        "only_init": only_init,
        "normalize_init": normalize_init,
        "center_psf": center_psf,
        "bas_nonneg": bas_nonneg,
        "border_pix": border_pix,
        # component evaluation
        "SNR_lowest": SNR_lowest,
        "min_SNR": min_SNR,
        "rval_lowest": rval_lowest,
        "rval_thr": rval_thr,
        "use_cnn": use_cnn,
        "cnn_lowest": cnn_lowest,
        "gSig_range": (
            gSig_range
            if gSig_range in [None, ""]
            else [int(n) for n in gSig_range.split(",")]
        ),
        "min_cnn_thr": min_cnn_thr,
        # patches
        "n_processes": n_processes,
    }
    parameters = params.CNMFParams(params_dict=params_dict)

    # load parameters from file
    if parameters_file is not None and overwrite_analysis_table_params:
        if len(parameters_file) > 1:
            logger.warning(
                f"More than 1 parameters files were provided. "
                f"The first file '{os.path.basename(parameters_file[0])}' "
                f"will be used for processing."
            )

        logger.info(
            f"Loading parameters from input file '{os.path.basename(parameters_file[0])}'"
        )
        parameters.change_params_from_jsonfile(parameters_file[0])

        # adjust parameters that were overridden by parameters file
        if parameters.data.get("fr") != fr:
            fr = parameters.data.get("fr")
            logger.info(
                f"'fr' set to {fr} based on the selected parameters file"
            )

        if parameters.motion.get("pw_rigid") != pw_rigid:
            pw_rigid = parameters.motion.get("pw_rigid")
            logger.info(
                f"'pw_rigid' set to {pw_rigid} based on the selected parameters file"
            )

    # determine input data frame rate & determine original input order
    file_ext = os.path.splitext(input_movie_files[0])[1][1:].lower()
    original_input_movie_indices = list(range(len(input_movie_files)))
    if file_ext == "isxd":
        # validate input files form a valid series
        # and order them by their start time
        # (keep track of the original order of the input files since this is used to statically name output files)
        original_input_movie_files = input_movie_files
        input_movie_files = movie_series(input_movie_files)
        original_input_movie_indices = [
            input_movie_files.index(f) for f in original_input_movie_files
        ]
        logger.info(
            f"Sorted input movies chronologically - original: {original_input_movie_files}, sorted: {input_movie_files}"
        )

        mov = isx.Movie.read(input_movie_files[0])
        fr = 1e6 / mov.timing.period.to_usecs()
        parameters.change_params(
            params_dict={"fr": fr, "fnames": input_movie_files}
        )
        logger.info(f"'fr' updated to {fr} based on file metadata")
        del mov
    elif file_ext in ["avi", "mp4"]:
        cap = cv2.VideoCapture(input_movie_files[0])
        fr = cap.get(cv2.CAP_PROP_FPS)
        parameters.change_params(params_dict={"fr": fr})
        logger.info(f"'fr' updated to {fr} based on file metadata")
        del cap
    else:
        if parameters.data.get("fr") is None:
            default_fr = 30
            parameters.change_params(params_dict={"fr": default_fr})
            logger.info(
                f"'fr' not specified, defaulting to {default_fr} frames per second"
            )

    # set up computing cluster
    logger.info("Setting up computing cluster")
    _, cluster, n_processes = cm.cluster.setup_cluster(n_processes=n_processes)
    logger.info(f"Computing cluster set up (n_processes={n_processes})")

    # apply motion-correction to the movies
    if motion_correct:
        logger.info("Applying motion correction algorithm to the data")

        # perform motion correction
        mot_correct = MotionCorrect(
            input_movie_files,
            dview=cluster,
            **parameters.get_group("motion"),
        )
        mot_correct.motion_correct(save_movie=True)

        fname_mc = (
            mot_correct.fname_tot_els
            if parameters.motion.get("pw_rigid")
            else mot_correct.fname_tot_rig
        )
        if fname_mc == [None]:
            fname_mc = mot_correct.mmap_file

        if parameters.motion.get("pw_rigid"):
            bord_px = np.ceil(
                np.maximum(
                    np.max(np.abs(mot_correct.x_shifts_els)),
                    np.max(np.abs(mot_correct.y_shifts_els)),
                )
            ).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(
                int
            )

        bord_px = 0 if border_nan == "copy" else bord_px
        fname_new = cm.save_memmap(
            fname_mc,
            base_name="memmap_",
            order="C",
            border_to_0=bord_px,
            dview=cluster,
        )

        if bord_px != border_pix:
            parameters.change_params(params_dict={"border_pix": bord_px})
            logger.warning(
                f"'border_pix' updated to {bord_px} to account for motion correction"
            )

        logger.info(
            f"Motion corrected data written to memory-mapped file "
            f"({os.path.basename(fname_new)}, "
            f"size: {get_file_size(fname_new)})"
        )
    else:
        # if no motion correction just memory map the file
        bord_px = border_pix
        fname_new = cm.save_memmap(
            input_movie_files,
            base_name="memmap_",
            order="C",
            border_to_0=0,
            dview=cluster,
        )

        logger.info(
            f"Input data written to memory-mapped file "
            f"({os.path.basename(fname_new)}, "
            f"size: {get_file_size(fname_new)})"
        )

    # load memory-mapped data
    Yr, dims, num_frames = cm.load_memmap(fname_new)
    images = Yr.T.reshape((num_frames,) + dims, order="F")

    # apply CNMF to the input movie
    logger.info("Fitting CNMF model to the data")
    model = cnmf.CNMF(
        n_processes=n_processes, dview=cluster, params=parameters
    )
    model.fit(images)

    if refit:
        # refine initial estimates by re-running the CNMF algorithm seeded
        # just on the spatial estimates from the previous step
        model = model.refit(images, dview=cluster)

    # generate caiman preview, which includes the search images
    correlation_image = generate_initialization_images_preview(images=images)

    # perform automated component evaluation
    if evaluate_components:
        model.estimates.evaluate_components(
            images, model.params, dview=cluster
        )
        logger.info(
            f"Number of accepted cells: {len(model.estimates.idx_components)}"
        )
        logger.info(
            f"Number of rejected cells: {len(model.estimates.idx_components_bad)}"
        )

    logger.info(f"Total number of cells identified: {len(model.estimates.C)}")

    # add correlation image to output file
    if "correlation_image" in locals():
        model.estimates.Cn = correlation_image

    # include first isxd movie's metadata in output model
    if file_ext == "isxd":
        isxd_metadata = read_isxd_metadata(input_movie_files[0])
        model.movie_metadata = isxd_metadata

    # save CaImAn output
    caiman_output_filename = os.path.join(output_dir, "caiman_output.hdf5")
    model.save(caiman_output_filename)
    logger.info(
        f"CaImAn output saved "
        f"({os.path.basename(caiman_output_filename)}, "
        f"size: {get_file_size(caiman_output_filename)})"
    )

    # (optional) save local correlation image
    if save_img:
        image_output_filename = os.path.join(output_dir, "local_corr_img.tif")
        save_local_correlation_image(
            correlation_image=correlation_image,
            image_output_filename=image_output_filename,
        )
        generate_local_correlation_image_preview(
            correlation_image=correlation_image,
        )

    # ensure some cells were identified
    num_cells = len(model.estimates.C)
    if num_cells < 1:
        cm.stop_server(dview=cluster)
        raise IdeasError("No cells were identified")

    # convert CaImAn output to corresponding ISXD files
    convert_caiman_output_to_isxd(
        model=model,
        caiman_output_filename=caiman_output_filename,
        input_movie_files=input_movie_files,
        original_input_movie_indices=original_input_movie_indices,
    )

    logger.info("Stopping computing cluster")
    cm.stop_server(dview=cluster)


def caiman_workflow(
    *,
    # Input Files
    input_movie_files: List[str],
    parameters_file: Optional[List[str]] = None,
    # Settings
    overwrite_analysis_table_params: bool = False,
    save_img: bool = True,
    # Dataset
    fr: str = "auto",
    decay_time: float = 0.4,
    dxy: float = 1.0,
    # Motion Correction
    motion_correct: bool = True,
    pw_rigid: bool = False,
    gSig_filt: Optional[int] = None,
    max_shifts: int = 5,
    strides: int = 48,
    overlaps: int = 24,
    max_deviation_rigid: int = 3,
    border_nan: str = "copy",
    # Cell Extraction
    p: int = 1,
    K: str = "auto",
    gSig: int = 3,
    gSiz: int = 7,
    merge_thr: float = 0.7,
    rf: int = 40,
    stride: int = 20,
    tsub: int = 2,
    ssub: int = 1,
    method_init: str = "corr_pnr",
    min_corr: float = 0.85,
    min_pnr: float = 12,
    ssub_B: int = 2,
    ring_size_factor: float = 1.4,
    method_deconvolution: str = "oasis",
    update_background_components: bool = True,
    del_duplicates: bool = True,
    nb: int = 0,
    low_rank_background: bool = False,
    nb_patch: int = 0,
    rolling_sum: bool = True,
    only_init: bool = True,
    normalize_init: bool = False,
    center_psf: bool = True,
    bas_nonneg: bool = False,
    border_pix: int = 0,
    refit: bool = False,
    # Component Evaluation
    SNR_lowest: float = 0.5,
    min_SNR: float = 3,
    rval_lowest: float = -1,
    rval_thr: float = 0.85,
    use_cnn: bool = False,
    cnn_lowest: float = 0.1,
    gSig_range: str = None,
    min_cnn_thr: float = 0.9,
    # Patches
    n_processes: int = 7,
):
    """Run CaImAn cell extraction workflow.

     Steps:
        1. Motion correction using the NorMCorre algorithm
        2. Cell identification using the CNMF/CNMF-E algorithm
        3. Automated cell classification

    INPUT FILES
    :param input_movie_files: list of paths to the input movie files (isxd, tif, tiff, avi)
    :param parameters_file: path to the json parameters file

    SETTINGS
    :param overwrite_analysis_table_params: if True and a parameters file is provided, the analysis table columns
                                            will be overwritten by the values specified in the parameters file
    :param save_img: if True, local correlation image will be saved as a standalone .tif file

    DATASET
    :param fr: imaging rate in frames per second (If set to 'auto', the frame rate will be set based on file metadata if available. Otherwise, it will use CaImAn's default frame rate of 30)
    :param decay_time: length of typical transient in seconds
    :param dxy: spatial resolution of FOV in pixels per um

    MOTION CORRECTION
    :param motion_correct: If True, motion correction will be performed
    :param pw_rigid: If True, piecewise-rigid motion correction will be performed
    :param gSig_filt: size of kernel for high pass spatial filtering in 1p data.
                      If None no spatial filtering is performed
    :param max_shifts: maximum shifts during motion correction
    :param strides: how often to start a new patch in pw-rigid registration
    :param overlaps: overlap between patches in pixels in pw-rigid motion correction
    :param max_deviation_rigid: maximum deviation in pixels between rigid shifts and
                                shifts of individual patches
    :param border_nan: flag for allowing NaN in the boundaries. True allows NaN, whereas 'copy' copies
                       the value of the nearest data point

    CELL EXTRACTION
    :param p: order of AR indicator dynamics
    :param K: number of components to be found (per patch or whole FOV depending on whether rf=None) (If set to 'auto', it will be automatically estimated)
    :param gSig: radius of average neurons (in pixels)
    :param gSiz: half-size of bounding box for each neuron
    :param merge_thr: Trace correlation threshold for merging two components
    :param rf: Half-size of patch in pixels
    :param stride: Overlap between neighboring patches in pixels
    :param tsub: temporal downsampling factor
    :param ssub: spatial downsampling factor
    :param method_init: initialization method
    :param min_corr: minimum value of correlation image for determining a candidate component during corr_pnr
    :param min_pnr: minimum value of pnr image for determining a candidate component during corr_pnr
    :param ssub_B: downsampling factor for background during corr_pnr
    :param ring_size_factor: radius of ring (*gSig) for computing background during corr_pnr
    :param method_deconvolution: method for solving the constrained deconvolution problem
    :param update_background_components: whether to update the spatial background components
    :param del_duplicates: Delete duplicate components in the overlapping regions between neighboring patches. If False, then merging is used.
    :param nb: number of background components
    :param low_rank_background: Whether to update the background using a low rank approximation. If False all the nonzero elements of the background components are updated using hal (to be used with one background per patch)
    :param nb_patch: Number of (local) background components per patch
    :param rolling_sum: use rolling sum (as opposed to full sum) for determining candidate centroids during greedy_roi
    :param only_init: whether to run only the initialization
    :param normalize_init: whether to equalize the movies during initialization
    :param center_psf: whether to use 1p data processing mode. Set to true for 1p
    :param bas_nonneg: whether to set a non-negative baseline (otherwise b >= min(y))
    :param border_pix: Number of pixels to exclude around each border
    :param refit: if True, the initial estimates will be refined by re-running the CNMF algorithm seeded just on the spatial estimates from the previous step

    COMPONENT EVALUATION
    :param SNR_lowest: minimum required trace SNR. Traces with SNR below this will get rejected
    :param min_SNR: trace SNR threshold. Traces with SNR above this will get accepted
    :param rval_lowest: minimum required space correlation. Components with correlation below this will get rejected
    :param rval_thr: space correlation threshold. Components with correlation higher than this will get accepted
    :param use_cnn: flag for using the CNN classifier
    :param cnn_lowest: minimum required CNN threshold. Components with score lower than this will get rejected
    :param gSig_range: gSig scale values for CNN classifier. In not None, multiple values are tested in the CNN classifier.
    :param min_cnn_thr: CNN classifier threshold. Components with score higher than this will get accepted

    PATCHES
    :param n_processes: Number of processes used for processing patches in parallel
    """
    logger.info("CaImAn cell extraction workflow started")

    _run_caiman_workflow(
        # Input Data
        input_movie_files=input_movie_files,
        parameters_file=parameters_file,
        # Setting
        overwrite_analysis_table_params=overwrite_analysis_table_params,
        save_img=save_img,
        # Dataset
        fr=fr,
        decay_time=decay_time,
        dxy=dxy,
        # Motion Correction
        motion_correct=motion_correct,
        pw_rigid=pw_rigid,
        gSig_filt=gSig_filt,
        max_shifts=max_shifts,
        strides=strides,
        overlaps=overlaps,
        max_deviation_rigid=max_deviation_rigid,
        border_nan=border_nan,
        # Cell Extraction
        p=p,
        K=K,
        gSig=gSig,
        gSiz=gSiz,
        merge_thr=merge_thr,
        rf=rf,
        stride=stride,
        tsub=tsub,
        ssub=ssub,
        method_init=method_init,
        min_corr=min_corr,
        min_pnr=min_pnr,
        ssub_B=ssub_B,
        ring_size_factor=ring_size_factor,
        method_deconvolution=method_deconvolution,
        update_background_components=update_background_components,
        del_duplicates=del_duplicates,
        nb=nb,
        low_rank_background=low_rank_background,
        nb_patch=nb_patch,
        rolling_sum=rolling_sum,
        only_init=only_init,
        normalize_init=normalize_init,
        center_psf=center_psf,
        bas_nonneg=bas_nonneg,
        border_pix=border_pix,
        refit=refit,
        # Component Evaluation
        evaluate_components=True,
        SNR_lowest=SNR_lowest,
        min_SNR=min_SNR,
        rval_lowest=rval_lowest,
        rval_thr=rval_thr,
        use_cnn=use_cnn,
        cnn_lowest=cnn_lowest,
        gSig_range=gSig_range,
        min_cnn_thr=min_cnn_thr,
        # Patches
        n_processes=n_processes,
    )

    logger.info("CaImAn cell extraction workflow completed")


def motion_correction(
    *,
    # Input Files
    input_movie_files: List[str],
    parameters_file: Optional[List[str]] = None,
    overwrite_analysis_table_params: bool = False,
    # Dataset
    fr: str = "auto",
    # General
    min_mov: str = "auto",
    shifts_opencv: bool = True,
    nonneg_movie: bool = True,
    gSig_filt: Optional[int] = None,
    border_nan: str = "copy",
    num_frames_split: int = 80,
    is3D: bool = False,
    # Rigid
    max_shifts: int = 6,
    niter_rig: int = 1,
    splits_rig: int = 14,
    num_splits_to_process_rig: int = None,
    # Piecewise Rigid
    pw_rigid: bool = True,
    strides: int = 48,
    overlaps: int = 24,
    splits_els: int = 14,
    upsample_factor_grid: int = 4,
    max_deviation_rigid: int = 3,
    # Patches
    n_processes: int = 7,
    # Output Settings
    output_movie_format: str = "auto",
):
    """Apply CaImAn motion correction algorithm to the input movies.

    INPUT FILES
    :param input_movie_files: list of paths to the input movie files (isxd, tif, tiff, avi)
    :param parameters_file: path to the json parameters file

    SETTINGS
    :param overwrite_analysis_table_params: if True and a parameters file is provided, the analysis table columns
                                            will be overwritten by the values specified in the parameters file

    DATASET
    :param fr: imaging rate in frames per second (If set to 'auto', the frame rate will be set based on file metadata if available. Otherwise, it will use CaImAn's default frame rate of 30)

    GENERAL
    :param min_mov: estimated minimum value of the movie to produce an output that is positive
    :param shifts_opencv: flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    :param nonneg_movie: make the output movie and template mostly nonnegative by removing min_mov from movie
    :param gSig_filt: size of kernel for high pass spatial filtering in 1p data.
                      If None no spatial filtering is performed
    :param border_nan: flag for allowing NaN in the boundaries. True allows NaN, whereas 'copy' copies
                       the value of the nearest data point
    :param num_frames_split: number of frames in each batch
    :param is3D: flag for 3D motion correction
    :param indices: use that to apply motion correction only on a part of the FOV

    RIGID
    :param max_shifts: maximum deviation in pixels between rigid shifts and
                       shifts of individual patches
    :param niter_rig: maximum number of iterations rigid motion correction
    :param splits_rig: for parallelization split the movies in num_splits chunks across time
    :param num_splits_to_process_rig: if None all the splits are processed and the movie is saved,
                                      otherwise at each iteration num_splits_to_process_rig are considered

    PIECEWISE RIGID
    :param pw_rigid: If True, piecewise-rigid motion correction will be performed
    :param strides: how often to start a new patch in pw-rigid registration
    :param overlaps: overlap between patches in pixels in pw-rigid motion correction
    :param splits_els: for parallelization split the movies in  num_splits chunks across time
    :param upsample_factor_grid: upsample factor of shifts per patches to avoid smearing when merging patches
    :param max_deviation_rigid: maximum deviation allowed for patch with respect to rigid shifts

    PATCHES
    :param n_processes: Number of processes used for processing patches in parallel

    OUTPUT SETTINGS
    :param output_movie_format: file format to use for saving the motion-corrected movie
    """
    logger.info("CaImAn motion correction started")

    # set output directory
    output_dir = os.getcwd()

    # update n_processes to match available resources
    cpu_count = psutil.cpu_count()
    new_n_processes = np.maximum(
        np.minimum(n_processes, int(cpu_count - 1)), 1
    )
    if n_processes != new_n_processes:
        logger.info(
            f"'n_processes' changed from {n_processes} to {new_n_processes} based on a CPU count of {cpu_count}"
        )
        n_processes = new_n_processes

    # adjust parameters that can be automatically estimated
    fr = 30 if fr in ["auto", None] else float(fr)
    min_mov = None if min_mov in ["auto", None] else float(min_mov)

    # initialize parameters
    params_dict = {
        # input files
        "fnames": input_movie_files,
        # dataset
        "fr": fr,
        # general
        "motion_correct": True,
        "min_mov": min_mov,
        "shifts_opencv": shifts_opencv,
        "nonneg_movie": nonneg_movie,
        "gSig_filt": (
            (gSig_filt, gSig_filt) if gSig_filt is not None else gSig_filt
        ),
        "border_nan": (
            border_nan
            if border_nan not in ["True", "False"]
            else ast.literal_eval(border_nan)
        ),
        "num_frames_split": num_frames_split,
        "is3D": is3D,
        # rigid
        "max_shifts": (
            (max_shifts, max_shifts) if max_shifts is not None else max_shifts
        ),
        "niter_rig": niter_rig,
        "splits_rig": splits_rig,
        "num_splits_to_process_rig": num_splits_to_process_rig,
        # piecewise rigid
        "pw_rigid": pw_rigid,
        "strides": (strides, strides) if strides is not None else strides,
        "overlaps": (overlaps, overlaps) if overlaps is not None else overlaps,
        "splits_els": splits_els,
        "upsample_factor_grid": upsample_factor_grid,
        "max_deviation_rigid": max_deviation_rigid,
        # patches
        "n_processes": n_processes,
    }
    parameters = params.CNMFParams(params_dict=params_dict)

    # load parameters from file
    if parameters_file is not None and overwrite_analysis_table_params:
        if len(parameters_file) > 1:
            logger.warning(
                f"More than 1 parameters files were provided. "
                f"The first file '{os.path.basename(parameters_file[0])}' "
                f"will be used for processing."
            )

        logger.info(
            f"Loading parameters from input file '{os.path.basename(parameters_file[0])}'"
        )
        parameters.change_params_from_jsonfile(parameters_file[0])

        # adjust parameters that were overridden by parameters file
        if parameters.motion.get("pw_rigid") != pw_rigid:
            if parameters.data.get("fr") != fr:
                fr = parameters.data.get("fr")
                logger.info(
                    f"'fr' set to {fr} based on the selected parameters file"
                )

            pw_rigid = parameters.motion.get("pw_rigid")
            logger.info(
                f"'pw_rigid' set to {pw_rigid} based on the selected parameters file"
            )

    # override motion_correct param
    if parameters.motion.get("motion_correct") in [False, None]:
        parameters.change_params(params_dict={"motion_correct": True})
        logger.info(
            f"'motion_correct' set to 'True' to enable motion correction"
        )

    # determine input data frame rate & determine original input order
    file_ext = os.path.splitext(input_movie_files[0])[1][1:]
    original_input_movie_indices = list(range(len(input_movie_files)))
    if file_ext.lower() == "isxd":
        # validate input files form a valid series
        # and order them by their start time
        # (keep track of the original order of the input files since this is used to statically name output files)
        original_input_movie_files = input_movie_files
        input_movie_files = movie_series(input_movie_files)
        original_input_movie_indices = [
            input_movie_files.index(f) for f in original_input_movie_files
        ]
        logger.info(
            f"Sorted input movies chronologically - original: {original_input_movie_files}, sorted: {input_movie_files}"
        )

        mov = isx.Movie.read(input_movie_files[0])
        fr = 1e6 / mov.timing.period.to_usecs()
        parameters.change_params(
            params_dict={"fr": fr, "fnames": input_movie_files}
        )
        logger.info(f"'fr' updated to {fr} based on file metadata")
        del mov
    elif file_ext.lower() in ["avi", "mp4"]:
        cap = cv2.VideoCapture(input_movie_files[0])
        fr = cap.get(cv2.CAP_PROP_FPS)
        parameters.change_params(params_dict={"fr": fr})
        logger.info(f"'fr' updated to {fr} based on file metadata")
        del cap
    else:
        if parameters.data.get("fr") is None:
            default_fr = 30
            parameters.change_params(params_dict={"fr": default_fr})
            logger.info(
                f"'fr' not specified, defaulting to {default_fr} frames per second"
            )

    # set output movie format to match input movie format
    if output_movie_format == "auto":
        output_movie_format = file_ext

    # set up computing cluster
    logger.info("Setting up computing cluster")
    _, cluster, n_processes = cm.cluster.setup_cluster(n_processes=n_processes)
    logger.info(f"Computing cluster set up (n_processes={n_processes})")

    # perform motion correction
    logger.info("Applying motion correction algorithm to the data")
    mot_correct = MotionCorrect(
        input_movie_files,
        dview=cluster,
        **parameters.get_group("motion"),
    )
    mot_correct.motion_correct(save_movie=True)

    if hasattr(mot_correct, "fname_tot_els") and parameters.motion.get(
        "pw_rigid"
    ):
        fname_mc = mot_correct.fname_tot_els
    else:
        fname_mc = mot_correct.fname_tot_rig

    if fname_mc == [None]:
        fname_mc = mot_correct.mmap_file

    if parameters.motion.get("pw_rigid"):
        bord_px = np.ceil(
            np.maximum(
                np.max(np.abs(mot_correct.x_shifts_els)),
                np.max(np.abs(mot_correct.y_shifts_els)),
            )
        ).astype(int)
    else:
        bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)

    bord_px = 0 if border_nan == "copy" else bord_px
    fname_new = cm.save_memmap(
        fname_mc,
        base_name="memmap_",
        order="C",
        border_to_0=bord_px,
        dview=cluster,
    )

    logger.info(
        f"Motion corrected data written to memory-mapped file "
        f"({os.path.basename(fname_new)}, "
        f"size: {get_file_size(fname_new)})"
    )

    # convert motion-corrected data to corresponding output files
    (
        mc_movie_filenames,
        num_frames_per_movie,
        frame_index_cutoffs,
    ) = convert_memmap_data_to_output_files(
        memmap_filename=fname_new,
        input_movie_files=input_movie_files,
        original_input_movie_indices=original_input_movie_indices,
        frame_rate=parameters.data.get("fr"),
        output_movie_format=output_movie_format,
        output_dir=output_dir,
    )

    # generate CaImAn motion correction quality assessment data
    mc_qc_filename = os.path.join(output_dir, "mc_qc_data.csv")
    generate_motion_correction_quality_assessment_data(
        mc_obj=mot_correct,
        mc_qc_filename=mc_qc_filename,
        num_frames_per_movie=num_frames_per_movie,
    )

    # generate previews
    logger.info("Generating motion-corrected data previews")
    generate_caiman_motion_corrected_previews(
        mc_movie_filenames=mc_movie_filenames,
        mc_obj=mot_correct,
        original_input_indices=original_input_movie_indices,
        frame_index_cutoffs=frame_index_cutoffs,
        frame_rate=parameters.data.get("fr"),
    )

    # generate metadata
    logger.info("Generating motion-corrected metadata")
    generate_caiman_motion_correction_metadata(
        mc_movie_filenames=mc_movie_filenames,
        mc_obj=mot_correct,
        original_input_indices=original_input_movie_indices,
        input_movies_files=input_movie_files,
        sampling_rate=parameters.data.get("fr"),
    )

    logger.info("Stopping computing cluster")
    cm.stop_server(dview=cluster)
    logger.info("CaImAn motion correction completed")


def source_extraction(
    *,
    # Input Files
    input_movie_files: List[str],
    parameters_file: Optional[List[str]] = None,
    # Settings
    overwrite_analysis_table_params: bool = False,
    save_img: bool = True,
    # Dataset
    fr: str = "auto",
    decay_time: float = 0.4,
    dxy: float = 1.0,
    # Cell Extraction
    p: int = 1,
    K: str = "auto",
    gSig: int = 3,
    gSiz: int = 7,
    merge_thr: float = 0.7,
    rf: int = 40,
    stride: int = 20,
    tsub: int = 2,
    ssub: int = 1,
    method_init: str = "corr_pnr",
    min_corr: float = 0.85,
    min_pnr: float = 12,
    ssub_B: int = 2,
    ring_size_factor: float = 1.4,
    method_deconvolution: str = "oasis",
    update_background_components: bool = True,
    del_duplicates: bool = True,
    nb: int = 0,
    low_rank_background: bool = False,
    nb_patch: int = 0,
    rolling_sum: bool = True,
    only_init: bool = True,
    normalize_init: bool = False,
    center_psf: bool = True,
    bas_nonneg: bool = False,
    border_pix: int = 0,
    refit: bool = False,
    # Patches
    n_processes: int = 7,
):
    """Run CaImAn CNMF/CNMF-E source extraction.

    INPUT FILES
    :param input_movie_files: list of paths to the input movie files (isxd, tif, tiff, avi)
    :param parameters_file: path to the json parameters file

    SETTINGS
    :param overwrite_analysis_table_params: if True and a parameters file is provided, the analysis table columns
                                            will be overwritten by the values specified in the parameters file

    DATASET
    :param fr: imaging rate in frames per second (If set to 'auto', the frame rate will be set based on file metadata if available. Otherwise, it will use CaImAn's default frame rate of 30)
    :param decay_time: length of typical transient in seconds
    :param dxy: spatial resolution of FOV in pixels per um

    CELL EXTRACTION
    :param p: order of AR indicator dynamics
    :param K: number of components to be found (per patch or whole FOV depending on whether rf=None) (If set to 'auto', it will be automatically estimated)
    :param gSig: radius of average neurons (in pixels)
    :param gSiz: half-size of bounding box for each neuron
    :param merge_thr: Trace correlation threshold for merging two components
    :param rf: Half-size of patch in pixels
    :param stride: Overlap between neighboring patches in pixels
    :param tsub: temporal downsampling factor
    :param ssub: spatial downsampling factor
    :param method_init: initialization method
    :param min_corr: minimum value of correlation image for determining a candidate component during corr_pnr
    :param min_pnr: minimum value of pnr image for determining a candidate component during corr_pnr
    :param ssub_B: downsampling factor for background during corr_pnr
    :param ring_size_factor: radius of ring (*gSig) for computing background during corr_pnr
    :param method_deconvolution: method for solving the constrained deconvolution problem
    :param update_background_components: whether to update the spatial background components
    :param del_duplicates: Delete duplicate components in the overlapping regions between neighboring patches. If False, then merging is used.
    :param nb: number of background components
    :param low_rank_background: Whether to update the background using a low rank approximation. If False all the nonzero elements of the background components are updated using hal (to be used with one background per patch)
    :param nb_patch: Number of (local) background components per patch
    :param rolling_sum: use rolling sum (as opposed to full sum) for determining candidate centroids during greedy_roi
    :param only_init: whether to run only the initialization
    :param normalize_init: whether to equalize the movies during initialization
    :param center_psf: whether to use 1p data processing mode. Set to true for 1p
    :param bas_nonneg: whether to set a non-negative baseline (otherwise b >= min(y))
    :param border_pix: Number of pixels to exclude around each border
    :param refit: if True, the initial estimates will be refined by re-running the CNMF algorithm seeded just on the spatial estimates from the previous step

    PATCHES
    :param n_processes: Number of processes used for processing patches in parallel
    """
    logger.info("CaImAn source extraction started")

    _run_caiman_workflow(
        # Input Data
        input_movie_files=input_movie_files,
        parameters_file=parameters_file,
        # Setting
        overwrite_analysis_table_params=overwrite_analysis_table_params,
        save_img=save_img,
        # Dataset
        fr=fr,
        decay_time=decay_time,
        dxy=dxy,
        # Motion Correction
        motion_correct=False,
        # Cell Extraction
        p=p,
        K=K,
        gSig=gSig,
        gSiz=gSiz,
        merge_thr=merge_thr,
        rf=rf,
        stride=stride,
        tsub=tsub,
        ssub=ssub,
        method_init=method_init,
        min_corr=min_corr,
        min_pnr=min_pnr,
        ssub_B=ssub_B,
        ring_size_factor=ring_size_factor,
        method_deconvolution=method_deconvolution,
        update_background_components=update_background_components,
        del_duplicates=del_duplicates,
        nb=nb,
        low_rank_background=low_rank_background,
        nb_patch=nb_patch,
        rolling_sum=rolling_sum,
        only_init=only_init,
        normalize_init=normalize_init,
        center_psf=center_psf,
        bas_nonneg=bas_nonneg,
        border_pix=border_pix,
        refit=refit,
        # Component Evaluation
        evaluate_components=False,
        # Patches
        n_processes=n_processes,
    )

    logger.info("CaImAn source extraction completed")


def spike_extraction(
    *,
    # Input Files
    input_cellset_files: List[str],
    # Spike Extraction
    bl: str = "auto",
    c1: str = "auto",
    g: str = "auto",
    sn: str = "auto",
    p: int = 1,
    method_deconvolution: str = "oasis",
    bas_nonneg: bool = True,
    noise_method: str = "logmexp",
    noise_range: str = "0.25,0.5",
    s_min: float = None,
    optimize_g: bool = False,
    fudge_factor: float = 0.96,
    lags: int = 5,
    solvers: str = "ECOS,SCS",
):
    """Run CaImAn spike extraction algorithm.

    INPUT FILES
    :param input_cellset_files: list of paths to the input cellset files (isxd)

    SPIKE EXTRACTION
    :param bl: Fluorescence baseline value. If set to 'auto', it will be estimated from the data.
    :param c1: Value of calcium at time 0. If set to 'auto', it will be set based on the data.
    :param g: Parameters of the autoregressive process that models the fluorescence impulse response. If set to 'auto', it will be estimated from the data.
    :param sn: Standard deviation of the noise distribution. If set to 'auto', it will be estimated from the data.
    :param p: order of AR indicator dynamics
    :param method_deconvolution: Method for solving the constrained deconvolution of temporal traces
    :param bas_nonneg: If True, a non-negative baseline will be used. If False, the baseline will be greater than or equal to the minimum value, which could be negative.
    :param noise_method: Power spectrum averaging method used for noise estimation
    :param noise_range: Range of normalized frequencies over which to compute the power spectrum for noise estimation. The range should be specified as 'fmin,fmax', where fmin and fmax refer to the minimum and maximum of the normalized frequency range to use (e.g.: 0.25,0.5).
    :param s_min: Minimum spike threshold amplitude
    :param optimize_g: If True, the time constants will be optimized. This applies only to the 'oasis' deconvolution method.
    :param fudge_factor: Bias correction factor for the discrete time constants
    :param lags: Number of lags for estimating the time constants of the autoregressive model. This should be an integer between 1 and the number of timepoints in the data.
    :param solvers: Primary and secondary solvers to use with the cvxpy deconvolution method. This should be specified as 'solver1,solver2', where solver1 and solver1 refer to the primary and secondary solvers (e.g.: ECOS,SCS). The solvers should be one of the following values: 'ECOS', 'SCS', and 'CVXOPT'.
    """
    logger.info("CaImAn neural activity extraction started")

    # set output directory
    output_dir = os.getcwd()

    # sort input cell set chronologically
    original_input_cellset_files = input_cellset_files
    input_cellset_files = cell_set_series(input_cellset_files)
    original_input_cellset_indices = [
        input_cellset_files.index(f) for f in original_input_cellset_files
    ]
    logger.info(
        f"Sorted input cell sets chronologically - original: {original_input_cellset_files}, sorted: {input_cellset_files}"
    )

    logger.info(
        "Converting input parameters to match CaImAn's expected format"
    )
    # adjust parameters that may be passed in as lists
    if isinstance(noise_range, list):
        noise_range = ",".join(noise_range)
    if isinstance(solvers, list):
        solvers = ",".join(solvers)

    # adjust parameters that can be automatically estimated
    # and those that needs to be reformatted
    bl = None if bl in ["auto", None] else float(bl)
    c1 = None if c1 in ["auto", None] else float(c1)
    g = None if g in ["auto", None] else [float(n) for n in g.split(",")]
    sn = None if sn in ["auto", None] else float(sn)
    noise_range = [float(n) for n in noise_range.split(",")]

    # process data
    cellset_denoised_filenames = []
    eventset_filenames = []
    for output_file_index, i in enumerate(original_input_cellset_indices):
        # read input cell set and gather corresponding metadata
        input_cellset = isx.CellSet.read(input_cellset_files[i])
        cell_names = [
            input_cellset.get_cell_name(j)
            for j in range(input_cellset.num_cells)
        ]
        cell_statuses = [
            input_cellset.get_cell_status(j)
            for j in range(input_cellset.num_cells)
        ]
        time_offsets = np.array(
            [
                x.to_usecs()
                for x in input_cellset.timing.get_offsets_since_start()
            ],
            np.uint64,
        )

        # initialize denoised cell set
        cellset_denoised_filename = (
            f"cellset_denoised.{str(output_file_index).zfill(3)}.isxd"
        )
        cellset_denoised_filenames.append(cellset_denoised_filename)
        denoised_cellset = isx.CellSet.write(
            cellset_denoised_filename,
            input_cellset.timing,
            input_cellset.spacing,
        )

        # initialize event set
        eventset_filename = (
            f"neural_events.{str(output_file_index).zfill(3)}.isxd"
        )
        eventset_filenames.append(eventset_filename)
        neural_events = isx.EventSet.write(
            eventset_filename, input_cellset.timing, cell_names
        )

        # deconvolve calcium traces and extract neural spikes
        for cell_index in range(input_cellset.num_cells):
            # gather input cell set data
            cell_name = input_cellset.get_cell_name(cell_index)
            footprint = input_cellset.get_cell_image_data(cell_index)
            trace = input_cellset.get_cell_trace_data(cell_index)

            # apply constrained foopsi to input trace
            denoised_trace, _, _, _, _, spikes, _ = constrained_foopsi(
                fluor=trace,
                bl=bl,
                c1=c1,
                g=g,
                sn=sn,
                p=p,
                method_deconvolution=method_deconvolution,
                bas_nonneg=bas_nonneg,
                noise_range=noise_range,
                noise_method=noise_method,
                lags=lags,
                fudge_factor=fudge_factor,
                solvers=solvers,
                optimize_g=optimize_g,
                s_min=s_min,
            )

            # add denoised trace to output cell set
            denoised_cellset.set_cell_data(
                cell_index,
                footprint,
                denoised_trace.astype(np.float32),
                cell_name,
            )

            # add neural spikes to output event set
            events_trace = spikes.astype(np.float32)
            event_indices = np.argwhere(events_trace > 0)
            event_amplitudes = events_trace[event_indices].reshape(-1)
            event_offsets = time_offsets[event_indices]
            neural_events.set_cell_data(
                index=cell_index,
                offsets=event_offsets,
                amplitudes=event_amplitudes,
            )

        denoised_cellset.flush()
        neural_events.flush()

        # update cell statuses
        write_cell_statuses(
            cell_statuses=cell_statuses,
            cell_set_filenames=[cellset_denoised_filename],
        )

        # log files created
        logger.info(
            f"Denoised cell set saved "
            f"({os.path.basename(cellset_denoised_filename)}, "
            f"size: {get_file_size(cellset_denoised_filename)})"
        )
        logger.info(
            f"Neural events saved "
            f"({os.path.basename(eventset_filename)}, "
            f"size: {get_file_size(eventset_filename)})"
        )

        # generate data previews
        logger.info(
            f"Generating cell set preview for '{os.path.basename(cellset_denoised_filename)}'"
        )
        try:
            generate_cell_set_previews(
                cellset_filename=cellset_denoised_filename,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.warning(
                f"Cell set preview could not be generated for file {os.path.basename(cellset_denoised_filename)}"
            )

        logger.info(
            f"Generating event set preview for '{os.path.basename(eventset_filename)}'"
        )
        try:
            generate_event_set_preview(
                eventset_filename=eventset_filename, output_dir=output_dir
            )
        except Exception as e:
            logger.warning(
                f"Neural events preview could not be generated for file {os.path.basename(eventset_filename)}"
            )

    logger.info(
        "Copying extra properties from input isxd cell sets to output isxd files"
    )
    copy_isxd_extra_properties(
        input_isxd_files=input_cellset_files,
        original_input_indices=original_input_cellset_indices,
        outputs_isxd_files=[cellset_denoised_filenames, eventset_filenames],
    )

    # generate metadata
    logger.info("Generating spike extraction metadata")
    generate_caiman_spike_extraction_metadata(
        cellset_denoised_filenames=cellset_denoised_filenames,
        eventset_filenames=eventset_filenames,
        original_input_indices=original_input_cellset_indices,
        input_cellset_files=input_cellset_files,
    )

    logger.info("CaImAn neural activity extraction completed")
