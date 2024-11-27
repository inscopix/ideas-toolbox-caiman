from multiprocessing import set_start_method

set_start_method("spawn", force=True)

from typing import Optional, List
import numpy as np
import isx
import shutil
import os
import ast
import cv2
import psutil
from matplotlib.image import imsave
import caiman as cm
from caiman.source_extraction.cnmf import params as params
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from toolbox.utils.exceptions import IdeasError
from toolbox.utils.utilities import movie_series
from toolbox.utils.utilities import get_file_size
from toolbox.utils.data_conversion import convert_caiman_output_to_isxd
from toolbox.utils.previews import generate_initialization_images_preview

import logging

logger = logging.getLogger()


def caiman_workflow(
    *,
    # Input Files
    input_movie_files: List[str],
    parameters_file: Optional[List[str]] = None,
    overwrite_analysis_table_params: bool = False,
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

    # determine input data frame rate & determine original input order
    file_ext = os.path.splitext(input_movie_files[0])[1][1:]
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

        mov = isx.Movie.read(input_movie_files[0])
        fr = 1e6 / mov.timing.period.to_usecs()
        parameters.change_params(params_dict={"fr": fr})
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

        # perform rigid motion correction
        mot_correct = MotionCorrect(
            input_movie_files,
            dview=cluster,
            **parameters.get_group("motion"),
        )
        mot_correct.motion_correct(save_movie=True)

        fname_mc = (
            mot_correct.fname_tot_els
            if pw_rigid
            else mot_correct.fname_tot_rig
        )
        if fname_mc == [None]:
            fname_mc = mot_correct.mmap_file

        if pw_rigid:
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
    model.estimates.evaluate_components(images, model.params, dview=cluster)
    logger.info(f"Total number of cells identified: {len(model.estimates.C)}")
    logger.info(
        f"Number of accepted cells: {len(model.estimates.idx_components)}"
    )
    logger.info(
        f"Number of rejected cells: {len(model.estimates.idx_components_bad)}"
    )

    # add correlation image to output file
    if "correlation_image" in locals():
        model.estimates.Cn = correlation_image

    # save CaImAn output
    caiman_output_filename = os.path.join(output_dir, "caiman_output.hdf5")
    model.save(caiman_output_filename)
    logger.info(
        f"CaImAn output saved "
        f"({os.path.basename(caiman_output_filename)}, "
        f"size: {get_file_size(caiman_output_filename)})"
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

    cm.stop_server(dview=cluster)
    logger.info("CaImAn cell extraction workflow completed")
