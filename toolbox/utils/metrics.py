from beartype import beartype
from beartype.typing import List, Tuple, Union
from copy import deepcopy
from ideas.exceptions import IdeasError
import logging
from numpy import ndarray
import numpy as np
import pandas as pd

logger = logging.getLogger()


@beartype
def compute_alignment_metrics(
    *,
    templates: List[ndarray],
    aligned_templates: List[ndarray],
    xy_shifts: Union[List[List[Tuple[float, float]]], List[ndarray]],
) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    """
    Get correlation coefficent between templates pre- and post-alignment,
    as well as x/y shifts used to align templates/footprints across sessions.
    """
    r_pre_list = []
    r_post_list = []
    mean_shifts_list = []
    max_shifts_list = []
    for idx in range(len(templates) - 1):
        # below, T stands for template
        T2 = templates[idx + 1]
        T1 = templates[idx]

        T1a = aligned_templates[idx]
        idx_ok = np.where(~np.isnan(T1a))
        T2_ok = T2[idx_ok[0], idx_ok[1]]
        T1a_ok = T1a[idx_ok[0], idx_ok[1]]

        r_pre = np.corrcoef(T2.ravel(), T1.ravel())[0, 1]
        r_post = np.corrcoef(T2_ok.ravel(), T1a_ok.ravel())[0, 1]
        r_pre_list.append(r_pre)
        r_post_list.append(r_post)

        mean_shifts, max_shifts = process_xy_shifts(shifts=xy_shifts[idx])
        mean_shifts_list.append(mean_shifts)
        max_shifts_list.append(tuple([float(x) for x in max_shifts]))

        mean_shifts_disp = tuple(round(x, 2) for x in mean_shifts)
        max_shifts_disp = tuple(round(x, 2) for x in max_shifts)
        logger.info(
            "Alignment statistics for template images "
            f"#{idx + 1} and #{idx + 2}: "
            f"pre-alignment Pearson's r = {round(r_pre, 3)} - "
            f"post-alignment Pearson's r = {round(r_post, 3)} - "
            f"average x/y shifts = {mean_shifts_disp} - "
            f"maximum x/y shifts = {max_shifts_disp}"
        )

    r_pre = np.mean(r_pre_list)
    r_post = np.mean(r_post_list)
    mean_shifts = tuple(np.mean(mean_shifts_list, axis=0).astype(float))
    _, max_shifts = process_xy_shifts(shifts=max_shifts_list)
    max_shifts = tuple(max_shifts.astype(float))

    return r_pre, r_post, mean_shifts, max_shifts


@beartype
def process_xy_shifts(
    *,
    shifts: Union[List[Tuple[float, float]], ndarray],
) -> Tuple[ndarray, ndarray]:
    """
    Compute average x/y shifts from pixel- or patch-wise x/y shifts
    (for optical flow or piecewise rigid alignment approach, respectively).
    """
    if isinstance(shifts, list):
        shifts = np.vstack(shifts)
        mean_shifts = np.mean(shifts, axis=0)
        idx_max = np.argmax(np.abs(shifts), axis=0)
        max_shifts = np.array([shifts[x, y] for y, x in enumerate(idx_max)])
    elif len(shifts.shape) == 3:
        shifts = shifts[:, :, ::-1]
        shifts = shifts.reshape(shifts.shape[0] * shifts.shape[1], 2)
        mean_shifts = np.array([-np.mean(shifts[:, x]) for x in range(2)])
        idx_max = np.argmax(np.abs(shifts), axis=0)
        max_shifts = np.array([-shifts[x, y] for y, x in enumerate(idx_max)])
    else:
        raise IdeasError(
            f"x/y shifts can only be of type List (NoRMCorre) or ndarray "
            f"(optical flow), but is of type {type(shifts)} instead..."
        )
    return mean_shifts, max_shifts


@beartype
def process_distance_matrices(
    *,
    df_assignments: pd.DataFrame,
    D: List[List[ndarray]],
    D_cm: List[List[ndarray]],
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Isolate distance matrix values for registered and non-registered cells.
    """
    # D and D_cm refer to the list of cost (1 - Jaccard Index) and Euclidean
    # distance matrices, respectively, as defined in CaImAn
    D_nonreg = deepcopy(D)
    D_cm_nonreg = deepcopy(D_cm)

    d_reg_list = []
    d_cm_reg_list = []
    for idx in range(len(D)):
        df_assign_sub = (
            df_assignments.loc[:, [0, idx + 1]]
            .dropna()
            .astype(int)
            .reset_index(drop=True)
        )
        for _, (idx_y, idx_x) in df_assign_sub.iterrows():
            d_reg_list.append(1 - D[idx][0][idx_x, idx_y])
            d_cm_reg_list.append(D_cm[idx][0][idx_x, idx_y])
            D_nonreg[idx][0][idx_x, idx_y] = np.nan
            D_cm_nonreg[idx][0][idx_x, idx_y] = np.nan

    d_reg = np.array(d_reg_list)
    d_cm_reg = np.array(d_cm_reg_list)

    d_nonreg = np.concatenate([np.ravel(x) for x in D_nonreg])
    d_nonreg = 1 - d_nonreg[(~np.isnan(d_nonreg)) & (d_nonreg != 1)]

    d_cm_nonreg = np.concatenate([np.ravel(x) for x in D_cm_nonreg])
    d_cm_nonreg = d_cm_nonreg[~np.isnan(d_cm_nonreg)]

    return d_reg, d_cm_reg, d_nonreg, d_cm_nonreg


@beartype
def compute_distance_matrix_metrics(
    *,
    df_assignments: pd.DataFrame,
    D: List[List[ndarray]],
    D_cm: List[List[ndarray]],
) -> Tuple[float, float, float, float]:
    """
    Get mean overlap and mean centroid distance for all registered cells
    (matches) and all non-matches across all available pairs of sessions.
    """
    # D and D_cm refer to the list of cost (1 - Jaccard Index) and Euclidean
    # distance matrices, respectively, as defined in CaImAn
    d_match, d_cm_match, d_nonmatch, d_cm_nonmatch = process_distance_matrices(
        df_assignments=df_assignments,
        D=D,
        D_cm=D_cm,
    )

    mean_overlap = np.mean(d_match)
    mean_centroid_dist = np.mean(d_cm_match)
    mean_overlap_nonmatch = np.mean(d_nonmatch)
    mean_centroid_dist_nonmatch = np.mean(d_cm_nonmatch)

    return (
        mean_overlap,
        mean_centroid_dist,
        mean_overlap_nonmatch,
        mean_centroid_dist_nonmatch,
    )
