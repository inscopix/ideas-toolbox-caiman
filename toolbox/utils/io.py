from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import h5py
from ideas.exceptions import IdeasError
import isx
import logging
from numpy import ndarray
import numpy as np
import os
import pandas as pd
from scipy.sparse import csc_matrix
from tifffile import tifffile
from toolbox.utils import config
from toolbox.utils.metrics import process_xy_shifts

logger = logging.getLogger()


@beartype
def get_footprints_from_cellset(
    *,
    cellset_path: str,
    use_cell_status: str,
) -> Tuple[csc_matrix, List[str]]:
    """
    Read a cellset and output footprints (as a Scipy sparse matrix)
    and cell names.
    """
    use_cell_status_dict = {
        "accepted": ["accepted"],
        "accepted & undecided": ["accepted", "undecided"],
        "all": ["accepted", "undecided", "rejected"],
    }

    ext = os.path.splitext(cellset_path)[1]
    if ext in [".h5", ".hdf5"]:
        model = load_CNMF(cellset_path)
        footprints_mat = model.estimates.A
        n_px_fov, n_cells = footprints_mat.shape
        dims = (model.dims[1], model.dims[0])

        # transpose h5/hdf5 footprints to match template dimensions
        fp_mat_tmp = footprints_mat.toarray()
        fp_mat_tmp = fp_mat_tmp.reshape(dims[0], dims[1], n_cells)
        fp_mat_tmp = fp_mat_tmp.transpose(1, 0, 2)
        footprints_mat = csc_matrix(fp_mat_tmp.reshape(n_px_fov, n_cells))

        cell_statuses = np.array(["undecided"] * n_cells)
        if model.estimates.idx_components is not None:
            cell_statuses[model.estimates.idx_components] = "accepted"
        if model.estimates.idx_components_bad is not None:
            cell_statuses[model.estimates.idx_components_bad] = "rejected"

        idx_ok = np.sort(
            np.concatenate(
                [
                    np.where(cell_statuses == x)[0]
                    for x in use_cell_status_dict[use_cell_status]
                ]
            )
        )
        footprints_mat = footprints_mat[:, idx_ok]
        n_cells = footprints_mat.shape[1]

        assert n_cells > 0, (
            f"No cell found in {cellset_path} when `use_cell_status` set to "
            f"{use_cell_status}."
        )

        n_digits = len(str(idx_ok[-1]))
        cell_names = [f"C{idx:0{n_digits}g}" for idx in idx_ok]

    elif ext == ".isxd":
        cs = isx.CellSet.read(cellset_path)

        A = []
        cell_names = []
        for idx in range(cs.num_cells):
            if (
                cs.get_cell_status(idx)
                in use_cell_status_dict[use_cell_status]
            ):
                A.append(cs.get_cell_image_data(idx))
                cell_names.append(cs.get_cell_name(idx))
        cs.flush()

        assert len(A) > 0, (
            f"No cell found in {cellset_path} when `use_cell_status` set to "
            f"{use_cell_status}."
        )

        n_px_fov = np.multiply(*A[0].shape)
        footprints_mat = csc_matrix(
            np.hstack([x.reshape(n_px_fov, -1) for x in A]).astype(np.float64)
        )
        dims = (A[0].shape[1], A[0].shape[0])

    else:
        raise IdeasError(f"File format `{ext}` is not supported for cellsets.")

    logger.info(
        f"Loaded {len(cell_names)} footprints of dimensions {dims} pixels "
        f"from {os.path.basename(cellset_path)}"
    )
    return footprints_mat, cell_names


@beartype
def load_template_image(
    *,
    template_path: str,
) -> ndarray:
    """
    Load a template image.
    """
    ext = os.path.splitext(template_path)[1]
    if ext == ".tif":
        template_img = tifffile.imread(template_path).T
    elif ext == ".isxd":
        template_img = isx.Image.read(template_path).get_data().T
    else:
        raise IdeasError(
            f"File format `{ext}` is not supported for template images."
        )
    return template_img


@beartype
def save_msr_output(
    *,
    spatial_union: ndarray,
    assignments: ndarray,
    matchings: List[List[int]],
    dims: Tuple[int, int],
    output_dir: str,
) -> str:
    """
    Save CaImAn MSR output variables in a .h5 file.
    """
    msr_output_path = os.path.join(
        output_dir,
        config.MSR_OUTPUT_FNAME,
    )

    # make matchings compatible with h5 saving
    max_len = np.max([len(x) for x in matchings])
    matchings = [x + [np.nan] * (max_len - len(x)) for x in matchings]

    with h5py.File(msr_output_path, "w") as h5:
        h5.create_dataset(
            "spatial_union", data=spatial_union, dtype=np.float32
        )
        h5.create_dataset("assignments", data=assignments, dtype=np.float32)
        h5.create_dataset("matchings", data=matchings, dtype=np.float32)
        h5.create_dataset("dims", data=dims, dtype=np.float32)

    logger.info(f"Saved {config.MSR_OUTPUT_FNAME}")

    return msr_output_path


@beartype
def save_registered_cellsets_eventsets(
    cellset_paths: List[str],
    eventset_paths: List[Optional[str]],
    df_assignments: pd.DataFrame,
    output_dir: str,
) -> Tuple[List[str], List[Optional[str]]]:
    """
    Save cellsets and eventsets containing registered cells in a unified order.
    """
    cs_out_paths = []
    es_out_paths = []
    for idx_session, (cs_path, es_path) in enumerate(
        zip(cellset_paths, eventset_paths)
    ):
        cs_out_name = f"{config.CS_REG_PREFIX}.{idx_session:03g}.isxd"
        cs_out_path = os.path.join(output_dir, cs_out_name)
        es_out_name = f"{config.ES_REG_PREFIX}.{idx_session:03g}.isxd"
        es_out_path = os.path.join(output_dir, es_out_name)

        df_assign_sub = df_assignments.loc[
            :, ["name", idx_session, f"name_in_{idx_session}"]
        ].dropna()
        df_assign_sub[idx_session] = df_assign_sub[idx_session].astype(int)
        df_assign_sub = df_assign_sub.reset_index(drop=True)

        ext = os.path.splitext(cs_path)[1]
        if ext in [".h5", ".hdf5"]:
            model = load_CNMF(cs_path)
            dims = model.dims
            A = model.estimates.A.toarray()
            n_cells = A.shape[1]
            A = A.reshape(dims[1], dims[0], n_cells)
            A = A.transpose(1, 0, 2)
            A = A.reshape(dims[0] * dims[1], n_cells)
            rawC = model.estimates.C + model.estimates.YrA
            S = model.estimates.S
            cell_statuses = np.array(["undecided"] * n_cells)
            if model.estimates.idx_components is not None:
                cell_statuses[model.estimates.idx_components] = "accepted"
            if model.estimates.idx_components_bad is not None:
                cell_statuses[model.estimates.idx_components_bad] = "rejected"

            n_digits = len(df_assign_sub.iloc[-1, -1]) - 1
            all_cell_names = [f"C{idx:0{n_digits}g}" for idx in range(n_cells)]
            reg_cell_names = df_assign_sub["name"].tolist()
            idx_cells = [
                all_cell_names.index(x)
                for x in df_assign_sub[f"name_in_{idx_session}"].tolist()
            ]

            L = rawC.shape[1]
            fs = model.params.data["fr"]
            period_s = 1 / fs
            num, den = period_s.as_integer_ratio()
            period = isx.Duration._from_num_den(num, den)
            timing = isx.Timing(
                num_samples=L,
                period=period,
            )
            spacing = isx.Spacing(dims)
            cs_out = isx.CellSet.write(
                file_path=cs_out_path, timing=timing, spacing=spacing
            )
            for idx_out, (reg_cell_name, idx_cell) in enumerate(
                zip(reg_cell_names, idx_cells)
            ):
                cs_out.set_cell_data(
                    index=idx_out,
                    image=A[:, idx_cell],
                    trace=rawC[idx_cell, :],
                    name=reg_cell_name,
                )

            for idx_out, idx_cell in enumerate(idx_cells):
                cs_out.set_cell_status(
                    index=idx_out,
                    status=cell_statuses[idx_cell],
                )

            es_out = isx.EventSet.write(
                file_path=es_out_path,
                timing=timing,
                cell_names=df_assign_sub["name"].tolist(),
            )
            tb = np.linspace(0, (L - 1) / fs, L)
            for idx_out, idx_cell in enumerate(idx_cells):
                events = S[idx_cell, :]
                idx_evt = np.where(events > 0)[0]
                offsets = np.array(tb[idx_evt] * 1e6, dtype=np.uint64)
                amplitudes = events[idx_evt]
                es_out.set_cell_data(
                    index=idx_out, offsets=offsets, amplitudes=amplitudes
                )
            es_out.flush()

        elif ext == ".isxd":
            cs = isx.CellSet.read(file_path=cs_path)
            all_cell_names = [cs.get_cell_name(x) for x in range(cs.num_cells)]
            reg_cell_names = df_assign_sub["name"].tolist()
            idx_cells = [
                all_cell_names.index(x)
                for x in df_assign_sub[f"name_in_{idx_session}"].tolist()
            ]
            cs_out = isx.CellSet.write(
                file_path=cs_out_path, timing=cs.timing, spacing=cs.spacing
            )
            for idx_out, (reg_cell_name, idx_cell) in enumerate(
                zip(reg_cell_names, idx_cells)
            ):
                image = cs.get_cell_image_data(idx_cell)
                trace = cs.get_cell_trace_data(idx_cell)
                cs_out.set_cell_data(
                    index=idx_out, image=image, trace=trace, name=reg_cell_name
                )

            for idx_out, idx_cell in enumerate(idx_cells):
                status = cs.get_cell_status(idx_cell)
                cs_out.set_cell_status(index=idx_out, status=status)

            if es_path is not None:
                es = isx.EventSet.read(file_path=es_path)
                es_out = isx.EventSet.write(
                    file_path=es_out_path,
                    timing=es.timing,
                    cell_names=reg_cell_names,
                )
                for idx_out, idx_cell in enumerate(idx_cells):
                    offsets, amplitudes = es.get_cell_data(idx_cell)
                    es_out.set_cell_data(
                        index=idx_out, offsets=offsets, amplitudes=amplitudes
                    )
                es_out.flush()

        else:
            raise IdeasError(
                f"File format `{ext}` is not supported for cellsets"
                " and optional eventsets."
            )

        cs_out.flush()
        cs_out_paths.append(cs_out_path)
        logger.info(f"Saved registered Inscopix cellset {cs_out_name}")
        if ext in [".h5", ".hdf5"] or (ext == ".isxd" and es_path is not None):
            es_out_paths.append(es_out_path)
            logger.info(f"Saved registered Inscopix eventset {es_out_name}")
        else:
            es_out_paths.append(None)

    return cs_out_paths, es_out_paths


@beartype
def save_alignment_metrics_csv(
    *,
    templates: List[ndarray],
    aligned_templates: List[ndarray],
    xy_shifts: Union[List[List[Tuple[float, float]]], List[ndarray]],
    output_dir: str,
) -> None:
    """
    Save alignment metrics for all pairs of sessions in a .csv file.
    """
    align_metrics_csv_path = os.path.join(
        output_dir,
        config.ALIGN_METRICS_CSV_FNAME,
    )

    columns = [
        "first_session_idx",
        "second_session_idx",
        "pre_alignment_Pearson_r",
        "post_alignment_Pearson_r",
        "mean_x_shift",
        "mean_y_shift",
        "max_x_shift",
        "max_y_shift",
    ]
    df_align_metrics = pd.DataFrame(columns=columns)

    for idx, (T1, T1a, shifts) in enumerate(
        zip(templates, aligned_templates, xy_shifts)
    ):
        T2 = templates[idx + 1]
        idx_ok = np.where(~np.isnan(T1a))
        T2_ok = T2[idx_ok[0], idx_ok[1]]
        T1a_ok = T1a[idx_ok[0], idx_ok[1]]
        r_pre = np.corrcoef(T2.ravel(), T1.ravel())[0, 1]
        r_post = np.corrcoef(T2_ok.ravel(), T1a_ok.ravel())[0, 1]
        mean_shifts, max_shifts = process_xy_shifts(shifts=shifts)
        values = [
            idx,
            idx + 1,
            round(r_pre, 3),
            round(r_post, 3),
            round(mean_shifts[0], 2),
            round(mean_shifts[1], 2),
            round(max_shifts[0], 2),
            round(max_shifts[1], 2),
        ]
        df_align_metrics.loc[idx, :] = values

    df_align_metrics.to_csv(
        path_or_buf=align_metrics_csv_path,
        sep=",",
        header=True,
        index=True,
    )

    logger.info(f"Saved {config.ALIGN_METRICS_CSV_FNAME}")


@beartype
def save_msr_metrics_csv(
    *,
    df_assignments: pd.DataFrame,
    D: List[List[ndarray]],
    D_cm: List[List[ndarray]],
    output_dir: str,
) -> None:
    """
    Save MSR metrics for all matches in a .csv file.
    """
    msr_metrics_csv_path = os.path.join(
        output_dir,
        config.MSR_METRICS_CSV_FNAME,
    )

    columns = [
        "registered_cell_name",
        "first_session_idx",
        "cell_idx_in_first_session",
        "cell_name_in_first_session",
        "second_session_idx",
        "cell_idx_in_second_session",
        "cell_name_in_second_session",
        "centroid_distance_px",
        "overlap_Jaccard_index",
    ]
    df_msr_metrics = pd.DataFrame(columns=columns)

    for idx in range(len(D)):
        df_assign_sub = (
            df_assignments.loc[
                :, ["name", 0, idx + 1, "name_in_0", f"name_in_{idx + 1}"]
            ]
            .dropna()
            .reset_index(drop=True)
        )
        df_assign_sub[[0, idx + 1]] = df_assign_sub[[0, idx + 1]].astype(int)
        for _, (
            name,
            idx_y,
            idx_x,
            name_y,
            name_x,
        ) in df_assign_sub.iterrows():
            d_cm = D_cm[idx][0][idx_x, idx_y]
            d = 1 - D[idx][0][idx_x, idx_y]
            values = [name, 0, idx_y, name_y, idx + 1, idx_x, name_x, d_cm, d]
            df_msr_metrics.loc[len(df_msr_metrics)] = values

    df_msr_metrics = df_msr_metrics.sort_values(
        by=["registered_cell_name", "second_session_idx"]
    ).reset_index(drop=True)

    df_msr_metrics.to_csv(
        path_or_buf=msr_metrics_csv_path,
        sep=",",
        header=True,
        index=True,
    )

    logger.info(f"Saved {config.MSR_METRICS_CSV_FNAME}")
