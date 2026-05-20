import numpy as np
import pytest

from fast_hdbscan.boruvka_cl import (
    BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE,
    BAND_MODE_ROUND_MIN_RELATIVE,
    _compute_global_mrd_threshold,
    _make_empty_banding_metadata,
    parallel_boruvka_cl,
)
from fast_hdbscan.numba_kdtree import build_kdtree


def _empty_cl_csr_arrays(n_samples):
    return np.empty(0, dtype=np.int32), np.zeros(n_samples + 1, dtype=np.int32)


def test_parallel_boruvka_cl_no_cl_metadata_schema_is_normalized():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    tree = build_kdtree(X)
    cl_indices, cl_indptr = _empty_cl_csr_arrays(X.shape[0])

    _edges, _neighbors, _core_distances, metadata = parallel_boruvka_cl(
        tree=tree,
        n_threads=1,
        min_samples=1,
        cl_indices=cl_indices,
        cl_indptr=cl_indptr,
        sample_weights=None,
        band_fraction=np.inf,
        band_mode=BAND_MODE_ROUND_MIN_RELATIVE,
        return_banding_metadata=True,
    )

    expected = _make_empty_banding_metadata(
        band_mode=BAND_MODE_ROUND_MIN_RELATIVE
    )
    assert metadata == expected


def test_parallel_boruvka_cl_validates_band_mode_with_no_cl_constraints():
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    tree = build_kdtree(X)
    cl_indices, cl_indptr = _empty_cl_csr_arrays(X.shape[0])

    with pytest.raises(ValueError, match="Unknown band_mode"):
        parallel_boruvka_cl(
            tree=tree,
            n_threads=1,
            min_samples=1,
            cl_indices=cl_indices,
            cl_indptr=cl_indptr,
            band_fraction=np.inf,
            band_mode="not_a_mode",
        )


@pytest.mark.parametrize("band_fraction", [-0.01, np.nan])
def test_parallel_boruvka_cl_validates_band_fraction_with_no_cl_constraints(
    band_fraction,
):
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    tree = build_kdtree(X)
    cl_indices, cl_indptr = _empty_cl_csr_arrays(X.shape[0])

    with pytest.raises(ValueError, match="band_fraction"):
        parallel_boruvka_cl(
            tree=tree,
            n_threads=1,
            min_samples=1,
            cl_indices=cl_indices,
            cl_indptr=cl_indptr,
            band_fraction=band_fraction,
            band_mode=BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE,
        )


def test_quantile_schedule_band_fraction_one_matches_inf_threshold():
    neighbors = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    distances = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    core_distances = np.zeros(2, dtype=np.float32)

    threshold_one, metadata_one = _compute_global_mrd_threshold(
        band_fraction=1.0,
        band_mode=BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE,
        neighbors=neighbors,
        distances=distances,
        core_distances=core_distances,
        sample_weights=None,
    )
    threshold_inf, metadata_inf = _compute_global_mrd_threshold(
        band_fraction=np.inf,
        band_mode=BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE,
        neighbors=neighbors,
        distances=distances,
        core_distances=core_distances,
        sample_weights=None,
    )

    assert threshold_one.shape == (1,)
    assert threshold_inf.shape == (1,)
    assert np.isinf(threshold_one[0])
    assert np.isinf(threshold_inf[0])
    assert metadata_one["band_schedule_n_levels"] == 1
    assert metadata_inf["band_schedule_n_levels"] == 1
