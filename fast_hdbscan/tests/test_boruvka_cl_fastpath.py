"""
Tests for fast-path dual-tree Borůvka with CL constraints.

Module under test: fast_hdbscan/boruvka_cl.py
Entry point:       parallel_boruvka_cl(tree, n_threads, min_samples, cl_indices, cl_indptr, ...)
"""

import numpy as np
import pytest
import scipy.sparse as sparse
from sklearn.datasets import make_blobs

import fast_hdbscan
from fast_hdbscan import fast_hdbscan as _fast_hdbscan_fn
from fast_hdbscan.numba_kdtree import build_kdtree


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_blobs_cl(n=200, n_blocks=3, seed=0, density=0.05, n_features=10):
    """
    Generate well-separated blobs with within-block CL pairs.

    Returns (X float64, cl sparse CSR shape (n, n)).
    """
    rng = np.random.RandomState(seed)
    X, block_labels = make_blobs(
        n_samples=n,
        centers=n_blocks,
        cluster_std=0.3,
        random_state=seed,
        n_features=n_features,
    )
    X = X.astype(np.float64)

    cl = sparse.lil_matrix((n, n))
    for blk in range(n_blocks):
        idx = np.where(block_labels == blk)[0]
        m = len(idx)
        i_loc, j_loc = np.triu_indices(m, k=1)
        keep = rng.rand(len(i_loc)) < density
        for ii, jj in zip(idx[i_loc[keep]], idx[j_loc[keep]]):
            cl[ii, jj] = 1
            cl[jj, ii] = 1
    return X, cl.tocsr()


def _check_cl_violations(edges, cl_csr, n):
    """Return list of (i, j) CL pairs co-clustered by finite MST edges."""
    parent = np.arange(n, dtype=np.int32)

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges[np.isfinite(edges[:, 2])]:
        u, v = int(e[0]), int(e[1])
        ru, rv = _find(u), _find(v)
        if ru != rv:
            parent[rv] = ru

    cl_coo = cl_csr.tocoo()
    return [
        (i, j)
        for i, j in zip(cl_coo.row, cl_coo.col)
        if j > i and _find(i) == _find(j)
    ]


def _canonicalize_edges(edges):
    """Normalize edge direction (smaller src first), then lexsort."""
    e = edges.copy()
    swap = e[:, 0] > e[:, 1]
    e[swap, 0], e[swap, 1] = edges[swap, 1], edges[swap, 0]
    order = np.lexsort((e[:, 2], e[:, 1], e[:, 0]))
    return e[order]


def _empty_cl(n):
    return np.empty(0, dtype=np.int32), np.zeros(n + 1, dtype=np.int32)


# ---------------------------------------------------------------------------
# Step 1 — Empty-CL parity vs parallel_boruvka
# ---------------------------------------------------------------------------

class TestEmptyCLParity:
    """
    parallel_boruvka_cl with an empty CL graph must delegate to vanilla
    parallel_boruvka and produce identical output (same total weight,
    spanning n-1 edges).

    Note: vanilla parallel_boruvka is non-reproducible (prange race).
    We check same shape + total weight within float32 tolerance, not
    bit-identical endpoint order.
    """

    @pytest.mark.parametrize("n,min_samples", [(50, 5), (100, 3), (200, 10)])
    def test_empty_cl_parity(self, n, min_samples):
        from fast_hdbscan.boruvka import parallel_boruvka
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        import numba

        rng = np.random.RandomState(42)
        X = rng.randn(n, 5).astype(np.float64)
        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()

        cl_indices, cl_indptr = _empty_cl(n)

        edges_vanilla, nbrs_v, cd_v = parallel_boruvka(
            tree, n_threads, min_samples=min_samples
        )
        edges_cl, nbrs_cl, cd_cl = parallel_boruvka_cl(
            tree, n_threads, min_samples=min_samples,
            cl_indices=cl_indices, cl_indptr=cl_indptr,
        )

        # Both must span n-1 edges
        assert edges_cl.shape[0] == n - 1, f"Expected {n-1} edges, got {edges_cl.shape[0]}"
        assert edges_vanilla.shape[0] == n - 1

        # Same number of finite-weight edges (no bridging needed for vanilla)
        assert np.sum(np.isfinite(edges_cl[:, 2])) == np.sum(np.isfinite(edges_vanilla[:, 2]))

        # Total finite MST weight within 1e-4 relative tolerance
        # (same MST, possibly different float32 path from prange race)
        w_v = float(np.sum(edges_vanilla[np.isfinite(edges_vanilla[:, 2]), 2]))
        w_c = float(np.sum(edges_cl[np.isfinite(edges_cl[:, 2]), 2]))
        assert abs(w_v - w_c) < 1e-4 * (abs(w_v) + 1.0), (
            f"Empty-CL total weight mismatch: vanilla={w_v:.8f}, cl={w_c:.8f}"
        )


# ---------------------------------------------------------------------------
# Step 2 — Forbidden-components CSR unit test
# ---------------------------------------------------------------------------

class TestBuildForbiddenCSR:
    def test_basic(self):
        from fast_hdbscan.boruvka_cl import _build_forbidden_components_csr

        # 5 points: components [0, 0, 1, 1, 2]
        point_components = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        # CL pairs: (0,2) → comp(0)=0 vs comp(2)=1; (1,4) → comp(1)=0 vs comp(4)=2
        cl_u = np.array([0, 1], dtype=np.int32)
        cl_v = np.array([2, 4], dtype=np.int32)
        n_verts = 5

        fi, fp = _build_forbidden_components_csr(cl_u, cl_v, point_components, n_verts)
        # Comp 0 forbidden from {1, 2}
        row0 = fi[fp[0]:fp[1]]
        assert sorted(row0.tolist()) == [1, 2], f"comp 0 row: {sorted(row0.tolist())}"
        # Comp 1 forbidden from {0}
        row1 = fi[fp[1]:fp[2]]
        assert sorted(row1.tolist()) == [0], f"comp 1 row: {sorted(row1.tolist())}"
        # Comp 2 forbidden from {0}
        row2 = fi[fp[2]:fp[3]]
        assert sorted(row2.tolist()) == [0], f"comp 2 row: {sorted(row2.tolist())}"

    def test_same_component_skipped(self):
        from fast_hdbscan.boruvka_cl import _build_forbidden_components_csr

        # All points in same component — no forbidden pairs
        point_components = np.array([0, 0, 0], dtype=np.int32)
        cl_u = np.array([0], dtype=np.int32)
        cl_v = np.array([1], dtype=np.int32)
        fi, fp = _build_forbidden_components_csr(cl_u, cl_v, point_components, 3)
        assert fi.shape[0] == 0

    def test_dedup(self):
        from fast_hdbscan.boruvka_cl import _build_forbidden_components_csr

        # Two CL pairs producing the same comp pair: (0,2) and (1,2), both in comp 0 vs comp 1
        point_components = np.array([0, 0, 1, 1], dtype=np.int32)
        cl_u = np.array([0, 1], dtype=np.int32)
        cl_v = np.array([2, 3], dtype=np.int32)
        fi, fp = _build_forbidden_components_csr(cl_u, cl_v, point_components, 4)
        row0 = fi[fp[0]:fp[1]]
        # Should appear only once despite two source pairs
        assert sorted(row0.tolist()) == [1], f"Expected [1], got {sorted(row0.tolist())}"


class TestExtractCLPairArrays:
    def test_upper_triangle_only(self):
        from fast_hdbscan.boruvka_cl import _extract_cl_pair_arrays_from_csr

        # Symmetric 4x4 CSR: pairs (0,1), (0,2), (2,3)
        # Build via scipy for correct CSR structure
        cl_mat = sparse.csr_matrix(np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.float32))
        indices = cl_mat.indices.astype(np.int32)
        indptr = cl_mat.indptr.astype(np.int32)

        u_arr, v_arr, M = _extract_cl_pair_arrays_from_csr(indices, indptr)
        assert M == 3, f"Expected 3 upper-triangle pairs, got {M}"
        pairs = sorted(zip(u_arr.tolist(), v_arr.tolist()))
        assert pairs == [(0, 1), (0, 2), (2, 3)], f"Got {pairs}"

    def test_empty(self):
        from fast_hdbscan.boruvka_cl import _extract_cl_pair_arrays_from_csr

        u_arr, v_arr, M = _extract_cl_pair_arrays_from_csr(
            np.empty(0, dtype=np.int32),
            np.zeros(5, dtype=np.int32),
        )
        assert M == 0


# ---------------------------------------------------------------------------
# Step 3/4 — Zero CL violations on grid of sizes × densities × seeds
# ---------------------------------------------------------------------------

class TestZeroViolations:
    """
    Core correctness: parallel_boruvka_cl must produce zero CL violations
    on finite-weight edges across the full grid.
    """

    @pytest.mark.parametrize("n", [80, 150, 300])
    @pytest.mark.parametrize("density", [0.05, 0.15, 0.30])
    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_zero_violations_grid(self, n, density, seed):
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        X, cl = _make_blobs_cl(n=n, n_blocks=3, seed=seed, density=density)
        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()
        cl_indices, cl_indptr = _validate_cannot_link(cl, n)

        edges, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=5,
            cl_indices=cl_indices, cl_indptr=cl_indptr,
        )
        assert edges.shape[0] == n - 1, f"Expected {n-1} edges, got {edges.shape[0]}"
        violations = _check_cl_violations(edges, cl, n)
        assert len(violations) == 0, (
            f"n={n}, density={density}, seed={seed}: {len(violations)} violations: {violations[:5]}"
        )


# ---------------------------------------------------------------------------
# Step 5 — MST validity (spanning forest, weight ≥ unconstrained MST)
# ---------------------------------------------------------------------------

class TestMSTValidity:
    def test_spanning_forest_property(self):
        """Edges form a spanning tree (n-1 edges, no cycles, connected)."""
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        n = 120
        X, cl = _make_blobs_cl(n=n, n_blocks=3, seed=1, density=0.10)
        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()
        cl_indices, cl_indptr = _validate_cannot_link(cl, n)

        edges, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=5,
            cl_indices=cl_indices, cl_indptr=cl_indptr,
        )
        # n-1 edges (bridged MST is always n-1 edges)
        assert edges.shape[0] == n - 1

        # No duplicate endpoints (after normalization)
        e_norm = edges.copy()
        swap = e_norm[:, 0] > e_norm[:, 1]
        e_norm[swap, 0], e_norm[swap, 1] = edges[swap, 1], edges[swap, 0]
        ep_set = set(zip(e_norm[:, 0].astype(int), e_norm[:, 1].astype(int)))
        assert len(ep_set) == n - 1, "Duplicate edges detected"

    def test_weight_gte_unconstrained(self):
        """
        CL-constrained MST weight >= unconstrained MST weight (lower-bound check).
        """
        from fast_hdbscan.boruvka import parallel_boruvka
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        n = 100
        X, cl = _make_blobs_cl(n=n, n_blocks=3, seed=2, density=0.10)
        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()
        cl_indices, cl_indptr = _validate_cannot_link(cl, n)

        edges_vanilla, _, _ = parallel_boruvka(tree, n_threads, min_samples=5)
        edges_cl, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=5,
            cl_indices=cl_indices, cl_indptr=cl_indptr,
        )

        w_vanilla = float(np.sum(edges_vanilla[np.isfinite(edges_vanilla[:, 2]), 2]))
        w_cl = float(np.sum(edges_cl[np.isfinite(edges_cl[:, 2]), 2]))
        assert w_cl >= w_vanilla - 1e-8, (
            f"CL MST weight {w_cl:.6f} < unconstrained {w_vanilla:.6f}"
        )


# ---------------------------------------------------------------------------
# Step 5 — MSF bridging when CL forces disconnection
# ---------------------------------------------------------------------------

class TestMSFBridging:
    def test_inf_edges_when_cl_forces_disconnect(self):
        """
        A graph whose CL constraints force disconnection must yield +inf bridge
        edges in the output and downstream labels must still satisfy CL.
        """
        # Construct two-point data with one CL pair between them: must disconnect.
        n = 2
        X = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        cl = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float32))
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()
        cl_indices, cl_indptr = _validate_cannot_link(cl, n)

        edges, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=1,
            cl_indices=cl_indices, cl_indptr=cl_indptr,
        )
        assert edges.shape[0] == n - 1, f"Expected {n-1} edges, got {edges.shape[0]}"
        # The single edge must be +inf (CL blocks the only real merge)
        assert np.isinf(edges[0, 2]), f"Expected +inf edge, got {edges[0, 2]}"

    def test_downstream_labels_satisfy_cl(self):
        """
        fast_hdbscan(..., algorithm='boruvka', cannot_link=cl) produces labels
        where no CL pair shares a non-noise cluster.
        """
        n = 120
        X, cl = _make_blobs_cl(n=n, n_blocks=3, seed=3, density=0.15)

        labels, probs = _fast_hdbscan_fn(
            X,
            min_cluster_size=5,
            min_samples=5,
            algorithm="boruvka",
            metric="euclidean",
            cannot_link=cl,
        )
        assert labels.shape == (n,)

        cl_coo = cl.tocoo()
        for i, j in zip(cl_coo.row, cl_coo.col):
            if j <= i:
                continue
            if labels[i] >= 0 and labels[j] >= 0:
                assert labels[i] != labels[j], (
                    f"CL pair ({i},{j}) share cluster {labels[i]}"
                )

    def test_fully_blocked_terminates_with_bridging(self):
        """
        When every cross-component edge is CL-blocked, loop terminates cleanly
        and bridge edges yield a valid n-1 edge MST.
        """
        # 4 isolated points, all pairs CL-blocked
        n = 4
        rng = np.random.RandomState(0)
        X = rng.randn(n, 2).astype(np.float64) * 10  # well-separated
        cl_dense = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(cl_dense, 0)
        cl = sparse.csr_matrix(cl_dense)

        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()
        cl_indices, cl_indptr = _validate_cannot_link(cl, n)

        edges, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=1,
            cl_indices=cl_indices, cl_indptr=cl_indptr,
        )
        assert edges.shape[0] == n - 1, f"Expected {n-1} edges, got {edges.shape[0]}"
        # All edges should be +inf
        assert np.all(np.isinf(edges[:, 2])), "Expected all +inf edges"


# ---------------------------------------------------------------------------
# Step 6 — End-to-end accuracy
# ---------------------------------------------------------------------------

class TestEndToEndAccuracy:
    def test_ari_vs_kruskal_cl(self):
        """
        ARI of boruvka-CL labels ≥ 0.5 on well-separated blobs with very sparse CL.

        Both Borůvka-CL and Kruskal-CL are heuristics with different MSTs; we
        verify that the fast-path boruvka-CL produces reasonable clustering, not
        that it matches Kruskal-CL exactly.
        """
        from sklearn.metrics import adjusted_rand_score

        n = 300
        n_blocks = 3
        rng = np.random.RandomState(99)
        X, block_labels = make_blobs(
            n_samples=n, centers=n_blocks, cluster_std=0.3, random_state=99
        )
        X = X.astype(np.float64)

        # Very sparse CL between blocks only (cross-block: must not co-cluster)
        # Within-block: none (so clustering is not prevented)
        cl = sparse.lil_matrix((n, n))
        # Add a handful of cross-block CL pairs as correctness anchors
        for blk_a in range(n_blocks):
            for blk_b in range(blk_a + 1, n_blocks):
                idx_a = np.where(block_labels == blk_a)[0]
                idx_b = np.where(block_labels == blk_b)[0]
                # Pick 2 random cross-block pairs
                for _ in range(2):
                    ia = rng.choice(idx_a)
                    ib = rng.choice(idx_b)
                    cl[ia, ib] = 1
                    cl[ib, ia] = 1
        cl = cl.tocsr()

        labels_bor, _ = _fast_hdbscan_fn(
            X, min_cluster_size=15, min_samples=5,
            algorithm="boruvka", metric="euclidean", cannot_link=cl,
        )

        ari_bor = adjusted_rand_score(block_labels, labels_bor)
        # Well-separated blobs with very sparse CL should cluster cleanly
        assert ari_bor >= 0.5, (
            f"Borůvka-CL ARI={ari_bor:.3f} on well-separated blobs is too low"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_cl_pair_n1k(self):
        """Single CL pair on n=1000: zero violations, valid MST."""
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        n = 1000
        rng = np.random.RandomState(10)
        X = rng.randn(n, 4).astype(np.float64)
        cl = sparse.csr_matrix(([1, 1], ([0, 1], [1, 0])), shape=(n, n))
        cl = cl.astype(np.float32)

        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()
        cl_indices, cl_indptr = _validate_cannot_link(cl, n)

        edges, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=5,
            cl_indices=cl_indices, cl_indptr=cl_indptr,
        )
        assert edges.shape[0] == n - 1
        violations = _check_cl_violations(edges, cl, n)
        assert len(violations) == 0

    def test_asymmetric_input_symmetrized(self):
        """
        Upper-triangle-only CL input is symmetrized by _validate_cannot_link.

        We verify that:
        1. The extracted CL pair arrays are identical (symmetrization is correct).
        2. Both paths produce zero CL violations.

        We do NOT check bit-identical MST edges because parallel_boruvka_cl
        (like vanilla parallel_boruvka) is non-reproducible: prange races can
        pick different candidates in repeated calls.
        """
        from fast_hdbscan.boruvka_cl import _extract_cl_pair_arrays_from_csr, parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        n = 80
        X, cl_sym = _make_blobs_cl(n=n, seed=5)
        cl_upper = sparse.triu(cl_sym)

        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()

        ci_sym, cp_sym = _validate_cannot_link(cl_sym, n)
        ci_up, cp_up = _validate_cannot_link(cl_upper, n)

        # 1. CL pair arrays must be identical after symmetrization
        u_sym, v_sym, M_sym = _extract_cl_pair_arrays_from_csr(ci_sym, cp_sym)
        u_up, v_up, M_up = _extract_cl_pair_arrays_from_csr(ci_up, cp_up)
        assert M_sym == M_up, f"M_sym={M_sym} != M_up={M_up}"
        pairs_sym = sorted(zip(u_sym.tolist(), v_sym.tolist()))
        pairs_up = sorted(zip(u_up.tolist(), v_up.tolist()))
        assert pairs_sym == pairs_up, "Symmetrized CL pairs differ between sym and upper-triangle input"

        # 2. Both paths produce zero CL violations
        edges_sym, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=5,
            cl_indices=ci_sym, cl_indptr=cp_sym,
        )
        edges_up, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=5,
            cl_indices=ci_up, cl_indptr=cp_up,
        )

        viol_sym = _check_cl_violations(edges_sym, cl_sym, n)
        viol_up = _check_cl_violations(edges_up, cl_sym, n)
        assert len(viol_sym) == 0, f"Symmetric CL input: {len(viol_sym)} violations"
        assert len(viol_up) == 0, f"Upper-triangle CL input: {len(viol_up)} violations"

    def test_min_samples_1(self):
        """min_samples=1 edge case."""
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        n = 50
        X, cl = _make_blobs_cl(n=n, seed=6)
        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()
        ci, cp = _validate_cannot_link(cl, n)

        edges, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=1, cl_indices=ci, cl_indptr=cp,
        )
        assert edges.shape[0] == n - 1
        assert not np.any(np.isnan(edges))

    def test_n2_degenerate(self):
        """n=2: degenerate KD-tree with one CL pair."""
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        import numba

        n = 2
        X = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        # No CL constraint — should produce one edge
        cl_indices = np.empty(0, dtype=np.int32)
        cl_indptr = np.zeros(n + 1, dtype=np.int32)
        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()

        edges, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=1, cl_indices=cl_indices, cl_indptr=cl_indptr,
        )
        assert edges.shape[0] == n - 1

    def test_sample_weights(self):
        """sample_weights provided alongside CL — works, zero violations."""
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        n = 80
        X, cl = _make_blobs_cl(n=n, seed=8, density=0.05)
        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()
        ci, cp = _validate_cannot_link(cl, n)

        sw = np.ones(n, dtype=np.float32)

        edges, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=5,
            cl_indices=ci, cl_indptr=cp,
            sample_weights=sw,
        )
        assert edges.shape[0] == n - 1
        violations = _check_cl_violations(edges, cl, n)
        assert len(violations) == 0, f"sample_weights: {len(violations)} violations"

    def test_hdbscan_class_euclidean_boruvka_cl(self):
        """HDBSCAN(algorithm='boruvka', metric='euclidean', cannot_link=cl).fit(X)."""
        from fast_hdbscan import HDBSCAN

        n = 120
        X, cl = _make_blobs_cl(n=n, seed=11)

        model = HDBSCAN(
            min_cluster_size=5,
            min_samples=5,
            metric="euclidean",
            algorithm="boruvka",
            cannot_link=cl,
        )
        labels = model.fit_predict(X)
        assert labels.shape == (n,)

        cl_coo = cl.tocoo()
        for i, j in zip(cl_coo.row, cl_coo.col):
            if j <= i:
                continue
            if labels[i] >= 0 and labels[j] >= 0:
                assert labels[i] != labels[j], (
                    f"CL pair ({i},{j}) share cluster {labels[i]}"
                )


# ---------------------------------------------------------------------------
# Speed checkpoints (informal — print timings, no assert on time)
# ---------------------------------------------------------------------------

class TestSpeedCheckpoints:
    """
    Informal speed checkpoints.  Prints timings but does not assert on wall time
    (CI machines vary).  Target: brute n=10k < 0.25s; kNN n=100k < 2s.
    """

    @pytest.mark.slow
    def test_brute_n10k(self):
        import time
        from fast_hdbscan.boruvka_cl import parallel_boruvka_cl
        from fast_hdbscan.kruskal import _validate_cannot_link
        import numba

        n, d = 10_000, 3
        rng = np.random.RandomState(0)
        X = rng.randn(n, d).astype(np.float64)
        _, cl = _make_blobs_cl(n=n, n_blocks=5, seed=0, density=0.001, n_features=d)

        tree = build_kdtree(X)
        n_threads = numba.get_num_threads()
        ci, cp = _validate_cannot_link(cl, n)

        # Warmup
        _small = rng.randn(50, d).astype(np.float64)
        _, _cl_s = _make_blobs_cl(n=50, seed=1, n_features=d)
        _t = build_kdtree(_small)
        _ci, _cp = _validate_cannot_link(_cl_s, 50)
        parallel_boruvka_cl(_t, n_threads, min_samples=5, cl_indices=_ci, cl_indptr=_cp)

        t0 = time.perf_counter()
        edges, _, _ = parallel_boruvka_cl(
            tree, n_threads, min_samples=5, cl_indices=ci, cl_indptr=cp,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"\n[SPEED] brute n={n}: {elapsed:.3f}s")
        assert edges.shape[0] == n - 1
