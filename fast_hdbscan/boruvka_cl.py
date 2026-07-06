"""Cannot-link constrained dual-tree Boruvka MST for Euclidean feature data.

Wires cannot-link pruning directly into the KD-tree dual-tree traversal so the
MST can be computed without materializing an O(nk) or O(n^2) CSR.

Key entry point: ``parallel_boruvka_cl``. Returns the same triple as
``parallel_boruvka`` (``edges, neighbors, core_distances``), and can optionally
return a fourth ``banding_metadata`` dictionary.
"""

import numba
import numpy as np

from numba import types
from numba.typed import Dict

from .boruvka import (
    parallel_boruvka,
    update_component_vectors,
    sample_weight_core_distance,
)
from .core_graph_cl import validate_and_prune_merges
from .disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
from .numba_kdtree import parallel_tree_query, rdist, point_to_node_lower_bound_rdist
from .variables import NUMBA_CACHE


# Inner "set" type for the component-constraint dictionary: a typed Dict used as
# a hash set (key = forbidden component root, value = dummy int8).  Membership
# tests, insertion, and deletion are all O(1).  This backs the ``mini_kruskal``
# merge that replaced the merge-then-break ``current`` strategy.
_CL_SET_TYPE = types.DictType(types.int64, types.int8)


BAND_MODE_ROUND_MIN_RELATIVE = "round_min_relative"
BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE = "global_mrd_quantile_schedule"
BAND_MODES = (
    BAND_MODE_ROUND_MIN_RELATIVE,
    BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE,
)


def _validate_banding_inputs(*, band_fraction, band_mode):
    """Validate ``band_fraction``/``band_mode`` and return float band_fraction."""
    if band_mode not in BAND_MODES:
        raise ValueError(f"Unknown band_mode={band_mode!r}; valid modes={BAND_MODES}.")

    try:
        band_fraction = float(band_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"band_fraction must be a real number; got {band_fraction!r}."
        ) from exc

    if np.isnan(band_fraction):
        raise ValueError("band_fraction must not be NaN.")
    if band_fraction < 0.0:
        raise ValueError(f"band_fraction must be non-negative; got {band_fraction}.")

    return band_fraction


def _extract_cl_pair_arrays_from_csr(cl_indices, cl_indptr):
    """
    Extract the upper-triangle CL pairs from a symmetric CSR.

    Parameters
    ----------
    cl_indices : int32[:], CSR column indices of the symmetric CL graph.
    cl_indptr  : int32[:], CSR row pointers (length n+1).

    Returns
    -------
    cl_pair_u : int32[:], shape (M,)
        Source vertex of each undirected pair (u < v).
    cl_pair_v : int32[:], shape (M,)
        Destination vertex of each undirected pair.
    M : int
        Number of undirected CL pairs.
    """
    n = len(cl_indptr) - 1
    # Keep one orientation for each undirected pair.
    pairs_u = []
    pairs_v = []
    for u in range(n):
        for p_idx in range(cl_indptr[u], cl_indptr[u + 1]):
            v = cl_indices[p_idx]
            if v > u:
                pairs_u.append(u)
                pairs_v.append(v)
    if len(pairs_u) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0
    return (
        np.array(pairs_u, dtype=np.int32),
        np.array(pairs_v, dtype=np.int32),
        len(pairs_u),
    )


@numba.njit(cache=NUMBA_CACHE)
def _build_forbidden_components_csr(
    cl_pair_u, cl_pair_v, point_components, n_verts
):
    """
    Build the per-round CSR of inter-component CL prohibitions.

    For each cannot-link pair (u, v), if they belong to distinct components
    (cu, cv), emit both directed component constraints into the CSR.

    Indexed by component id (= DSU root, range 0..n_verts-1), giving the
    sorted, deduplicated list of components that cannot merge with this one
    this round.

    Parameters
    ----------
    cl_pair_u      : int32[:], shape (M,)
    cl_pair_v      : int32[:], shape (M,)
    point_components : int32[:], shape (n,)
        Current DSU root per point.
    n_verts        : int

    Returns
    -------
    forbid_indices : int32[:]
        CSR column indices.
    forbid_indptr : int32[:]
        CSR row pointers, length (n_verts + 1).
    """
    M = cl_pair_u.shape[0]

    # Count directed pairs (cu, cv) with cu != cv.
    counts = np.zeros(n_verts, dtype=np.int32)
    active_cu = np.empty(2 * M, dtype=np.int32)
    active_cv = np.empty(2 * M, dtype=np.int32)
    n_active = np.int32(0)
    for k in range(M):
        cu = point_components[cl_pair_u[k]]
        cv = point_components[cl_pair_v[k]]
        if cu == cv:
            continue
        active_cu[n_active] = cu
        active_cv[n_active] = cv
        active_cu[n_active + 1] = cv
        active_cv[n_active + 1] = cu
        counts[cu] += np.int32(1)
        counts[cv] += np.int32(1)
        n_active += np.int32(2)

    # Build indptr from counts.
    forbid_indptr = np.zeros(n_verts + 1, dtype=np.int32)
    for i in range(n_verts):
        forbid_indptr[i + 1] = forbid_indptr[i] + counts[i]

    total = forbid_indptr[n_verts]
    if total == 0:
        return np.empty(0, dtype=np.int32), forbid_indptr

    # Fill raw column indices.
    raw_indices = np.empty(total, dtype=np.int32)
    fill_pos = np.zeros(n_verts, dtype=np.int32)
    for k in range(n_active):
        cu = active_cu[k]
        cv = active_cv[k]
        pos = forbid_indptr[cu] + fill_pos[cu]
        raw_indices[pos] = cv
        fill_pos[cu] += np.int32(1)

    # Sort each row and deduplicate in-place.
    forbid_indices_out = np.empty(total, dtype=np.int32)
    new_indptr = np.zeros(n_verts + 1, dtype=np.int32)
    write_ptr = np.int32(0)
    for cu in range(n_verts):
        start = forbid_indptr[cu]
        end = forbid_indptr[cu + 1]
        if start == end:
            new_indptr[cu + 1] = write_ptr
            continue
        row = raw_indices[start:end].copy()
        row.sort()
        prev = np.int32(-1)
        for idx in range(row.shape[0]):
            val = row[idx]
            if val != prev:
                forbid_indices_out[write_ptr] = val
                write_ptr += np.int32(1)
                prev = val
        new_indptr[cu + 1] = write_ptr

    return forbid_indices_out[:write_ptr], new_indptr


@numba.njit(inline="always", cache=NUMBA_CACHE)
def _in_sorted(forbid_indices, forbid_indptr, row, target):
    """
    Return True if ``target`` is in the sorted CSR row.
    """
    lo = forbid_indptr[row]
    hi = forbid_indptr[row + 1] - np.int32(1)
    while lo <= hi:
        mid = (lo + hi) >> 1
        val = forbid_indices[mid]
        if val == target:
            return True
        elif val < target:
            lo = mid + np.int32(1)
        else:
            hi = mid - np.int32(1)
    return False


@numba.njit(
    locals={
        "i": numba.types.int32,
        "idx": numba.types.int32,
        "left": numba.types.int32,
        "right": numba.types.int32,
        "d": numba.types.float32,
        "dist_lower_bound_left": numba.types.float32,
        "dist_lower_bound_right": numba.types.float32,
        "nc": numba.types.int32,
    },
    cache=NUMBA_CACHE,
    fastmath=True,
)
def _component_aware_query_recursion_cl(
    tree,
    node,
    point,
    heap_p,
    heap_i,
    current_core_distance,
    core_distances,
    current_component,
    node_components,
    point_components,
    dist_lower_bound,
    component_nearest_neighbor_dist,
    forbid_indices,
    forbid_indptr,
):
    """
    CL-aware dual-tree query recursion.

    Query one point against a KD-tree while excluding forbidden components.

    The recursion prunes mono-component subtrees when their component is
    forbidden for the query component, then filters remaining leaf candidates
    by current component constraints.
    """
    is_leaf = tree.is_leaf[node]
    idx_start = tree.idx_start[node]
    idx_end = tree.idx_end[node]

    # Case 1a: outside node radius.
    if dist_lower_bound > heap_p[0]:
        return

    # Case 1b: cannot beat best distance for this component.
    elif (
        dist_lower_bound > component_nearest_neighbor_dist[0]
        or current_core_distance > component_nearest_neighbor_dist[0]
    ):
        return

    # Case 1c: node contains only same component as query.
    elif node_components[node] == current_component:
        return

    else:
        nc = node_components[node]
        if nc != np.int32(-1) and nc != current_component:
            if _in_sorted(forbid_indices, forbid_indptr, current_component, nc):
                return

    # Case 2: leaf node.
    if is_leaf:
        for i in range(idx_start, idx_end):
            idx = tree.idx_array[i]
            pc = point_components[idx]
            if pc == current_component:
                continue
            if _in_sorted(forbid_indices, forbid_indptr, current_component, pc):
                continue
            if core_distances[idx] < component_nearest_neighbor_dist[0]:
                d = max(
                    rdist(point, tree.data[idx]),
                    current_core_distance,
                    core_distances[idx],
                )
                if d < heap_p[0]:
                    heap_p[0] = d
                    heap_i[0] = idx
                    if d < component_nearest_neighbor_dist[0]:
                        component_nearest_neighbor_dist[0] = d

    # Case 3: internal node.
    else:
        left = 2 * node + 1
        right = left + 1
        dist_lower_bound_left = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, left], tree.node_bounds[1, left], point
        )
        dist_lower_bound_right = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, right], tree.node_bounds[1, right], point
        )

        if dist_lower_bound_left <= dist_lower_bound_right:
            _component_aware_query_recursion_cl(
                tree,
                left,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_left,
                component_nearest_neighbor_dist,
                forbid_indices,
                forbid_indptr,
            )
            _component_aware_query_recursion_cl(
                tree,
                right,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_right,
                component_nearest_neighbor_dist,
                forbid_indices,
                forbid_indptr,
            )
        else:
            _component_aware_query_recursion_cl(
                tree,
                right,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_right,
                component_nearest_neighbor_dist,
                forbid_indices,
                forbid_indptr,
            )
            _component_aware_query_recursion_cl(
                tree,
                left,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_left,
                component_nearest_neighbor_dist,
                forbid_indices,
                forbid_indptr,
            )

    return


@numba.njit(
    locals={
        "i": numba.types.int32,
        "distance_lower_bound": numba.types.float32,
        "current_component": numba.types.int32,
    },
    parallel=True,
    cache=NUMBA_CACHE,
    fastmath=True,
)
def _boruvka_tree_query_cl(
    tree, node_components, point_components, core_distances,
    forbid_indices, forbid_indptr, distance_upper_bound,
):
    """
    CL-aware parallel dual-tree query.

    Returns ``(candidate_distances, candidate_indices)``, the same shape as
    ``boruvka_tree_query``.  Only candidates not violating CL constraints
    (as captured in ``forbid_indices/forbid_indptr``) are accepted.
    """
    n_pts = tree.data.shape[0]
    candidate_distances = np.full(n_pts, distance_upper_bound, dtype=np.float32)
    candidate_indices = np.full(n_pts, -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(
        n_pts, distance_upper_bound, dtype=np.float32
    )

    data = tree.data.astype(np.float32)

    for i in numba.prange(n_pts):
        current_component = point_components[i]
        distance_lower_bound = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i]
        )
        heap_p = candidate_distances[i : i + 1]
        heap_i = candidate_indices[i : i + 1]
        _component_aware_query_recursion_cl(
            tree,
            np.int32(0),
            data[i],
            heap_p,
            heap_i,
            core_distances[i],
            core_distances,
            current_component,
            node_components,
            point_components,
            distance_lower_bound,
            component_nearest_neighbor_dist[current_component : current_component + 1],
            forbid_indices,
            forbid_indptr,
        )

    return candidate_distances, candidate_indices


@numba.njit(cache=NUMBA_CACHE)
def _collect_knn_mrd_weights(knn_indices, knn_distances, core_distances):
    """
    Collect mutual-reachability weights from already-computed kNNs.

    The Euclidean Boruvka path stores internal weights as squared Euclidean
    distances (``rdist``) until the final ``sqrt`` conversion.  The collected
    weights intentionally stay in those internal units so thresholds can be
    used as query bounds without conversion.
    """
    max_observed = knn_indices.shape[0] * (knn_indices.shape[1] - 1)
    weights = np.empty(max_observed, dtype=np.float32)
    n_observed = np.int64(0)

    for i in range(knn_indices.shape[0]):
        for j in range(1, knn_indices.shape[1]):
            neighbor = knn_indices[i, j]
            if neighbor < 0 or neighbor == i:
                continue
            weight = knn_distances[i, j]
            if core_distances[i] > weight:
                weight = core_distances[i]
            if core_distances[neighbor] > weight:
                weight = core_distances[neighbor]
            if not np.isfinite(weight):
                continue
            weights[n_observed] = weight
            n_observed += np.int64(1)

    return weights[:n_observed]


def _make_empty_banding_metadata(*, band_mode):
    """Return standard band metadata fields for modes without a finite estimate."""
    return {
        "band_mode": band_mode,
        "band_range_low_internal": None,
        "band_range_high_internal": None,
        "band_threshold_internal": None,
        "band_range_n_observed": 0,
        "band_schedule_n_levels": None,
        "band_schedule_first_threshold_internal": None,
        "band_schedule_last_finite_threshold_internal": None,
        "band_schedule_levels_visited": None,
        "band_schedule_n_advances": None,
        "band_schedule_final_threshold_internal": None,
    }


def _compute_global_mrd_threshold(
    *,
    band_fraction,
    band_mode,
    neighbors,
    distances,
    core_distances,
    sample_weights,
):
    """
    Resolve banding semantics into a single internal-distance cutoff.

    ``round_min_relative`` preserves the legacy per-round behavior.
    ``global_mrd_quantile_schedule`` returns an increasing threshold schedule
    and finishes with an unbounded level. In global scheduled mode,
    ``band >= 1.0`` and ``band = inf`` are both explicitly unbounded.
    """
    band_fraction = _validate_banding_inputs(
        band_fraction=band_fraction, band_mode=band_mode
    )

    if band_mode == BAND_MODE_ROUND_MIN_RELATIVE:
        return np.inf, _make_empty_banding_metadata(band_mode=band_mode)

    if band_mode == BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE:
        thresholds, metadata = _compute_global_mrd_quantile_schedule(
            band_fraction=band_fraction,
            neighbors=neighbors,
            distances=distances,
            core_distances=core_distances,
            sample_weights=sample_weights,
        )
        return thresholds, metadata

    raise ValueError(f"Unhandled band_mode={band_mode!r}.")


def _compute_global_mrd_quantile_schedule(
    *,
    band_fraction,
    neighbors,
    distances,
    core_distances,
    sample_weights,
):
    """
    Build an increasing absolute-threshold schedule from observed MRD quantiles.

    A finite ``band_fraction`` is interpreted as the fraction of the observed
    kNN/core-distance weight distribution processed per level.  For example,
    ``band_fraction=0.1`` uses roughly 10%, 20%, ..., 100% quantile thresholds,
    then a final unbounded level.  This keeps the band global while preventing
    finite bands from being final hard caps.
    """
    band_fraction = float(band_fraction)
    if not np.isfinite(band_fraction) or band_fraction >= 1.0:
        metadata = _make_empty_banding_metadata(
            band_mode=BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE
        )
        metadata["band_schedule_n_levels"] = 1
        metadata["band_schedule_first_threshold_internal"] = None
        metadata["band_schedule_last_finite_threshold_internal"] = None
        metadata["band_schedule_final_threshold_internal"] = float("inf")
        return np.asarray([np.inf], dtype=np.float64), metadata
    if band_fraction <= 0.0:
        raise ValueError(
            f"{BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE!r} requires "
            f"0 < band_fraction < 1; got {band_fraction}."
        )
    if sample_weights is not None:
        raise ValueError(
            f"{BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE!r} is currently implemented "
            "only for unweighted Euclidean Boruvka-CL runs."
        )

    weights = _collect_knn_mrd_weights(neighbors, distances, core_distances)
    if weights.shape[0] == 0:
        metadata = _make_empty_banding_metadata(
            band_mode=BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE
        )
        metadata["band_schedule_n_levels"] = 1
        metadata["band_schedule_final_threshold_internal"] = float("inf")
        return np.asarray([np.inf], dtype=np.float64), metadata

    probabilities = np.arange(
        band_fraction,
        1.0 + 0.5 * band_fraction,
        band_fraction,
        dtype=np.float64,
    )
    probabilities = np.clip(probabilities, 0.0, 1.0)
    if probabilities[-1] < 1.0:
        probabilities = np.concatenate((probabilities, np.asarray([1.0])))

    thresholds_finite = np.quantile(weights.astype(np.float64), probabilities)
    thresholds_finite = np.unique(thresholds_finite.astype(np.float64))
    thresholds = np.concatenate((thresholds_finite, np.asarray([np.inf])))

    weight_low = float(np.min(weights))
    weight_high = float(np.max(weights))
    first_threshold = float(thresholds_finite[0])
    last_finite_threshold = float(thresholds_finite[-1])
    metadata = {
        "band_mode": BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE,
        "band_range_low_internal": weight_low,
        "band_range_high_internal": weight_high,
        "band_threshold_internal": first_threshold,
        "band_range_n_observed": int(weights.shape[0]),
        "band_schedule_n_levels": int(thresholds.shape[0]),
        "band_schedule_first_threshold_internal": first_threshold,
        "band_schedule_last_finite_threshold_internal": last_finite_threshold,
        "band_schedule_levels_visited": None,
        "band_schedule_n_advances": None,
        "band_schedule_final_threshold_internal": None,
    }
    return thresholds, metadata


def _next_query_upper_bound(threshold):
    """Return a float32 KD-tree query bound that includes threshold-equal edges."""
    if not np.isfinite(threshold):
        return np.float32(np.inf)
    return np.nextafter(np.float32(threshold), np.float32(np.inf))


def _advance_global_band_threshold(
    *,
    thresholds,
    level_index,
):
    """Move to the next global threshold level if one exists."""
    if level_index + 1 >= thresholds.shape[0]:
        return level_index, False
    return level_index + 1, True


def _advance_global_band_state(*, thresholds, level_index, n_advances):
    """Advance a scheduled global threshold and return the updated state."""
    next_level_index, advanced = _advance_global_band_threshold(
        thresholds=thresholds,
        level_index=level_index,
    )
    if not advanced:
        return level_index, n_advances, float(thresholds[level_index]), False
    return (
        next_level_index,
        n_advances + 1,
        float(thresholds[next_level_index]),
        True,
    )


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _collect_knn_candidates_cl(knn_indices, knn_distances, core_distances):
    """
    Collect per-point best outgoing kNN candidates without touching a DSU.

    For each point i, returns the first neighbor k (in knn order) where
    core_distances[i] >= core_distances[k] using the same selection rule as
    ``initialize_boruvka_from_knn``.

    Returns ``(cand_src, cand_dst, cand_wt, n_cand)`` using raw vertex ids
    (not compressed components); all components are initially distinct.
    """
    n = knn_indices.shape[0]
    raw_src = np.full(n, np.int32(-1), dtype=np.int32)
    raw_dst = np.full(n, np.int32(-1), dtype=np.int32)
    raw_wt = np.full(n, np.inf, dtype=np.float32)

    for i in numba.prange(n):
        for j in range(1, knn_indices.shape[1]):
            k = np.int32(knn_indices[i, j])
            if core_distances[i] >= core_distances[k]:
                raw_src[i] = np.int32(i)
                raw_dst[i] = k
                raw_wt[i] = np.float32(max(core_distances[i], knn_distances[i, j]))
                break

    # Compact valid candidates.
    n_cand = np.int32(0)
    for i in range(n):
        if raw_src[i] >= 0:
            n_cand += np.int32(1)

    out_src = np.empty(n_cand, dtype=np.int32)
    out_dst = np.empty(n_cand, dtype=np.int32)
    out_wt = np.empty(n_cand, dtype=np.float32)
    j = np.int32(0)
    for i in range(n):
        if raw_src[i] >= 0:
            out_src[j] = raw_src[i]
            out_dst[j] = raw_dst[i]
            out_wt[j] = raw_wt[i]
            j += np.int32(1)
    return out_src, out_dst, out_wt, n_cand


@numba.njit(
    locals={
        "i": numba.types.int32,
        "from_comp": numba.types.int32,
        "to_comp": numba.types.int32,
    },
    cache=NUMBA_CACHE,
)
def _select_components_cl(
    candidate_distances,
    candidate_indices,
    point_components,
    n_verts,
):
    """
    Reduce per-vertex candidates to per-component best (src, dst, wt).

    Returns ``(cand_src, cand_dst, cand_wt, n_cand)``. Only candidate edges with
    ``candidate_indices[i] >= 0`` (a valid neighbor found) are included.
    """
    # One entry per component: best_src, best_dst, best_wt.
    comp_best_src = np.full(n_verts, np.int32(-1), dtype=np.int32)
    comp_best_dst = np.full(n_verts, np.int32(-1), dtype=np.int32)
    comp_best_wt = np.full(n_verts, np.inf, dtype=np.float32)

    for i in range(n_verts):
        if candidate_indices[i] < 0:
            continue
        from_comp = point_components[i]
        wt = candidate_distances[i]
        if wt < comp_best_wt[from_comp]:
            comp_best_src[from_comp] = np.int32(i)
            comp_best_dst[from_comp] = candidate_indices[i]
            comp_best_wt[from_comp] = wt

    # Collect non-empty components.
    n_cand = np.int32(0)
    for c in range(n_verts):
        if comp_best_src[c] >= 0:
            n_cand += np.int32(1)

    out_src = np.empty(n_cand, dtype=np.int32)
    out_dst = np.empty(n_cand, dtype=np.int32)
    out_wt = np.empty(n_cand, dtype=np.float32)
    j = np.int32(0)
    for c in range(n_verts):
        if comp_best_src[c] >= 0:
            out_src[j] = comp_best_src[c]
            out_dst[j] = comp_best_dst[c]
            out_wt[j] = comp_best_wt[c]
            j += np.int32(1)

    return out_src, out_dst, out_wt, n_cand


@numba.njit(cache=NUMBA_CACHE)
def _commit_edges_cl(
    surv_src, surv_dst, surv_wt, n_surviving, disjoint_set, point_components
):
    """
    Commit surviving round edges into the main DSU.

    Returns ``(new_edges, n_added)``:
    - ``new_edges``: float64[:, 3], (src, dst, weight) for distinct-root pairs.
    - ``n_added``: int, edges actually committed.

    Also updates ``point_components`` inline by calling ds_find after each union.
    """
    new_edges = np.empty((n_surviving, 3), dtype=np.float64)
    n_added = np.int32(0)
    for i in range(n_surviving):
        src = surv_src[i]
        dst = surv_dst[i]
        rs = ds_find(disjoint_set, np.int32(src))
        rd = ds_find(disjoint_set, np.int32(dst))
        if rs != rd:
            new_edges[n_added, 0] = np.float64(src)
            new_edges[n_added, 1] = np.float64(dst)
            new_edges[n_added, 2] = np.float64(surv_wt[i])
            n_added += np.int32(1)
            ds_union_by_rank(disjoint_set, rs, rd)
    # Refresh point_components after all unions.
    for i in range(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))
    return new_edges, n_added


# ===========================================================================
# Component-constraint-dictionary merge ("mini_kruskal")
# ===========================================================================
# Instead of tentatively merging each round's candidates and then running a BFS
# heaviest-edge cleanup to break violating paths (the old "current" merge), we
# maintain a single dictionary
#
#     constraints : Dict[component_root -> set(component_root)]
#
# keyed by the *currently active* DSU roots.  ``constraints[A]`` is the set of
# roots A may never merge with.  Symmetry (B in constraints[A] <=> A in
# constraints[B]) and completeness (is_constrained(A, B) is True iff some point
# in A and some point in B form a cannot-link pair) are preserved by folding the
# loser's constraint set into the winner on every accepted union.  All checks
# are O(1); candidates are processed lightest-first so any edge blocked by a
# constraint is the heavier of the conflicting pair.


@numba.njit(cache=NUMBA_CACHE)
def _init_constraint_dict(cl_pair_u, cl_pair_v):
    """Build the initial component-constraint dictionary.

    At the start every point is its own component, so the dictionary is keyed by
    point index.  For each undirected CL pair (u, v) we record the symmetric
    membership ``v in constraints[u]`` and ``u in constraints[v]``.
    """
    constraints = Dict.empty(types.int64, _CL_SET_TYPE)
    M = cl_pair_u.shape[0]
    for k in range(M):
        u = np.int64(cl_pair_u[k])
        v = np.int64(cl_pair_v[k])
        if u not in constraints:
            constraints[u] = Dict.empty(types.int64, types.int8)
        if v not in constraints:
            constraints[v] = Dict.empty(types.int64, types.int8)
        constraints[u][v] = np.int8(1)
        constraints[v][u] = np.int8(1)
    return constraints


@numba.njit(inline="always", cache=NUMBA_CACHE)
def _is_constrained(constraints, from_root, to_root):
    """O(1) test: would merging ``from_root`` and ``to_root`` violate a CL pair?"""
    if from_root not in constraints:
        return False
    return to_root in constraints[from_root]


@numba.njit(cache=NUMBA_CACHE)
def _update_constraints(constraints, from_root, to_root, new_root):
    """Fold the losing component's constraints into the surviving root."""
    old_root = to_root if new_root == from_root else from_root
    if old_root not in constraints:
        return  # loser had no constraints -- nothing to migrate

    old_set = constraints[old_root]
    del constraints[old_root]

    if new_root not in constraints:
        constraints[new_root] = Dict.empty(types.int64, types.int8)
    new_set = constraints[new_root]

    for peer in old_set:
        if peer == new_root:
            continue
        if peer in constraints:
            peer_set = constraints[peer]
            if old_root in peer_set:
                del peer_set[old_root]
            peer_set[new_root] = np.int8(1)
        new_set[peer] = np.int8(1)


@numba.njit(cache=NUMBA_CACHE)
def _merge_components_constraint_dict(
    cand_src, cand_dst, cand_wt, n_cand,
    disjoint_set, point_components, constraints, order_in,
):
    """CL-aware component merge driven by the constraint dictionary (mini_kruskal).

    Replaces ``validate_and_prune_merges`` + ``_commit_edges_cl``.  No BFS, no
    per-round forbidden-CSR rebuild for the merge decision -- every candidate is
    admitted or rejected by an O(1) lookup.  Candidates are processed
    lightest-first so the edge blocked by a constraint is the heavier of any
    conflicting pair.  Pass an empty int64 ``order_in`` to have the lightest-first
    ``argsort`` computed internally.
    """
    if order_in.shape[0] == 0 and n_cand > 0:
        order = np.argsort(cand_wt[:n_cand])
    else:
        order = order_in

    new_edges = np.empty((n_cand, 3), dtype=np.float64)
    n_added = np.int32(0)

    for oi in range(n_cand):
        i = order[oi]
        src = np.int32(cand_src[i])
        dst = np.int32(cand_dst[i])

        from_root = ds_find(disjoint_set, src)
        to_root = ds_find(disjoint_set, dst)
        if from_root == to_root:
            continue

        # O(1) preventive CL check -- no violation ever enters the MST.
        if _is_constrained(constraints, np.int64(from_root), np.int64(to_root)):
            continue

        new_edges[n_added, 0] = np.float64(src)
        new_edges[n_added, 1] = np.float64(dst)
        new_edges[n_added, 2] = np.float64(cand_wt[i])
        n_added += np.int32(1)

        ds_union_by_rank(disjoint_set, from_root, to_root)
        new_root = ds_find(disjoint_set, from_root)
        _update_constraints(
            constraints, np.int64(from_root), np.int64(to_root), np.int64(new_root)
        )

    # Refresh point_components after all unions this round.
    for i in range(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))

    return new_edges[:n_added], n_added


def parallel_boruvka_cl(
    tree,
    n_threads,
    min_samples,
    cl_indices,
    cl_indptr,
    sample_weights=None,
    band_fraction=np.inf,
    band_mode=BAND_MODE_ROUND_MIN_RELATIVE,
    return_banding_metadata=False,
):
    """
    CL-constrained dual-tree Boruvka MST for Euclidean data.

    Uses the KD-tree dual-tree traversal (same as ``parallel_boruvka``) with
    CL pruning inserted at two sites in the recursion.  A per-round
    tentative-merge validation pass removes transitive violations that slip
    through the per-component candidate filter.

    Parameters
    ----------
    tree           : NumbaKDTree
    n_threads      : int (unused; kept for signature parity with parallel_boruvka)
    min_samples    : int
    cl_indices     : int32[:], CSR column indices of symmetric CL graph.
    cl_indptr      : int32[:], CSR row pointers (length n+1).
    sample_weights : float32[:] or None
    band_fraction  : float
        Banding value interpreted according to ``band_mode``. Must be
        non-negative. ``np.inf`` is allowed.
    band_mode : str
        ``"round_min_relative"`` preserves the legacy per-round window:
        candidates within ``w_min * (1 + band_fraction)`` of the current round
        minimum are merged. ``"global_mrd_quantile_schedule"`` uses increasing
        absolute cutoffs from the global initial kNN/core-distance distribution,
        then finishes with an unbounded level so finite bands do not strand the
        forest. In scheduled mode, ``band >= 1.0`` and ``np.inf`` both mean no
        cutoff.
    return_banding_metadata : bool
        If True, return a fourth dict with the estimated band range and
        threshold in internal squared-distance units.

    Returns
    -------
    edges : float64[:, 3]
        (src, dst, mrd_weight), shape (n - 1, 3).
    neighbors : int32[:, :]
        kNN indices from initial query, shape (n, k).
    core_distances : float64[:]
        Core distances, shape (n,).
    """
    from .precomputed import bridge_forest_with_inf

    band_fraction = _validate_banding_inputs(
        band_fraction=band_fraction, band_mode=band_mode
    )
    n = tree.data.shape[0]

    # Delegate to vanilla Boruvka when there are no cannot-link constraints.
    if cl_indices.shape[0] == 0:
        boruvka_sample_weights = (
            np.zeros(1, dtype=np.float32) if sample_weights is None else sample_weights
        )
        result = parallel_boruvka(
            tree, n_threads, min_samples=min_samples,
            sample_weights=boruvka_sample_weights, reproducible=False,
        )
        if return_banding_metadata:
            metadata = _make_empty_banding_metadata(band_mode=band_mode)
            return (*result, metadata)
        return result

    # Compute core distances and neighbors using the same rules as parallel_boruvka.
    if sample_weights is not None:
        mean_sample_weight = float(np.mean(sample_weights))
        expected_neighbors = min_samples / mean_sample_weight
        distances, neighbors = parallel_tree_query(
            tree, tree.data, k=int(2 * expected_neighbors)
        )
        core_distances = sample_weight_core_distance(
            distances, neighbors, sample_weights, min_samples
        )
    elif min_samples > 1:
        distances, neighbors = parallel_tree_query(
            tree, tree.data, k=min_samples + 1, output_rdist=True
        )
        core_distances = distances.T[-1]
    else:
        core_distances = np.zeros(n, dtype=np.float32)
        distances, neighbors = parallel_tree_query(
            tree, tree.data, k=2, output_rdist=True
        )

    global_band_plan, global_band_metadata = _compute_global_mrd_threshold(
        band_fraction=band_fraction,
        band_mode=band_mode,
        neighbors=neighbors,
        distances=distances,
        core_distances=core_distances,
        sample_weights=sample_weights,
    )
    use_round_min_banding = (
        band_mode == BAND_MODE_ROUND_MIN_RELATIVE and band_fraction < 1e30
    )
    use_threshold_schedule = band_mode == BAND_MODE_GLOBAL_MRD_QUANTILE_SCHEDULE
    if use_threshold_schedule:
        global_band_thresholds = global_band_plan.astype(np.float64)
    else:
        global_band_thresholds = np.asarray([float(global_band_plan)], dtype=np.float64)
    global_band_level_index = 0
    global_band_n_advances = 0
    global_band_threshold = float(global_band_thresholds[global_band_level_index])
    use_global_banding = np.isfinite(global_band_threshold)
    candidate_distance_upper_bound = _next_query_upper_bound(global_band_threshold)

    # Extract flat cannot-link pair arrays once, then rebuild component-level
    # constraints each round.
    cl_pair_u, cl_pair_v, M_pairs = _extract_cl_pair_arrays_from_csr(
        cl_indices, cl_indptr
    )

    # mini_kruskal merge state: one maintained component-constraint dictionary
    # (folded on every accepted union) plus a reusable empty order array that
    # tells ``_merge_components_constraint_dict`` to sort candidates internally.
    constraints = _init_constraint_dict(cl_pair_u, cl_pair_v)
    _mk_order = np.empty(0, dtype=np.int64)

    # Reuse validation scratch arrays across rounds.
    max_adj = 2 * n
    _adj_head = np.full(n, -1, dtype=np.int32)
    _adj_next = np.empty(max_adj, dtype=np.int32)
    _adj_edge_idx = np.empty(max_adj, dtype=np.int32)
    _bfs_queue = np.empty(n, dtype=np.int32)
    _bfs_parent_edge = np.full(n, -1, dtype=np.int32)
    _bfs_visited = np.zeros(n, dtype=np.bool_)
    _temp_parent = np.arange(n, dtype=np.int32)

    # Initialize DSU and component arrays.
    components_disjoint_set = ds_rank_create(n)
    point_components = np.arange(n, dtype=np.int32)
    node_components = np.full(tree.idx_start.shape[0], -1, dtype=np.int32)

    # Initial round: collect kNN candidates, validate, and commit.
    init_src, init_dst, init_wt, n_init = _collect_knn_candidates_cl(
        neighbors, distances, core_distances
    )

    if n_init > 0:
        # Drop kNN candidates above the configured band before validation.
        if use_round_min_banding and n_init > 0:
            w_min_init = float(np.min(init_wt))
            band_hi_init = w_min_init * (1.0 + band_fraction)
            mask_init = init_wt <= band_hi_init
            init_src = init_src[mask_init]
            init_dst = init_dst[mask_init]
            init_wt = init_wt[mask_init]
            n_init = int(mask_init.sum())
        elif use_global_banding and n_init > 0:
            mask_init = init_wt <= global_band_threshold
            init_src = init_src[mask_init]
            init_dst = init_dst[mask_init]
            init_wt = init_wt[mask_init]
            n_init = int(mask_init.sum())

        if n_init > 0:
            new_edges, n_added = _merge_components_constraint_dict(
                init_src,
                init_dst,
                init_wt,
                n_init,
                components_disjoint_set,
                point_components,
                constraints,
                _mk_order,
            )
            all_edges = new_edges[:n_added]
            n_components = n - n_added
        else:
            all_edges = np.empty((0, 3), dtype=np.float64)
            n_components = n
    else:
        all_edges = np.empty((0, 3), dtype=np.float64)
        n_components = n

    update_component_vectors(
        tree, components_disjoint_set, node_components, point_components
    )

    # Main Boruvka-CL loop.
    while n_components > 1:
        use_global_banding = np.isfinite(global_band_threshold)
        candidate_distance_upper_bound = _next_query_upper_bound(global_band_threshold)
        if M_pairs > 0:
            forbid_indices, forbid_indptr = _build_forbidden_components_csr(
                cl_pair_u, cl_pair_v, point_components, n
            )
        else:
            forbid_indices = np.empty(0, dtype=np.int32)
            forbid_indptr = np.zeros(n + 1, dtype=np.int32)

        candidate_distances, candidate_indices = _boruvka_tree_query_cl(
            tree,
            node_components,
            point_components,
            core_distances,
            forbid_indices,
            forbid_indptr,
            candidate_distance_upper_bound,
        )

        cand_src, cand_dst, cand_wt, n_cand = _select_components_cl(
            candidate_distances, candidate_indices, point_components, n
        )

        if n_cand == 0:
            if use_threshold_schedule:
                (
                    global_band_level_index,
                    global_band_n_advances,
                    global_band_threshold,
                    advanced,
                ) = _advance_global_band_state(
                    thresholds=global_band_thresholds,
                    level_index=global_band_level_index,
                    n_advances=global_band_n_advances,
                )
                if advanced:
                    continue
            break

        if use_round_min_banding:
            w_min = float(np.min(cand_wt[:n_cand]))
            band_hi = w_min * (1.0 + band_fraction)
            mask = cand_wt[:n_cand] <= band_hi
            cand_src = cand_src[mask]
            cand_dst = cand_dst[mask]
            cand_wt = cand_wt[mask]
            n_cand = int(mask.sum())
            if n_cand == 0:
                if use_threshold_schedule:
                    (
                        global_band_level_index,
                        global_band_n_advances,
                        global_band_threshold,
                        advanced,
                    ) = _advance_global_band_state(
                        thresholds=global_band_thresholds,
                        level_index=global_band_level_index,
                        n_advances=global_band_n_advances,
                    )
                    if advanced:
                        continue
                break
        elif use_global_banding:
            mask = cand_wt[:n_cand] <= global_band_threshold
            cand_src = cand_src[mask]
            cand_dst = cand_dst[mask]
            cand_wt = cand_wt[mask]
            n_cand = int(mask.sum())
            if n_cand == 0:
                if use_threshold_schedule:
                    (
                        global_band_level_index,
                        global_band_n_advances,
                        global_band_threshold,
                        advanced,
                    ) = _advance_global_band_state(
                        thresholds=global_band_thresholds,
                        level_index=global_band_level_index,
                        n_advances=global_band_n_advances,
                    )
                    if advanced:
                        continue
                break

        new_edges, n_added = _merge_components_constraint_dict(
            cand_src,
            cand_dst,
            cand_wt,
            n_cand,
            components_disjoint_set,
            point_components,
            constraints,
            _mk_order,
        )

        if n_added == 0:
            if use_threshold_schedule:
                (
                    global_band_level_index,
                    global_band_n_advances,
                    global_band_threshold,
                    advanced,
                ) = _advance_global_band_state(
                    thresholds=global_band_thresholds,
                    level_index=global_band_level_index,
                    n_advances=global_band_n_advances,
                )
                if advanced:
                    continue
            break

        all_edges = np.vstack((all_edges, new_edges[:n_added]))
        n_components -= n_added

        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components
        )

    # Convert rdist to Euclidean distance.
    all_edges[:, 2] = np.sqrt(all_edges[:, 2])

    # Bridge the forest if cannot-link constraints forced disconnection.
    if n_components > 1:
        all_edges = bridge_forest_with_inf(all_edges, point_components, n)

    result = (all_edges, neighbors[:, 1:], np.sqrt(core_distances))
    if return_banding_metadata:
        if use_threshold_schedule:
            global_band_metadata["band_schedule_levels_visited"] = int(
                global_band_level_index + 1
            )
            global_band_metadata["band_schedule_n_advances"] = int(
                global_band_n_advances
            )
            global_band_metadata["band_schedule_final_threshold_internal"] = float(
                global_band_thresholds[global_band_level_index]
            )
        return (*result, global_band_metadata)
    return result
