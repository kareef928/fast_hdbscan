"""Cannot-link constrained dual-tree Boruvka MST for Euclidean feature data.

Wires cannot-link pruning directly into the KD-tree dual-tree traversal so the
MST can be computed without materializing an O(nk) or O(n^2) CSR.

Key entry point: ``parallel_boruvka_cl``.  Returns the same triple as
``parallel_boruvka``: ``(edges, neighbors, core_distances)``.
"""

import numba
import numpy as np

from .boruvka import (
    parallel_boruvka,
    update_component_vectors,
    sample_weight_core_distance,
)
from .core_graph import validate_and_prune_merges
from .disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
from .numba_kdtree import parallel_tree_query, rdist, point_to_node_lower_bound_rdist
from .variables import NUMBA_CACHE


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
    forbid_indices, forbid_indptr,
):
    """
    CL-aware parallel dual-tree query.

    Returns ``(candidate_distances, candidate_indices)``, the same shape as
    ``boruvka_tree_query``.  Only candidates not violating CL constraints
    (as captured in ``forbid_indices/forbid_indptr``) are accepted.
    """
    n_pts = tree.data.shape[0]
    candidate_distances = np.full(n_pts, np.inf, dtype=np.float32)
    candidate_indices = np.full(n_pts, -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(n_pts, np.inf, dtype=np.float32)

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


def parallel_boruvka_cl(
    tree,
    n_threads,
    min_samples,
    cl_indices,
    cl_indptr,
    sample_weights=None,
    band_fraction=np.inf,
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
        Per-round weight window: only candidates within
        ``w_min * (1 + band_fraction)`` of the round minimum are merged.
        ``np.inf`` = no banding (standard Boruvka, default).
        ``0.05``   = tight band (close to Kruskal ordering, lower
        over-fragmentation under CL constraints).

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

    n = tree.data.shape[0]

    # Delegate to vanilla Boruvka when there are no cannot-link constraints.
    if cl_indices.shape[0] == 0:
        boruvka_sample_weights = (
            np.zeros(1, dtype=np.float32) if sample_weights is None else sample_weights
        )
        return parallel_boruvka(
            tree, n_threads, min_samples=min_samples,
            sample_weights=boruvka_sample_weights, reproducible=False,
        )

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

    use_banding = band_fraction < 1e30

    # Extract flat cannot-link pair arrays once, then rebuild component-level
    # constraints each round.
    cl_pair_u, cl_pair_v, M_pairs = _extract_cl_pair_arrays_from_csr(
        cl_indices, cl_indptr
    )

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
        # Drop kNN candidates above w_min * (1 + band_fraction).
        if use_banding and n_init > 0:
            w_min_init = float(np.min(init_wt))
            band_hi_init = w_min_init * (1.0 + band_fraction)
            mask_init = init_wt <= band_hi_init
            init_src = init_src[mask_init]
            init_dst = init_dst[mask_init]
            init_wt = init_wt[mask_init]
            n_init = int(mask_init.sum())

        if n_init > 0 and M_pairs > 0:
            surv_src, surv_dst, surv_wt, n_surviving = validate_and_prune_merges(
                init_src,
                init_dst,
                init_wt,
                n_init,
                cl_indices,
                cl_indptr,
                point_components,
                n,
                _adj_head,
                _adj_next,
                _adj_edge_idx,
                _bfs_queue,
                _bfs_parent_edge,
                _bfs_visited,
                _temp_parent,
            )
        elif n_init > 0:
            surv_src, surv_dst, surv_wt, n_surviving = (
                init_src,
                init_dst,
                init_wt,
                n_init,
            )
        else:
            n_surviving = 0

        if n_surviving > 0:
            new_edges, n_added = _commit_edges_cl(
                surv_src,
                surv_dst,
                surv_wt,
                n_surviving,
                components_disjoint_set,
                point_components,
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
        )

        cand_src, cand_dst, cand_wt, n_cand = _select_components_cl(
            candidate_distances, candidate_indices, point_components, n
        )

        if n_cand == 0:
            break

        if use_banding:
            w_min = float(np.min(cand_wt[:n_cand]))
            band_hi = w_min * (1.0 + band_fraction)
            mask = cand_wt[:n_cand] <= band_hi
            cand_src = cand_src[mask]
            cand_dst = cand_dst[mask]
            cand_wt = cand_wt[mask]
            n_cand = int(mask.sum())
            if n_cand == 0:
                break

        surv_src, surv_dst, surv_wt, n_surviving = validate_and_prune_merges(
            cand_src,
            cand_dst,
            cand_wt,
            n_cand,
            cl_indices,
            cl_indptr,
            point_components,
            n,
            _adj_head,
            _adj_next,
            _adj_edge_idx,
            _bfs_queue,
            _bfs_parent_edge,
            _bfs_visited,
            _temp_parent,
        )

        if n_surviving == 0:
            break

        new_edges, n_added = _commit_edges_cl(
            surv_src,
            surv_dst,
            surv_wt,
            n_surviving,
            components_disjoint_set,
            point_components,
        )

        if n_added == 0:
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

    return all_edges, neighbors[:, 1:], np.sqrt(core_distances)
