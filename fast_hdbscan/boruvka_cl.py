"""
Fast-path CL-constrained dual-tree Borůvka MST for Euclidean feature data.

Wires cannot-link pruning directly into the KD-tree dual-tree traversal so
the MST can be computed without materializing an O(nk) or O(n²) CSR.

Key entry point: ``parallel_boruvka_cl``.  Returns the same triple as
``parallel_boruvka``: ``(edges, neighbors, core_distances)``.
"""

import time

import numba
import numpy as np

from numba import types
from numba.typed import Dict

from .boruvka import (
    parallel_boruvka,
    update_component_vectors,
    boruvka_tree_query,
    initialize_boruvka_from_knn,
    sample_weight_core_distance,
)
from .core_graph import (
    validate_and_prune_merges,
    validate_and_prune_merges_csr,
)
from .disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
from .numba_kdtree import parallel_tree_query, rdist, point_to_node_lower_bound_rdist
from .variables import NUMBA_CACHE


# Inner "set" type for the component-constraint dictionary: a typed Dict used
# as a hash set (key = forbidden component root, value = dummy int8).  Membership
# tests, insertion, and deletion are all O(1).
_CL_SET_TYPE = types.DictType(types.int64, types.int8)


# ---------------------------------------------------------------------------
# Helper: flat CL pair arrays from symmetric CSR (cached once per run)
# ---------------------------------------------------------------------------

def _extract_cl_pair_arrays_from_csr(cl_indices, cl_indptr):
    """
    Extract the upper-triangle CL pairs from a symmetric CSR.

    Parameters
    ----------
    cl_indices : int32[:], CSR column indices of the symmetric CL graph.
    cl_indptr  : int32[:], CSR row pointers (length n+1).

    Returns
    -------
    cl_pair_u : int32[:], shape (M,) — source vertex of each undirected pair (u < v).
    cl_pair_v : int32[:], shape (M,) — destination vertex.
    M         : int — number of undirected CL pairs.
    """
    n = len(cl_indptr) - 1
    # Upper-triangle only (u < v)
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


# ---------------------------------------------------------------------------
# Per-round helper: rebuild forbidden-components CSR
# ---------------------------------------------------------------------------

@numba.njit(cache=NUMBA_CACHE)
def _build_forbidden_components_csr(
    cl_pair_u, cl_pair_v, point_components, n_verts
):
    """
    Build the per-round CSR of inter-component CL prohibitions.

    For each undirected CL pair (u, v), if they belong to distinct components
    (cu, cv), emit both (cu → cv) and (cv → cu) into the CSR.

    Indexed by component id (= DSU root, range 0..n_verts-1), giving the
    sorted, deduplicated list of components that cannot merge with this one
    this round.

    Parameters
    ----------
    cl_pair_u      : int32[:], shape (M,)
    cl_pair_v      : int32[:], shape (M,)
    point_components : int32[:], shape (n,) — current DSU root per point.
    n_verts        : int

    Returns
    -------
    forbid_indices : int32[:] — CSR column indices.
    forbid_indptr  : int32[:] — CSR row pointers, length (n_verts + 1).
    """
    M = cl_pair_u.shape[0]

    # First pass: count directed pairs (cu, cv) with cu != cv
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

    # Build indptr from counts
    forbid_indptr = np.zeros(n_verts + 1, dtype=np.int32)
    for i in range(n_verts):
        forbid_indptr[i + 1] = forbid_indptr[i] + counts[i]

    total = forbid_indptr[n_verts]
    if total == 0:
        return np.empty(0, dtype=np.int32), forbid_indptr

    # Second pass: fill raw (unsorted) column indices
    raw_indices = np.empty(total, dtype=np.int32)
    fill_pos = np.zeros(n_verts, dtype=np.int32)
    for k in range(n_active):
        cu = active_cu[k]
        cv = active_cv[k]
        pos = forbid_indptr[cu] + fill_pos[cu]
        raw_indices[pos] = cv
        fill_pos[cu] += np.int32(1)

    # Third pass: sort each row and deduplicate in-place
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
        # Dedup
        prev = np.int32(-1)
        for idx in range(row.shape[0]):
            val = row[idx]
            if val != prev:
                forbid_indices_out[write_ptr] = val
                write_ptr += np.int32(1)
                prev = val
        new_indptr[cu + 1] = write_ptr

    return forbid_indices_out[:write_ptr], new_indptr


# ---------------------------------------------------------------------------
# Binary search: is target in sorted array[start:end]?
# ---------------------------------------------------------------------------

@numba.njit(inline="always", cache=NUMBA_CACHE)
def _in_sorted(forbid_indices, forbid_indptr, row, target):
    """
    Return True iff ``target`` is in the sorted list ``forbid_indices[forbid_indptr[row]:forbid_indptr[row+1]]``.
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


# ---------------------------------------------------------------------------
# CL-aware tree recursion (clone of component_aware_query_recursion + 2 prune sites)
# ---------------------------------------------------------------------------

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

    Extends ``component_aware_query_recursion`` with two CL-pruning sites:

    §3.1 — Internal-node prune: if a subtree is entirely within one forbidden
           component, skip it (the current component cannot merge with it).
    §3.2 — Leaf filter: candidates in a forbidden component are excluded.
    """
    is_leaf = tree.is_leaf[node]
    idx_start = tree.idx_start[node]
    idx_end = tree.idx_end[node]

    # Case 1a: outside node radius
    if dist_lower_bound > heap_p[0]:
        return

    # Case 1b: cannot beat best distance for this component
    elif (
        dist_lower_bound > component_nearest_neighbor_dist[0]
        or current_core_distance > component_nearest_neighbor_dist[0]
    ):
        return

    # Case 1c: node contains only same component as query
    elif node_components[node] == current_component:
        return

    # §3.1 — CL internal-node prune: node is mono-component AND forbidden
    else:
        nc = node_components[node]
        if nc != np.int32(-1) and nc != current_component:
            if _in_sorted(forbid_indices, forbid_indptr, current_component, nc):
                return

    # Case 2: leaf node
    if is_leaf:
        for i in range(idx_start, idx_end):
            idx = tree.idx_array[i]
            pc = point_components[idx]
            if pc == current_component:
                continue
            ## §3.2 — CL leaf filter: skip forbidden candidates
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

    # Case 3: internal node — recurse both children
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
                tree, left, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_left, component_nearest_neighbor_dist,
                forbid_indices, forbid_indptr,
            )
            _component_aware_query_recursion_cl(
                tree, right, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_right, component_nearest_neighbor_dist,
                forbid_indices, forbid_indptr,
            )
        else:
            _component_aware_query_recursion_cl(
                tree, right, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_right, component_nearest_neighbor_dist,
                forbid_indices, forbid_indptr,
            )
            _component_aware_query_recursion_cl(
                tree, left, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_left, component_nearest_neighbor_dist,
                forbid_indices, forbid_indptr,
            )

    return


# ---------------------------------------------------------------------------
# CL-aware tree query (parallel prange driver)
# ---------------------------------------------------------------------------

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

    Returns ``(candidate_distances, candidate_indices)`` — same shape as
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


# ---------------------------------------------------------------------------
# CL-aware KNN initializer: collect candidates without immediately committing
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _collect_knn_candidates_cl(knn_indices, knn_distances, core_distances):
    """
    Collect per-point best outgoing kNN candidates without touching a DSU.

    For each point i, returns the first neighbor k (in knn order) where
    core_distances[i] >= core_distances[k] — the same selection rule as
    ``initialize_boruvka_from_knn``.

    Returns ``(cand_src, cand_dst, cand_wt, n_cand)`` using raw vertex ids
    (not compressed components) — all initially distinct.
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

    # Compact: keep only entries with valid dst
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


# ---------------------------------------------------------------------------
# Per-component selection: per-vertex candidates → per-component best
# ---------------------------------------------------------------------------

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

    Returns ``(cand_src, cand_dst, cand_wt, n_cand)`` — parallel to
    ``select_components_cl`` in the CSR kernel.  Only candidate edges with
    ``candidate_indices[i] >= 0`` (a valid neighbor found) are included.
    """
    # One entry per component: best_src, best_dst, best_wt
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

    # Collect non-empty components
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


# ---------------------------------------------------------------------------
# Commit surviving edges: union the DSU + emit edge records
# ---------------------------------------------------------------------------

@numba.njit(cache=NUMBA_CACHE)
def _commit_edges_cl(
    surv_src, surv_dst, surv_wt, n_surviving, disjoint_set, point_components
):
    """
    Commit surviving round edges into the main DSU.

    Returns ``(new_edges, n_added)``:
    - ``new_edges``: float64[:, 3] — (src, dst, weight) for distinct-root pairs.
    - ``n_added``  : int — edges actually committed.

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
    # Refresh point_components after all unions
    for i in range(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))
    return new_edges, n_added


# ===========================================================================
# Component-constraint-dictionary path ("mini_kruskal" method)
# ===========================================================================
# Instead of rebuilding a forbidden-components CSR + running a BFS heaviest-edge
# cleanup every round (the "current" method), we maintain a single dictionary
#
#     constraints : Dict[component_root -> set(component_root)]
#
# keyed by the *currently active* DSU roots.  ``constraints[A]`` is the set of
# roots that A may never be merged with.  Two invariants make this both correct
# and cheap:
#
#   * Symmetry:  B in constraints[A]  <=>  A in constraints[B].
#   * Completeness:  for current roots A, B,  is_constrained(A, B) is True iff
#     there exist points a in A and b in B that form a cannot-link pair.  This
#     is *exactly* the CL feasibility condition, including transitive cases,
#     because every merge folds the loser's constraint set into the winner's.
#
# All membership tests are O(1); the total maintenance cost is amortized
# O(C log N) over all Borůvka rounds (C = number of constraints), since each
# constraint is migrated at most O(log N) times as its component grows.


@numba.njit(cache=NUMBA_CACHE)
def _init_constraint_dict(cl_pair_u, cl_pair_v):
    """Build the initial component-constraint dictionary.

    At the start every point is its own component, so the dictionary is keyed
    by point index.  For each undirected CL pair (u, v) we record the symmetric
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
    """O(1) test: would merging ``from_root`` and ``to_root`` violate a CL pair?

    Symmetry means a single-direction lookup suffices.
    """
    if from_root not in constraints:
        return False
    return to_root in constraints[from_root]


@numba.njit(cache=NUMBA_CACHE)
def _update_constraints(constraints, from_root, to_root, new_root):
    """Fold the losing component's constraints into the surviving root.

    Parameters
    ----------
    from_root, to_root : the two roots that were just unioned (pre-union ids).
    new_root           : ``ds_find`` result after the union -- the survivor.

    The loser (the root that is no longer a representative) has its constraint
    set removed from tracking; every peer that referenced the loser is rewired
    to reference ``new_root``; finally the loser's constraints are absorbed into
    the survivor's set so future O(1) checks remain complete.
    """
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
            # Should never happen (we only call this for accepted merges), but
            # guard against creating a self-constraint.
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
    """CL-aware component merge driven by the constraint dictionary (mini-Kruskal).

    This replaces ``validate_and_prune_merges`` + ``_commit_edges_cl`` for the
    ``mini_kruskal`` method.  No BFS, no per-round forbidden-CSR rebuild for the
    merge decision -- every candidate is admitted or rejected by an O(1) lookup.
    Candidates are processed lightest-first so the edge that ends up blocked by
    a constraint is the heavier of any conflicting pair.

    Parameters
    ----------
    cand_src, cand_dst, cand_wt : per-component best candidate edges (point ids
        for src/dst, rdist weight).
    n_cand          : number of candidates.
    disjoint_set    : the main component DSU (mutated in place).
    point_components: refreshed in place after the merges.
    constraints     : the live component-constraint dictionary (mutated).
    order_in        : precomputed lightest-first ``argsort(cand_wt[:n_cand])``
        (int64).  Pass an empty int64 array to have it computed internally --
        the driver hoists this sort out so it can be timed as its own phase.

    Returns
    -------
    new_edges : float64[:, 3] -- accepted (src, dst, weight) records.
    n_added   : int.
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


# ---------------------------------------------------------------------------
# mini_kruskal_dsu: Leland's constraint-root indirection
# ---------------------------------------------------------------------------
# Same lightest-first dict merge as ``mini_kruskal``, but the constraint dict is
# keyed by a *stable constraint-key* decoupled from the main DSU root via two
# flat int64 maps:
#   croot_of[main_root]  -> constraint-key that currently owns that component
#   key2root[constraint] -> main_root that currently owns that constraint-key
# Decoupling lets the merge always absorb the *smaller* forbidden-set into the
# larger (small-to-large), instead of being forced to move whichever set the
# main DSU happened to demote.  That bounds total set-migration at O(C log C)
# and -- more importantly in practice -- avoids the scattered peer-rewriting of
# ``_update_constraints`` (no peer's set is rewritten when a root id changes;
# only a single ``constraint-key`` union is recorded).  The accept/reject
# relation is identical to ``mini_kruskal`` by construction, so the output tree
# is identical; only the bookkeeping cost differs.


@numba.njit(inline="always", cache=NUMBA_CACHE)
def _is_constrained_dsu(constraints, croot_of, from_root, to_root):
    """O(1) CL test through the constraint-key indirection (symmetric)."""
    kf = croot_of[from_root]
    if kf not in constraints:
        return False
    kt = croot_of[to_root]
    return kt in constraints[kf]


@numba.njit(cache=NUMBA_CACHE)
def _update_constraints_dsu(constraints, croot_of, key2root, from_root, to_root, new_root):
    """Fold constraints small-to-large via the constraint-key indirection.

    ``from_root``/``to_root`` are the pre-union main roots, ``new_root`` is the
    surviving main root.  We resolve each to its constraint-key; if both carry
    constraints we absorb the *smaller* set into the larger (small-to-large),
    rewiring only the smaller set's peers, then point the new main root at the
    surviving constraint-key.  No peer touch happens when the loser is
    unconstrained -- the common case -- so most merges cost O(1).
    """
    kf = croot_of[from_root]
    kt = croot_of[to_root]
    has_f = kf in constraints
    has_t = kt in constraints

    if not has_f and not has_t:
        return
    if has_f and not has_t:
        croot_of[new_root] = kf
        key2root[kf] = new_root
        return
    if has_t and not has_f:
        croot_of[new_root] = kt
        key2root[kt] = new_root
        return
    if kf == kt:  # already the same constraint component (defensive)
        croot_of[new_root] = kf
        key2root[kf] = new_root
        return

    # Both constrained: keep the larger set, fold the smaller one in.
    if len(constraints[kf]) >= len(constraints[kt]):
        keep = kf
        drop = kt
    else:
        keep = kt
        drop = kf
    keep_set = constraints[keep]
    drop_set = constraints[drop]
    for peer in drop_set:
        if peer == keep:
            continue
        if peer in constraints:
            peer_set = constraints[peer]
            if drop in peer_set:
                del peer_set[drop]
            peer_set[keep] = np.int8(1)
        keep_set[peer] = np.int8(1)
    del constraints[drop]
    croot_of[new_root] = keep
    key2root[keep] = new_root


@numba.njit(cache=NUMBA_CACHE)
def _materialize_forbidden_csr_from_dict_dsu(constraints, key2root, n_verts):
    """Build the pruning CSR (main-root indexed) from the constraint-key dict.

    Mirrors ``_materialize_forbidden_csr_from_dict`` but translates each
    constraint-key back to its current main root via ``key2root`` so the CSR is
    indexed/valued by main components, matching what the hot query consults.
    """
    counts = np.zeros(n_verts, dtype=np.int32)
    total = np.int64(0)
    for k in constraints:
        r = key2root[k]
        c = len(constraints[k])
        counts[r] = np.int32(c)
        total += np.int64(c)

    forbid_indptr = np.zeros(n_verts + 1, dtype=np.int32)
    for i in range(n_verts):
        forbid_indptr[i + 1] = forbid_indptr[i] + counts[i]

    if total == 0:
        return np.empty(0, dtype=np.int32), forbid_indptr

    forbid_indices = np.empty(total, dtype=np.int32)
    fill_pos = np.zeros(n_verts, dtype=np.int32)
    for k in constraints:
        r = key2root[k]
        base = forbid_indptr[r]
        for k2 in constraints[k]:
            forbid_indices[base + fill_pos[r]] = np.int32(key2root[k2])
            fill_pos[r] += np.int32(1)

    for a in range(n_verts):
        start = forbid_indptr[a]
        end = forbid_indptr[a + 1]
        if end - start > 1:
            forbid_indices[start:end].sort()

    return forbid_indices, forbid_indptr


@numba.njit(cache=NUMBA_CACHE)
def _merge_components_constraint_dict_dsu(
    cand_src, cand_dst, cand_wt, n_cand,
    disjoint_set, point_components, constraints, croot_of, key2root, order_in,
):
    """Lightest-first CL merge driven by the constraint-key indirection.

    Identical decision logic to ``_merge_components_constraint_dict`` (so the
    output tree is identical); only the constraint lookups/updates route through
    ``croot_of``/``key2root`` (Leland's DSU check).  ``order_in`` is the
    precomputed lightest-first order (empty int64 array => compute internally),
    so the driver can time the sort as its own phase.
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

        if _is_constrained_dsu(
            constraints, croot_of, np.int64(from_root), np.int64(to_root)
        ):
            continue

        new_edges[n_added, 0] = np.float64(src)
        new_edges[n_added, 1] = np.float64(dst)
        new_edges[n_added, 2] = np.float64(cand_wt[i])
        n_added += np.int32(1)

        ds_union_by_rank(disjoint_set, from_root, to_root)
        new_root = ds_find(disjoint_set, from_root)
        _update_constraints_dsu(
            constraints, croot_of, key2root,
            np.int64(from_root), np.int64(to_root), np.int64(new_root),
        )

    for i in range(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))

    return new_edges[:n_added], n_added


@numba.njit(cache=NUMBA_CACHE)
def _materialize_forbidden_csr_from_dict(constraints, n_verts):
    """Build the pruning forbidden-components CSR from the maintained dict (Q2).

    Produces the same ``(forbid_indices, forbid_indptr)`` layout as
    ``_build_forbidden_components_csr`` (sorted rows for binary search), but the
    input is the maintained dictionary's *active, deduplicated* forbidden
    component-pairs -- which shrink as components merge -- rather than a fresh
    walk over all C raw point-pairs.
    """
    counts = np.zeros(n_verts, dtype=np.int32)
    total = np.int64(0)
    for a in constraints:
        c = len(constraints[a])
        counts[a] = np.int32(c)
        total += np.int64(c)

    forbid_indptr = np.zeros(n_verts + 1, dtype=np.int32)
    for i in range(n_verts):
        forbid_indptr[i + 1] = forbid_indptr[i] + counts[i]

    if total == 0:
        return np.empty(0, dtype=np.int32), forbid_indptr

    forbid_indices = np.empty(total, dtype=np.int32)
    fill_pos = np.zeros(n_verts, dtype=np.int32)
    for a in constraints:
        base = forbid_indptr[a]
        for b in constraints[a]:
            forbid_indices[base + fill_pos[a]] = np.int32(b)
            fill_pos[a] += np.int32(1)

    # Sort each row (entries are already unique within a set).
    for a in range(n_verts):
        start = forbid_indptr[a]
        end = forbid_indptr[a + 1]
        if end - start > 1:
            forbid_indices[start:end].sort()

    return forbid_indices, forbid_indptr


@numba.njit(cache=NUMBA_CACHE)
def _commit_edges_cl_dict(
    surv_src, surv_dst, surv_wt, n_surviving,
    disjoint_set, point_components, constraints,
):
    """Commit BFS-surviving edges and fold each merge into the dict (Q2).

    The surviving set produced by ``validate_and_prune_merges_dict`` is already
    CL-feasible, so every distinct-root pair is unioned unconditionally; we just
    keep the maintained dictionary current via ``_update_constraints`` so the
    next round's pruning + violation scan stay correct.
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
            new_root = ds_find(disjoint_set, rs)
            _update_constraints(
                constraints, np.int64(rs), np.int64(rd), np.int64(new_root)
            )
    for i in range(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))
    return new_edges, n_added


# ---------------------------------------------------------------------------
# Main driver: parallel_boruvka_cl
# ---------------------------------------------------------------------------

def parallel_boruvka_cl(
    tree,
    n_threads,
    min_samples,
    cl_indices,
    cl_indptr,
    sample_weights=None,
    band_fraction=np.inf,
    method="current",
    timings=None,
):
    """
    CL-constrained dual-tree Borůvka MST for Euclidean data.

    Uses the KD-tree dual-tree traversal (same as ``parallel_boruvka``) with
    CL pruning inserted at two sites in the recursion.  The per-round
    merge/validation step has three interchangeable implementations selected
    by ``method``; the candidate query and banding are identical across all
    three so the comparison isolates the merge strategy.

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
        ``np.inf`` = no banding (standard Borůvka, default).
        ``0.05``   = tight band (close to Kruskal ordering, lower
        over-fragmentation under CL constraints).
    method         : {"current", "current_csr_merge", "mini_kruskal",
                       "maintained_bfs"}
        Merge / CL-validation strategy.  All track which *components* cannot
        merge; they differ in the data structure (rebuilt-per-round CSR vs.
        maintained dictionary), where it is consulted (query pruning vs. merge),
        and the merge order (BFS heaviest-edge cleanup vs. lightest-first).

        - ``"current"`` (default): each round rebuilds a forbidden-components
          CSR (``_build_forbidden_components_csr``) from the raw CL pairs for
          query pruning, then runs ``validate_and_prune_merges`` -- a BFS
          heaviest-edge cleanup of transitive violations (re-scans raw pairs).
        - ``"current_csr_merge"``: identical to ``current`` but the BFS cleanup
          (``validate_and_prune_merges_csr``) iterates the forbidden-components
          CSR already built for pruning instead of re-walking the raw pairs --
          one fewer raw pass per round; output weight-identical to ``current``
          (up to equal-weight ties).
        - ``"mini_kruskal"``: maintains a component-constraint dictionary (built
          once, folded on every merge); candidates are sorted by distance and
          merged lightest-first with an O(1) check, no BFS.  The pruning CSR is
          materialized from the dict each round (no raw rebuild) -- so it shares
          ``maintained_bfs``'s pruning and differs only in the merge step.
        - ``"maintained_bfs"``: keeps ``current``'s BFS heaviest-edge cleanup
          (so output is weight-identical to ``current``) but maintains a
          dictionary, materializes the per-round pruning CSR from it
          (``_materialize_forbidden_csr_from_dict``, O(A)), and runs the BFS
          violation scan over that same CSR (``validate_and_prune_merges_csr``).
          The dict is used only to materialize + fold -- never iterated in the
          hot scan.  Differs from ``current_csr_merge`` only in CSR source
          (dict-materialized vs. raw-rebuilt).

    timings        : dict or None
        If a dict is passed, lightweight per-phase wall-clock seconds are
        accumulated into it (negligible overhead): ``core`` (core-distance + kNN
        setup), ``init`` (initial round), and, summed over main-loop rounds,
        ``tracker`` (per-round forbidden-CSR build / dict materialize),
        ``query`` (dual-tree candidate query), ``select`` (per-component select +
        band filter), ``merge`` (merge / CL-validation), ``update`` (component
        vectors).  Isolates the merge/tracker step from the shared query cost.

    Returns
    -------
    edges          : float64[:, 3] — (src, dst, mrd_weight), shape (n-1, 3).
    neighbors      : int32[:, :]  — kNN indices from initial query (n, k).
    core_distances : float64[:]   — shape (n,).
    """
    from .precomputed import bridge_forest_with_inf

    _valid_methods = (
        "current", "current_csr_merge", "mini_kruskal", "maintained_bfs",
        "mini_kruskal_dsu",
    )
    if method not in _valid_methods:
        raise ValueError(
            "method must be one of %s. Got: %s" % (_valid_methods, method)
        )
    # Optional lightweight per-phase profiling.
    _prof = timings is not None
    if _prof:
        for _k in ("core", "init", "tracker", "query", "select", "sort",
                   "merge", "update"):
            timings[_k] = timings.get(_k, 0.0)
        timings["rounds"] = timings.get("rounds", 0)  # init + main-loop rounds
    # Which methods maintain the component-constraint dictionary at all.
    _use_constraint_dict = method in (
        "mini_kruskal", "maintained_bfs", "mini_kruskal_dsu"
    )
    # Lightest-first dict merge (mini_kruskal) vs. BFS-driven merge.
    _mini_kruskal_merge = method == "mini_kruskal"
    # Leland's constraint-root indirection (small-to-large, no peer rewiring).
    _mini_kruskal_dsu = method == "mini_kruskal_dsu"
    # Q2: BFS cleanup driven by the maintained dict.
    _maintained_bfs = method == "maintained_bfs"
    # current's BFS cleanup, but driven by the per-round forbidden CSR (which is
    # already built for pruning) instead of a second raw-pair scan.
    _csr_merge = method == "current_csr_merge"

    n = tree.data.shape[0]

    # ── Empty-CL fast path: delegate to vanilla parallel_boruvka ──
    if cl_indices.shape[0] == 0:
        return parallel_boruvka(
            tree, n_threads, min_samples=min_samples,
            sample_weights=sample_weights, reproducible=False,
        )

    # ── Compute core distances + neighbors (same as parallel_boruvka) ──
    if _prof:
        _t0 = time.perf_counter()
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
    if _prof:
        timings["core"] += time.perf_counter() - _t0

    # ── Band-fraction gate (mirrors CSR kernel: core_graph.py:1452) ──
    use_banding = band_fraction < 1e30

    # ── Extract flat pair arrays (upper-triangle only, cached once) ──
    cl_pair_u, cl_pair_v, M_pairs = _extract_cl_pair_arrays_from_csr(
        cl_indices, cl_indptr
    )

    # ── Constraint dictionary (dict-based methods): built once, mutated in place ──
    if _use_constraint_dict and M_pairs > 0:
        constraints = _init_constraint_dict(cl_pair_u, cl_pair_v)
    else:
        constraints = None

    # ── Constraint-key indirection maps (mini_kruskal_dsu only) ──
    # Initially every component is its own constraint-key (identity maps).
    if _mini_kruskal_dsu and M_pairs > 0:
        croot_of = np.arange(n, dtype=np.int64)
        key2root = np.arange(n, dtype=np.int64)
    else:
        croot_of = None
        key2root = None

    # ── Scratch arrays for validate_and_prune_merges (reused across rounds) ──
    max_adj = 2 * n
    _adj_head = np.full(n, -1, dtype=np.int32)
    _adj_next = np.empty(max_adj, dtype=np.int32)
    _adj_edge_idx = np.empty(max_adj, dtype=np.int32)
    _bfs_queue = np.empty(n, dtype=np.int32)
    _bfs_parent_edge = np.full(n, -1, dtype=np.int32)
    _bfs_visited = np.zeros(n, dtype=np.bool_)
    _temp_parent = np.arange(n, dtype=np.int32)

    # ── Initialize DSU + component arrays ──
    components_disjoint_set = ds_rank_create(n)
    point_components = np.arange(n, dtype=np.int32)
    node_components = np.full(tree.idx_start.shape[0], -1, dtype=np.int32)

    # ── Per-round merge dispatch (shared by the init round and the main loop) ──
    # Returns (new_edges[:n_added], n_added).  Selects the validation/commit
    # strategy by method; with no constraints (M_pairs == 0) every method just
    # commits all distinct-root candidates.
    _empty_order = np.empty(0, dtype=np.int64)

    def _merge_round(cs, cd, cw, nc, fi=None, fp=None, order=None):
        # ``order`` is the precomputed lightest-first argsort for the dict-merge
        # methods (hoisted + timed in the caller); empty array => merge fn sorts.
        ord_in = _empty_order if order is None else order
        if M_pairs == 0:
            e, na = _commit_edges_cl(
                cs, cd, cw, nc, components_disjoint_set, point_components
            )
            return e[:na], na
        if _mini_kruskal_merge:
            return _merge_components_constraint_dict(
                cs, cd, cw, nc,
                components_disjoint_set, point_components, constraints, ord_in,
            )
        if _mini_kruskal_dsu:
            return _merge_components_constraint_dict_dsu(
                cs, cd, cw, nc,
                components_disjoint_set, point_components,
                constraints, croot_of, key2root, ord_in,
            )
        if _maintained_bfs:
            # Scan the dict-materialized CSR (fast flat-array scan, same as
            # current_csr_merge); the dict is used only to materialize that CSR
            # and to fold at commit -- never iterated in the hot scan.
            ss, sd, sw, ns, _c = validate_and_prune_merges_csr(
                cs, cd, cw, nc, fi, fp, point_components, n,
                _adj_head, _adj_next, _adj_edge_idx,
                _bfs_queue, _bfs_parent_edge, _bfs_visited, _temp_parent,
                np.int32(1),
            )
            e, na = _commit_edges_cl_dict(
                ss, sd, sw, ns,
                components_disjoint_set, point_components, constraints,
            )
            return e[:na], na
        if _csr_merge:
            # current's BFS cleanup, but the violation scan iterates the
            # per-round forbidden CSR (fi/fp) already built for pruning.
            ss, sd, sw, ns, _c = validate_and_prune_merges_csr(
                cs, cd, cw, nc, fi, fp, point_components, n,
                _adj_head, _adj_next, _adj_edge_idx,
                _bfs_queue, _bfs_parent_edge, _bfs_visited, _temp_parent,
                np.int32(1),
            )
            e, na = _commit_edges_cl(
                ss, sd, sw, ns, components_disjoint_set, point_components
            )
            return e[:na], na
        # "current": rebuilt-CSR BFS cleanup over raw CL pairs.
        ss, sd, sw, ns, _c = validate_and_prune_merges(
            cs, cd, cw, nc, cl_indices, cl_indptr, point_components, n,
            _adj_head, _adj_next, _adj_edge_idx,
            _bfs_queue, _bfs_parent_edge, _bfs_visited, _temp_parent,
            np.int32(2), np.int32(1),
        )
        e, na = _commit_edges_cl(
            ss, sd, sw, ns, components_disjoint_set, point_components
        )
        return e[:na], na

    # ── Initial round: collect kNN candidates, validate, commit ──
    # Done in Python to avoid pulling kNN arrays into JIT across boundaries.
    init_src, init_dst, init_wt, n_init = _collect_knn_candidates_cl(
        neighbors, distances, core_distances
    )

    if n_init > 0:
        # Band filter: drop kNN candidates above w_min * (1 + band_fraction)
        if use_banding and n_init > 0:
            w_min_init = float(np.min(init_wt))
            band_hi_init = w_min_init * (1.0 + band_fraction)
            mask_init = init_wt <= band_hi_init
            init_src = init_src[mask_init]
            init_dst = init_dst[mask_init]
            init_wt = init_wt[mask_init]
            n_init = int(mask_init.sum())

        if n_init > 0:
            # The CSR-scan BFS methods need a forbidden CSR for the merge scan.
            # current_csr_merge builds it from raw pairs; maintained_bfs
            # materializes it from the (just-initialized) dict.
            if M_pairs > 0 and _csr_merge:
                fi_init, fp_init = _build_forbidden_components_csr(
                    cl_pair_u, cl_pair_v, point_components, n
                )
            elif M_pairs > 0 and _maintained_bfs:
                fi_init, fp_init = _materialize_forbidden_csr_from_dict(constraints, n)
            else:
                fi_init, fp_init = None, None
            # Hoist + time the lightest-first sort (dict-merge methods only) so
            # it is attributed to "sort" rather than the init/merge phase.  This
            # is the largest single sort (n_components ≈ n in the first round).
            init_order = None
            if (_mini_kruskal_merge or _mini_kruskal_dsu) and M_pairs > 0 and n_init > 0:
                if _prof:
                    _t0 = time.perf_counter()
                init_order = np.argsort(init_wt[:n_init]).astype(np.int64)
                if _prof:
                    timings["sort"] += time.perf_counter() - _t0
            if _prof:
                _t0 = time.perf_counter()
            new_edges, n_added = _merge_round(
                init_src, init_dst, init_wt, n_init, fi_init, fp_init, init_order
            )
            if _prof:
                timings["init"] += time.perf_counter() - _t0
                timings["rounds"] += 1
            all_edges = new_edges
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

    # ── Main Borůvka-CL loop ──
    # Note: _boruvka_tree_query_cl is parallel=True. Calling it from Python
    # allows full thread parallelism. The lightweight serial steps (validate,
    # commit, update) are JIT-compiled helpers called between parallel rounds.
    while n_components > 1:
        if _prof:
            timings["rounds"] += 1
        # ── Candidate query (CL-pruned).  Pruning consults the per-round
        #    forbidden CSR: materialized from the maintained dict when we have
        #    one (mini_kruskal, maintained_bfs) -- O(A), shrinking -- else
        #    rebuilt from raw pairs (current, current_csr_merge) -- O(C). ──
        if _prof:
            _t0 = time.perf_counter()
        if M_pairs > 0:
            if _mini_kruskal_dsu:
                forbid_indices, forbid_indptr = _materialize_forbidden_csr_from_dict_dsu(
                    constraints, key2root, n
                )
            elif _use_constraint_dict:
                forbid_indices, forbid_indptr = _materialize_forbidden_csr_from_dict(
                    constraints, n
                )
            else:
                forbid_indices, forbid_indptr = _build_forbidden_components_csr(
                    cl_pair_u, cl_pair_v, point_components, n
                )
        else:
            forbid_indices = np.empty(0, dtype=np.int32)
            forbid_indptr = np.zeros(n + 1, dtype=np.int32)
        if _prof:
            timings["tracker"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        candidate_distances, candidate_indices = _boruvka_tree_query_cl(
            tree, node_components, point_components, core_distances,
            forbid_indices, forbid_indptr,
        )
        if _prof:
            timings["query"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        cand_src, cand_dst, cand_wt, n_cand = _select_components_cl(
            candidate_distances, candidate_indices, point_components, n
        )

        if n_cand == 0:
            if _prof:
                timings["select"] += time.perf_counter() - _t0
            break

        # Band filter: mirrors CSR kernel core_graph.py:1491-1505
        if use_banding:
            w_min = float(np.min(cand_wt[:n_cand]))
            band_hi = w_min * (1.0 + band_fraction)
            mask = cand_wt[:n_cand] <= band_hi
            cand_src = cand_src[mask]
            cand_dst = cand_dst[mask]
            cand_wt = cand_wt[mask]
            n_cand = int(mask.sum())
            if n_cand == 0:
                if _prof:
                    timings["select"] += time.perf_counter() - _t0
                break
        if _prof:
            timings["select"] += time.perf_counter() - _t0

        # Hoist + time the lightest-first sort (dict-merge methods only) so the
        # "merge" phase is sort-free and the sort scaling is measured on its own.
        round_order = None
        if (_mini_kruskal_merge or _mini_kruskal_dsu) and n_cand > 0:
            if _prof:
                _t0 = time.perf_counter()
            round_order = np.argsort(cand_wt[:n_cand]).astype(np.int64)
            if _prof:
                timings["sort"] += time.perf_counter() - _t0
        if _prof:
            _t0 = time.perf_counter()

        # current_csr_merge reuses this round's pruning CSR for its merge scan.
        new_edges, n_added = _merge_round(
            cand_src, cand_dst, cand_wt, n_cand, forbid_indices, forbid_indptr,
            round_order,
        )
        if _prof:
            timings["merge"] += time.perf_counter() - _t0

        if n_added == 0:
            break

        all_edges = np.vstack((all_edges, new_edges))
        n_components -= n_added

        if _prof:
            _t0 = time.perf_counter()
        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components
        )
        if _prof:
            timings["update"] += time.perf_counter() - _t0

    # ── Convert rdist → Euclidean distance (sqrt) ──
    all_edges[:, 2] = np.sqrt(all_edges[:, 2])

    # ── MSF bridging if CL forced disconnection ──
    if n_components > 1:
        all_edges = bridge_forest_with_inf(all_edges, point_components, n)

    return all_edges, neighbors[:, 1:], np.sqrt(core_distances)
