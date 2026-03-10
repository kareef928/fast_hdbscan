"""Constrained Boruvka MST -- KD-tree + precomputed paths with CL enforcement.

Supports two metric pathways:
  - euclidean via KD-tree  -> parallel_boruvka()
  - precomputed via sparse CSR -> minimum_spanning_tree_constrained()

CL (cannot-link) enforcement uses a single unified strategy on both paths:
  1. L1 -- Direct CL pair skip during candidate selection (KD-tree leaf filter
     or CSR neighbor scan).  Prevents individual CL pairs from being proposed.
  2. Preventive DSU check -- Before each merge, walk the linked-list DSU to
     verify that no *transitive* CL violation would be created.  If blocked,
     the merge is skipped and the pair is recorded.
  3. CL expansion -- Blocked pairs are added to the CSR CL graph (grow_cl_csr)
     so that the next round's L1 filter learns about them.  This closes the
     loop: transitive violations discovered by the DSU check become direct
     CL entries for future L1 filtering.
  4. Bridge with +inf -- After the MSF is built, disconnected components are
     bridged with np.inf edges via bridge_forest_with_inf.  The downstream
     cluster_trees.py lambda=0 logic prevents cross-component cluster merging.

No L2 (fix_violations) or L3 (posthoc cleanup) layers are needed -- the
preventive DSU check eliminates all violations before they enter the MST.

Functions whose Numba JIT signatures match the unconstrained base modules are
imported directly.  Only those gaining CL parameters are re-declared here.
"""

import numba
import numpy as np

from .disjoint_set import ds_rank_create, ds_find, ds_find_readonly, ds_union_by_rank
from .variables import NUMBA_CACHE

# -- Imported unchanged from boruvka.py --
# These utility functions have identical signatures and semantics in the
# constrained path; no CL parameters needed.
from .boruvka import (
    update_component_vectors,   # parallel update of point/node component labels
    calculate_block_size,       # adaptive block sizing for reproducible queries
    update_component_bounds_from_block,  # sequential bound merge after prange block
    sample_weight_core_distance,  # weighted core distance computation
)

# -- KD-tree primitives (used by CL-extended query functions) --
from .numba_kdtree import (
    parallel_tree_query, rdist, point_to_node_lower_bound_rdist,
    NumbaKDTree, build_kdtree, simple_heap_push,
)

# -- Precomputed path imports --
from .core_graph import CoreGraph, update_point_components, update_graph_components
from .precomputed import (
    validate_precomputed_sparse_graph,
    bridge_forest_with_inf,
)

# -- Clustering pipeline --
from .hdbscan import clusters_from_spanning_tree


# ===========================================================================
# Constraint validation (pure Python)
# ===========================================================================

def validate_constraints(n_points, cl_indptr, cl_indices):
    """Validate CSR cannot-link arrays (shapes, dtypes, symmetry, no self-links).

    Parameters
    ----------
    n_points : int
        Number of data points.
    cl_indptr : ndarray, int64, shape (n_points + 1,)
        CSR row pointers for the CL graph.
    cl_indices : ndarray, int32, shape (nnz,)
        CSR column indices for the CL graph.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    cl_indptr = np.asarray(cl_indptr)
    cl_indices = np.asarray(cl_indices)

    if cl_indptr.shape != (n_points + 1,):
        raise ValueError(
            f"cl_indptr must have shape ({n_points + 1},), got {cl_indptr.shape}")
    if cl_indptr.dtype != np.int64:
        raise ValueError(f"cl_indptr must have dtype int64, got {cl_indptr.dtype}")
    if cl_indices.dtype != np.int32:
        raise ValueError(f"cl_indices must have dtype int32, got {cl_indices.dtype}")
    if len(cl_indices) > 0:
        if np.any(cl_indices < 0) or np.any(cl_indices >= n_points):
            raise ValueError(
                f"cl_indices contains values outside [0, n_points). "
                f"Range: [{cl_indices.min()}, {cl_indices.max()}], n_points={n_points}")
    for i in range(n_points):
        start, end = int(cl_indptr[i]), int(cl_indptr[i + 1])
        for k in range(start, end):
            if cl_indices[k] == i:
                raise ValueError(f"Self-link detected: point {i} has a CL with itself.")
    for i in range(n_points):
        start, end = int(cl_indptr[i]), int(cl_indptr[i + 1])
        for k in range(start, end):
            j = int(cl_indices[k])
            j_start, j_end = int(cl_indptr[j]), int(cl_indptr[j + 1])
            found = False
            for m in range(j_start, j_end):
                if cl_indices[m] == i:
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Asymmetric CL constraint: ({i}, {j}) present but ({j}, {i}) missing.")


# ===========================================================================
# CL helpers (Numba JIT -- shared by both KD-tree and precomputed paths)
# ===========================================================================

@numba.njit(cache=NUMBA_CACHE, inline="always")
def points_have_cannot_link(point_a, point_b, cl_indptr, cl_indices):
    """Return True if (point_a, point_b) is a direct CL pair.

    Scans point_a's CL neighbor list -- O(degree) per call.  Used for L1
    filtering in both the KD-tree leaf scan and the precomputed CSR scan.
    """
    for k in range(cl_indptr[point_a], cl_indptr[point_a + 1]):
        if cl_indices[k] == point_b:
            return True
    return False


@numba.njit(cache=NUMBA_CACHE)
def check_merge_violates(root_small, root_large, parent, head, next_node,
                         cl_indptr, cl_indices):
    """Return True if merging root_small into root_large would create a CL violation.

    Walks the linked-list of members in root_small's component. For each
    member, scans its CL neighbors and checks (via read-only DSU find)
    whether any CL neighbor already belongs to root_large's component.

    This is the *preventive* check -- it runs BEFORE the union, so no
    violation ever enters the MST.
    """
    node = head[root_small]
    while node != -1:
        for k in range(np.int64(cl_indptr[node]), np.int64(cl_indptr[node + 1])):
            if ds_find_readonly(parent, np.int32(cl_indices[k])) == root_large:
                return True
        node = next_node[node]
    return False


@numba.njit(cache=NUMBA_CACHE, inline="always")
def linked_list_merge(head, tail, next_node, winner, loser):
    """Splice loser's member list onto winner's linked-list chain.

    O(1) operation -- just redirects the tail pointer.  After this call,
    iterating head[winner] -> next_node -> ... visits all members of both
    former components.
    """
    next_node[tail[winner]] = head[loser]
    tail[winner] = tail[loser]


# ===========================================================================
# Dynamic CL expansion (grow CSR in-place within Numba)
# ===========================================================================

@numba.njit(cache=NUMBA_CACHE)
def grow_cl_csr(cl_indptr, cl_indices, new_u, new_v, n_new):
    """Add symmetric CL pairs to CSR arrays.  Returns new (indptr, indices).

    For each pair (u, v), adds u->v and v->u if not already present.
    This is called after each Boruvka round to record blocked merges, so
    that the next round's L1 filter automatically skips those pairs.
    """
    n = cl_indptr.shape[0] - 1

    # First pass: count genuinely-new entries per row
    extra = np.zeros(n, dtype=np.int64)
    for k in range(n_new):
        u = np.int32(new_u[k])
        v = np.int32(new_v[k])
        # u -> v
        found = False
        for j in range(cl_indptr[u], cl_indptr[u + 1]):
            if cl_indices[j] == v:
                found = True
                break
        if not found:
            extra[u] += 1
        # v -> u
        found = False
        for j in range(cl_indptr[v], cl_indptr[v + 1]):
            if cl_indices[j] == u:
                found = True
                break
        if not found:
            extra[v] += 1

    total_extra = np.int64(0)
    for i in range(n):
        total_extra += extra[i]
    if total_extra == 0:
        return cl_indptr, cl_indices

    # Build new CSR arrays with room for the extra entries
    new_nnz = cl_indices.shape[0] + total_extra
    new_indptr = np.empty(n + 1, dtype=np.int64)
    new_indices = np.empty(new_nnz, dtype=np.int32)

    new_indptr[0] = np.int64(0)
    for i in range(n):
        new_indptr[i + 1] = new_indptr[i] + (cl_indptr[i + 1] - cl_indptr[i]) + extra[i]

    for i in range(n):
        old_start = cl_indptr[i]
        old_end = cl_indptr[i + 1]
        ns = new_indptr[i]
        # Copy existing entries
        for j in range(old_end - old_start):
            new_indices[ns + j] = cl_indices[old_start + j]
        # Append new entries for this row
        pos = ns + (old_end - old_start)
        for k in range(n_new):
            u = np.int32(new_u[k])
            v = np.int32(new_v[k])
            target = np.int32(-1)
            if u == np.int32(i):
                target = v
            elif v == np.int32(i):
                target = u
            if target >= 0:
                already = False
                for j in range(old_start, old_end):
                    if cl_indices[j] == target:
                        already = True
                        break
                if not already:
                    new_indices[pos] = target
                    pos += 1

    return new_indptr, new_indices


# ===========================================================================
# KD-tree path -- CL-extended Numba functions
# ===========================================================================
# These functions mirror their boruvka.py counterparts but add CL parameters
# (cl_indptr, cl_indices) and linked-list DSU arrays (head, tail, next_node).
# They cannot be imported because Numba JIT specialises on function signature.

@numba.njit(locals={"i": numba.types.int32, "result_idx": numba.types.int32,
                    "blocked_idx": numba.types.int32},
            cache=NUMBA_CACHE)
def merge_components(disjoint_set, candidate_neighbors, candidate_neighbor_distances,
                     point_components, head, tail, next_node, cl_indptr, cl_indices):
    """Select cheapest cross-component edge per component and merge (KD-tree path).

    Uses multiple passes over the candidate array with flat arrays.  On each
    pass, the cheapest remaining edge per component is found.  If a preventive
    CL check blocks that edge, its distance is invalidated (set to inf) so the
    next pass automatically promotes the component's next-cheapest candidate.

    Returns
    -------
    merged_edges : float64[:, 3]  -- accepted MST edges
    blocked_u, blocked_v : int32[:] -- endpoints of blocked merges (for CL expansion)
    n_blocked : int32
    """
    n_points = candidate_neighbors.shape[0]

    # Work on a copy so we can invalidate blocked edges
    cand_dist = candidate_neighbor_distances.copy()

    component_done = np.zeros(n_points, dtype=numba.boolean)
    result = np.empty((n_points, 3), dtype=np.float64)
    blocked_u = np.empty(n_points, dtype=np.int32)
    blocked_v = np.empty(n_points, dtype=np.int32)
    result_idx = np.int32(0)
    blocked_idx = np.int32(0)

    # Flat arrays indexed by component ID (faster than typed dict)
    best_dist = np.empty(n_points, dtype=np.float32)
    best_src = np.empty(n_points, dtype=np.int32)
    best_dst = np.empty(n_points, dtype=np.int32)

    max_passes = np.int32(5)  # small cap; outer loop handles rest
    for _pass in range(max_passes):
        # Reset per-component best
        for i in range(n_points):
            best_dist[i] = np.float32(np.inf)
            best_src[i] = np.int32(-1)

        # Find cheapest valid edge per unfinished component
        for i in range(n_points):
            comp = point_components[i]
            if component_done[comp]:
                continue
            d = cand_dist[i]
            j = candidate_neighbors[i]
            if j < 0 or np.isnan(d) or d == np.inf:
                continue
            if point_components[j] == comp:
                continue
            if d < best_dist[comp]:
                best_dist[comp] = d
                best_src[comp] = np.int32(i)
                best_dst[comp] = np.int32(j)

        # Try merging each component's best
        had_block = False
        any_candidate = False
        for comp in range(n_points):
            src = best_src[comp]
            if src < 0:
                continue
            any_candidate = True
            dst = best_dst[comp]
            from_comp = ds_find(disjoint_set, src)
            to_comp = ds_find(disjoint_set, dst)
            if from_comp == to_comp:
                continue

            orig_comp = point_components[src]
            if component_done[orig_comp]:
                continue

            # -- Preventive CL check --
            if cl_indptr.shape[0] > 1:
                if check_merge_violates(from_comp, to_comp,
                                        disjoint_set.parent, head, next_node,
                                        cl_indptr, cl_indices):
                    blocked_u[blocked_idx] = src
                    blocked_v[blocked_idx] = dst
                    blocked_idx += 1
                    cand_dist[src] = np.float32(np.inf)
                    had_block = True
                    continue

            # -- Accept merge --
            result[result_idx] = (np.float64(src), np.float64(dst),
                                  np.float64(best_dist[comp]))
            result_idx += 1
            component_done[orig_comp] = True

            from_comp = ds_find(disjoint_set, src)
            to_comp = ds_find(disjoint_set, dst)
            if from_comp == to_comp:
                result_idx -= 1
                continue
            ds_union_by_rank(disjoint_set, from_comp, to_comp)
            new_root = ds_find(disjoint_set, from_comp)
            loser = to_comp if new_root == from_comp else from_comp
            linked_list_merge(head, tail, next_node, new_root, loser)

        if not had_block or not any_candidate:
            break  # no blocked merges or no candidates left

    return (result[:result_idx],
            blocked_u[:blocked_idx], blocked_v[:blocked_idx], blocked_idx)


@numba.njit(
    locals={
        "i": numba.types.int32, "idx": numba.types.int32,
        "left": numba.types.int32, "right": numba.types.int32,
        "d": numba.types.float32,
        "dist_lower_bound_left": numba.types.float32,
        "dist_lower_bound_right": numba.types.float32,
    },
    cache=NUMBA_CACHE, fastmath=True,
)
def component_aware_query_recursion(
        tree, node, point, heap_p, heap_i,
        current_core_distance, core_distances,
        current_component, node_components, point_components,
        dist_lower_bound, component_nearest_neighbor_dist,
        query_point_index, cl_indptr, cl_indices):
    """KD-tree nearest-neighbor search with component awareness and L1 CL filtering.

    Mirrors boruvka.py's component_aware_query_recursion but adds three extra
    parameters (query_point_index, cl_indptr, cl_indices) to skip direct CL
    pairs at the leaf level.  This is the L1 filter for the KD-tree path.

    The pruning rules (Cases 1a-1c) are identical to the unconstrained version:
      1a. dist_lower_bound > heap_p[0]  -> can't improve on current best
      1b. dist_lower_bound or core_dist > component bound -> no gain
      1c. node contains only same-component points -> skip entire subtree
    """
    is_leaf = tree.is_leaf[node]
    idx_start = tree.idx_start[node]
    idx_end = tree.idx_end[node]

    # Case 1a: can't improve on current best distance
    if dist_lower_bound > heap_p[0]:
        return
    # Case 1b: can't improve on component's best distance
    elif (dist_lower_bound > component_nearest_neighbor_dist[0]
          or current_core_distance > component_nearest_neighbor_dist[0]):
        return
    # Case 1c: entire subtree is same component as query
    elif node_components[node] == current_component:
        return
    # Case 2: leaf node -- scan points with L1 CL filtering
    elif is_leaf:
        for i in range(idx_start, idx_end):
            idx = tree.idx_array[i]
            if point_components[idx] == current_component:
                continue
            if core_distances[idx] >= component_nearest_neighbor_dist[0]:
                continue
            # -- L1 filter: skip if direct CL pair --
            if cl_indptr.shape[0] > 1 and points_have_cannot_link(
                    query_point_index, idx, cl_indptr, cl_indices):
                continue
            d = max(rdist(point, tree.data[idx]), current_core_distance, core_distances[idx])
            if d < heap_p[0]:
                simple_heap_push(heap_p, heap_i, d, idx)
                if d < component_nearest_neighbor_dist[0]:
                    component_nearest_neighbor_dist[0] = d
    # Case 3: internal node -- recurse into closer child first
    else:
        left = 2 * node + 1
        right = left + 1
        dist_lower_bound_left = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, left], tree.node_bounds[1, left], point)
        dist_lower_bound_right = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, right], tree.node_bounds[1, right], point)
        if dist_lower_bound_left <= dist_lower_bound_right:
            component_aware_query_recursion(
                tree, left, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_left, component_nearest_neighbor_dist,
                query_point_index, cl_indptr, cl_indices)
            component_aware_query_recursion(
                tree, right, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_right, component_nearest_neighbor_dist,
                query_point_index, cl_indptr, cl_indices)
        else:
            component_aware_query_recursion(
                tree, right, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_right, component_nearest_neighbor_dist,
                query_point_index, cl_indptr, cl_indices)
            component_aware_query_recursion(
                tree, left, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_left, component_nearest_neighbor_dist,
                query_point_index, cl_indptr, cl_indices)
    return


@numba.njit(
    locals={"i": numba.types.int32, "distance_lower_bound": numba.types.float32,
            "current_component": numba.types.int32},
    parallel=True, cache=NUMBA_CACHE, fastmath=True,
)
def boruvka_tree_query(tree, node_components, point_components, core_distances,
                       cl_indptr, cl_indices):
    """Parallel KD-tree query: find each point's cheapest cross-component neighbor.

    Runs component_aware_query_recursion in parallel over all points.
    L1 CL filtering is applied at the leaf level inside the recursion.
    """
    candidate_distances = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    candidate_indices = np.full(tree.data.shape[0], -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    data = tree.data.astype(np.float32)
    for i in numba.prange(tree.data.shape[0]):
        distance_lower_bound = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i])
        heap_p, heap_i = candidate_distances[i:i+1], candidate_indices[i:i+1]
        component_aware_query_recursion(
            tree, 0, data[i], heap_p, heap_i,
            core_distances[i], core_distances,
            point_components[i], node_components, point_components,
            distance_lower_bound,
            component_nearest_neighbor_dist[point_components[i]:point_components[i]+1],
            i, cl_indptr, cl_indices)
    return candidate_distances, candidate_indices


@numba.njit(
    locals={
        "block_start": numba.types.int32, "block_end": numba.types.int32,
        "block_size_actual": numba.types.int32, "i": numba.types.int32,
        "distance_lower_bound": numba.types.float32,
        "current_component": numba.types.int32,
    },
    parallel=True, cache=NUMBA_CACHE, fastmath=True,
)
def boruvka_tree_query_reproducible(tree, node_components, point_components,
                                    core_distances, block_size, cl_indptr, cl_indices):
    """Block-based reproducible KD-tree query with L1 CL filtering.

    Processes points in blocks to avoid race conditions on the shared
    component_nearest_neighbor_dist array.  Within each block, points are
    processed in parallel; between blocks, bounds are merged sequentially.
    """
    candidate_distances = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    candidate_indices = np.full(tree.data.shape[0], -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    data = tree.data.astype(np.float32)
    max_block_component_bounds = np.full(block_size, np.inf, dtype=np.float32)
    for block_start in range(0, tree.data.shape[0], block_size):
        block_end = min(block_start + block_size, tree.data.shape[0])
        block_size_actual = block_end - block_start
        max_block_component_bounds[:block_size_actual] = np.inf
        for i in numba.prange(block_start, block_end):
            distance_lower_bound = point_to_node_lower_bound_rdist(
                tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i])
            heap_p, heap_i = candidate_distances[i:i+1], candidate_indices[i:i+1]
            current_component = point_components[i]
            local_component_bound = component_nearest_neighbor_dist[
                current_component:current_component+1]
            component_aware_query_recursion(
                tree, 0, data[i], heap_p, heap_i,
                core_distances[i], core_distances,
                point_components[i], node_components, point_components,
                distance_lower_bound, local_component_bound,
                i, cl_indptr, cl_indices)
            max_block_component_bounds[i - block_start] = local_component_bound[0]
        update_component_bounds_from_block(
            component_nearest_neighbor_dist, max_block_component_bounds,
            point_components, block_start, block_end)
    return candidate_distances, candidate_indices


@numba.njit(
    locals={
        "i": numba.types.int32, "j": numba.types.int32, "k": numba.types.int32,
        "result_idx": numba.types.int32,
        "from_component": numba.types.int32, "to_component": numba.types.int32,
    },
    parallel=True, cache=NUMBA_CACHE,
)
def initialize_boruvka_from_knn(knn_indices, knn_distances, core_distances,
                                disjoint_set, head, tail, next_node,
                                cl_indptr, cl_indices):
    """Bootstrap MST from KNN edges with CL enforcement.

    For each point, finds the first KNN neighbor that:
      - is not a direct CL pair (L1 skip)
      - has a lower core distance (standard Boruvka direction tie-break)
    Then merges sequentially, checking each merge with check_merge_violates
    (preventive DSU check) before accepting.

    The parallel prange computes candidate edges; the sequential loop merges.
    """
    component_edges = np.full((knn_indices.shape[0], 3), -1, dtype=np.float64)
    for i in numba.prange(knn_indices.shape[0]):
        for j in range(1, knn_indices.shape[1]):
            k = np.int32(knn_indices[i, j])
            # L1: skip direct CL pairs
            if cl_indptr.shape[0] > 1 and points_have_cannot_link(
                    i, k, cl_indptr, cl_indices):
                continue
            if core_distances[i] >= core_distances[k]:
                edge_weight = max(core_distances[i], knn_distances[i, j])
                component_edges[i, 0] = np.float64(i)
                component_edges[i, 1] = np.float64(k)
                component_edges[i, 2] = np.float64(edge_weight)
                break

    # Sequential merge with preventive CL check
    result = np.empty((len(component_edges), 3), dtype=np.float64)
    result_idx = 0
    for edge in component_edges:
        if edge[0] < 0:
            continue
        from_component = ds_find(disjoint_set, np.int32(edge[0]))
        to_component = ds_find(disjoint_set, np.int32(edge[1]))
        if from_component == to_component:
            continue

        # Preventive CL check before merge
        if cl_indptr.shape[0] > 1:
            if check_merge_violates(from_component, to_component,
                                    disjoint_set.parent, head, next_node,
                                    cl_indptr, cl_indices):
                continue  # skip -- main Boruvka loop will find alternatives

        result[result_idx] = (np.float64(edge[0]), np.float64(edge[1]), np.float64(edge[2]))
        result_idx += 1
        from_component = ds_find(disjoint_set, np.int32(edge[0]))
        to_component = ds_find(disjoint_set, np.int32(edge[1]))
        if from_component == to_component:
            result_idx -= 1
            continue
        ds_union_by_rank(disjoint_set, from_component, to_component)
        new_root = ds_find(disjoint_set, from_component)
        loser = to_component if new_root == from_component else from_component
        linked_list_merge(head, tail, next_node, new_root, loser)
    return result[:result_idx]


@numba.njit(cache=NUMBA_CACHE)
def parallel_boruvka(tree, n_threads, min_samples=10, sample_weights=None,
                     reproducible=False, cl_indptr=None, cl_indices=None):
    """Build constrained MST via parallel Boruvka on a KD-tree.

    This is the main entry point for the KD-tree (euclidean) path.
    CL enforcement uses L1 filtering + preventive DSU checks + CL expansion.

    Parameters
    ----------
    tree : NumbaKDTree
        KD-tree built from the data.
    n_threads : int
        Number of threads for parallel query.
    min_samples : int
        Core distance parameter (k-th nearest neighbor distance).
    sample_weights : float32[:] or None
        Per-point sample weights for weighted core distances.
    reproducible : bool
        If True, use block-based query for deterministic results.
    cl_indptr : int64[:] or None
        CSR row pointers for CL graph.  None = no constraints.
    cl_indices : int32[:] or None
        CSR column indices for CL graph.  None = no constraints.

    Returns
    -------
    edges : float64[:, 3] -- MST/MSF edges [src, dst, weight] (euclidean distances)
    knn_neighbors : int32[:, :] -- KNN indices (excluding self)
    core_distances : float32[:] -- per-point core distances (euclidean)
    n_rounds : int -- number of Boruvka rounds
    n_blocked : int -- total number of blocked merges (CL expansion events)
    """
    n_points = tree.data.shape[0]
    if cl_indptr is None:
        cl_indptr = np.zeros(n_points + 1, dtype=np.int64)
    if cl_indices is None:
        cl_indices = np.empty(0, dtype=np.int32)

    # -- Initialise DSU + linked-list structures --
    components_disjoint_set = ds_rank_create(n_points)
    point_components = np.arange(n_points)
    node_components = np.full(tree.idx_start.shape[0], -1)
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)

    # -- KNN initialisation -- compute distances, neighbors, core_distances --
    if sample_weights is not None:
        expected_neighbors = min_samples / np.mean(sample_weights)
        distances, neighbors = parallel_tree_query(
            tree, tree.data, k=int(2 * expected_neighbors))
        core_distances = sample_weight_core_distance(
            distances, neighbors, sample_weights, min_samples)
    elif min_samples > 1:
        distances, neighbors = parallel_tree_query(
            tree, tree.data, k=min_samples + 1, output_rdist=True)
        core_distances = distances.T[-1]
    else:
        distances, neighbors = parallel_tree_query(
            tree, tree.data, k=2, output_rdist=True)
        core_distances = np.zeros(n_points, dtype=np.float32)

    # -- Bootstrap MST from KNN edges (parallel candidate + sequential merge) --
    initial_edges = initialize_boruvka_from_knn(
        neighbors, distances, core_distances, components_disjoint_set,
        head, tail, next_node, cl_indptr, cl_indices)
    update_component_vectors(
        tree, components_disjoint_set, node_components, point_components)

    all_edges = initial_edges
    round_number = np.int64(0)
    total_blocked = np.int64(0)
    n_components = len(np.unique(point_components))

    # -- Main Boruvka loop --
    while n_components > 1:
        round_number += np.int64(1)

        # Step 1: KD-tree query (L1 filters CL pairs at the leaf)
        if reproducible:
            block_size = calculate_block_size(n_components, n_points, n_threads)
            candidate_distances, candidate_indices = boruvka_tree_query_reproducible(
                tree, node_components, point_components, core_distances,
                block_size, cl_indptr, cl_indices)
        else:
            candidate_distances, candidate_indices = boruvka_tree_query(
                tree, node_components, point_components, core_distances,
                cl_indptr, cl_indices)

        # Step 2: merge with preventive DSU check + multi-pass fallback
        (new_edges, blocked_u, blocked_v,
         n_blocked) = merge_components(
            components_disjoint_set, candidate_indices, candidate_distances,
            point_components, head, tail, next_node, cl_indptr, cl_indices)

        if new_edges.shape[0] == 0:
            break  # no progress -- remaining components are CL-separated

        all_edges = np.vstack((all_edges, new_edges))

        # Step 3: grow CL with blocked pairs so next round's L1 skips them
        if n_blocked > 0:
            total_blocked += np.int64(n_blocked)
            cl_indptr, cl_indices = grow_cl_csr(
                cl_indptr, cl_indices, blocked_u, blocked_v, n_blocked)

        # Step 4: update component vectors for next round
        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components)
        n_components = len(np.unique(point_components))

    # Convert from squared distances (rdist) to euclidean distances
    all_edges[:, 2] = np.sqrt(all_edges.T[2])
    return (all_edges, neighbors[:, 1:], np.sqrt(core_distances),
            round_number, total_blocked)


# ===========================================================================
# Precomputed path -- constrained graph-Boruvka functions
# ===========================================================================
# These mirror core_graph.py's select_components / merge_components /
# boruvka_mst but add CL enforcement with the SAME preventive DSU check
# strategy as the KD-tree path above.

@numba.njit(locals={"parent": numba.types.int32}, cache=NUMBA_CACHE)
def select_components_constrained(distances, indices, indptr, point_components,
                                  cl_indptr, cl_indices):
    """Find cheapest cross-component edge per component, skipping CL pairs (L1).

    Mirrors core_graph.select_components but adds L1 CL filtering: for each
    point's sorted neighbor list, skips neighbors that are direct CL pairs.
    Returns a typed dict mapping component_id -> (src, dst, weight).
    """
    component_edges = {
        np.int64(0): (np.int32(0), np.int32(1), np.float32(0.0)) for _ in range(0)
    }
    for parent, from_component in enumerate(point_components):
        start = indptr[parent]
        end = indptr[parent + 1] if parent + 1 < len(indptr) else len(indices)
        best_neighbor = np.int32(-1)
        best_distance = np.float32(np.inf)
        for idx in range(start, end):
            if indices[idx] == -1:
                break
            neighbor = indices[idx]
            distance = distances[idx]
            # Skip same-component neighbors
            if point_components[neighbor] == from_component:
                continue
            # L1: skip direct CL pairs
            if cl_indptr.shape[0] > 1 and points_have_cannot_link(
                    np.int32(parent), neighbor, cl_indptr, cl_indices):
                continue
            if distance < best_distance:
                best_distance = distance
                best_neighbor = neighbor
        if best_neighbor == -1:
            continue
        fc = np.int64(from_component)
        if fc in component_edges:
            if best_distance < component_edges[fc][2]:
                component_edges[fc] = (np.int32(parent), best_neighbor, best_distance)
        else:
            component_edges[fc] = (np.int32(parent), best_neighbor, best_distance)
    return component_edges


@numba.njit(cache=NUMBA_CACHE)
def merge_components_constrained(disjoint_set, component_edges,
                                 head, tail, next_node,
                                 cl_indptr, cl_indices):
    """Merge components from cheapest edges with preventive CL check.

    Like the KD-tree merge_components, each candidate edge is checked with
    check_merge_violates before the union.  Blocked merges are recorded and
    returned for CL expansion.

    Returns
    -------
    merged_edges : float64[:, 3]
    blocked_u, blocked_v : int32[:] -- blocked merge endpoints
    n_blocked : int32
    """
    result = np.empty((len(component_edges), 3), dtype=np.float64)
    blocked_u = np.empty(len(component_edges), dtype=np.int32)
    blocked_v = np.empty(len(component_edges), dtype=np.int32)
    result_idx = np.int32(0)
    blocked_idx = np.int32(0)

    for edge in component_edges.values():
        from_component = ds_find(disjoint_set, edge[0])
        to_component = ds_find(disjoint_set, edge[1])
        if from_component == to_component:
            continue

        # -- Preventive CL check (same as KD-tree path) --
        if cl_indptr.shape[0] > 1:
            if check_merge_violates(from_component, to_component,
                                    disjoint_set.parent, head, next_node,
                                    cl_indptr, cl_indices):
                blocked_u[blocked_idx] = edge[0]
                blocked_v[blocked_idx] = edge[1]
                blocked_idx += 1
                continue

        # -- Accept merge --
        result[result_idx] = (np.float64(edge[0]), np.float64(edge[1]),
                              np.float64(edge[2]))
        result_idx += 1

        from_component = ds_find(disjoint_set, edge[0])
        to_component = ds_find(disjoint_set, edge[1])
        if from_component == to_component:
            result_idx -= 1
            continue
        ds_union_by_rank(disjoint_set, from_component, to_component)
        new_root = ds_find(disjoint_set, from_component)
        loser = to_component if new_root == from_component else from_component
        linked_list_merge(head, tail, next_node, new_root, loser)

    return (result[:result_idx],
            blocked_u[:blocked_idx], blocked_v[:blocked_idx], blocked_idx)


@numba.njit(cache=NUMBA_CACHE)
def minimum_spanning_tree_constrained(graph, cl_indptr, cl_indices, overwrite=False):
    """Constrained graph-Boruvka MST on a CoreGraph (precomputed path).

    Uses the SAME CL enforcement strategy as the KD-tree parallel_boruvka:
      1. L1 skip in select_components_constrained
      2. Preventive DSU check in merge_components_constrained
      3. CL expansion via grow_cl_csr for blocked pairs

    No L2 (fix_violations) or L3 (posthoc cleanup) needed.

    Parameters
    ----------
    graph : CoreGraph namedtuple (weights, distances, indices, indptr)
    cl_indptr : int64[:] -- CSR row pointers for CL graph
    cl_indices : int32[:] -- CSR column indices for CL graph
    overwrite : bool -- if True, modify graph arrays in-place

    Returns
    -------
    n_components : int
    point_components : int32[:]
    edges : float64[:, 3] -- MST/MSF edges
    """
    distances = graph.weights
    indices = graph.indices
    indptr = graph.indptr
    n_points = len(indptr) - 1
    if not overwrite:
        indices = indices.copy()
        distances = distances.copy()

    # -- Initialise DSU + linked-list structures --
    disjoint_set = ds_rank_create(n_points)
    point_components = np.arange(n_points, dtype=np.int32)
    n_components = n_points
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)

    edges_list = [np.empty((0, 3), dtype=np.float64) for _ in range(0)]

    # -- Main Boruvka loop --
    while n_components > 1:
        # Step 1: select cheapest cross-component edge per component (L1 filter)
        comp_edges = select_components_constrained(
            distances, indices, indptr, point_components, cl_indptr, cl_indices)

        # Step 2: merge with preventive DSU check
        (new_edges, blocked_u, blocked_v,
         n_blocked) = merge_components_constrained(
            disjoint_set, comp_edges, head, tail, next_node,
            cl_indptr, cl_indices)

        if new_edges.shape[0] == 0:
            break  # no progress -- remaining components are CL-separated

        edges_list.append(new_edges)

        # Step 3: grow CL with blocked pairs for next round's L1 filter
        if n_blocked > 0:
            cl_indptr, cl_indices = grow_cl_csr(
                cl_indptr, cl_indices, blocked_u, blocked_v, n_blocked)

        # Step 4: update component labels and prune same-component edges
        update_point_components(disjoint_set, point_components)
        update_graph_components(distances, indices, indptr, point_components)
        n_components -= new_edges.shape[0]

    # Concatenate collected edges
    num_edges = 0
    for el in edges_list:
        num_edges += el.shape[0]
    result = np.empty((num_edges, 3), dtype=np.float64)
    counter = 0
    for el in edges_list:
        for e in range(el.shape[0]):
            result[counter, 0] = el[e, 0]
            result[counter, 1] = el[e, 1]
            result[counter, 2] = el[e, 2]
            counter += 1
    return n_components, point_components, result


# ===========================================================================
# High-level entry point
# ===========================================================================

def constrained_hdbscan_from_boruvka(
        tree_or_X, n_threads, cl_indptr, cl_indices, *,
        min_samples=10, min_cluster_size=10,
        sample_weights=None, reproducible=False, return_metadata=False,
        metric="euclidean"):
    """Run constrained HDBSCAN: Boruvka MST -> bridge -> cluster extraction.

    Builds a constrained MST/MSF using preventive DSU checks (no L2/L3 needed).
    Disconnected components are bridged with +inf edges so that cluster_trees.py
    receives exactly n-1 edges.  The lambda=0 logic in get_cluster_label_vector
    prevents cross-component cluster absorption.

    Clustering follows the same pipeline as kruskal.py:
      1. Build MST/MSF with CL enforcement
      2. Bridge disconnected components with bridge_forest_with_inf
      3. Pass n-1 edges to clusters_from_spanning_tree for extraction

    Parameters
    ----------
    tree_or_X : NumbaKDTree or scipy.sparse matrix
        metric='euclidean': NumbaKDTree from build_kdtree.
        metric='precomputed': scipy sparse CSR distance matrix.
    n_threads : int
        Thread count for parallel Boruvka.
    cl_indptr : int64[:] -- CSR row pointers for CL graph.
    cl_indices : int32[:] -- CSR column indices for CL graph.
    min_samples : int
        Core distance parameter.
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.
    sample_weights : float32[:] or None
        Per-point sample weights.
    reproducible : bool
        If True, use block-based deterministic queries (KD-tree path only).
    return_metadata : bool
        If True, return additional metadata dict.
    metric : str
        'euclidean' (KD-tree) or 'precomputed' (sparse CSR).

    Returns
    -------
    labels : int64[:]
    probs : float64[:]
    edges : float64[:, 3]
    metadata : dict (only if return_metadata=True)
    """
    if metric == "euclidean":
        tree = tree_or_X
        n_points = tree.data.shape[0]
        edges, nbrs, core_dists, n_rounds, n_viols = parallel_boruvka(
            tree, n_threads, min_samples=min_samples,
            sample_weights=sample_weights, reproducible=reproducible,
            cl_indptr=cl_indptr, cl_indices=cl_indices)

    elif metric == "precomputed":
        from .precomputed import (
            _symmetrize_min_csr, _core_distances_csr,
            _build_core_graph_csr, _patch_mst_weights,
        )
        X = tree_or_X
        validate_precomputed_sparse_graph(X)
        n_points = X.shape[0]

        # 1. Symmetrize: min weight per undirected pair, no diagonal
        X_sym = _symmetrize_min_csr(X)

        # 2. Core distances and nearest-neighbor indices (parallel over nodes)
        nbrs, core_dists_f64 = _core_distances_csr(
            X_sym.data, X_sym.indices, X_sym.indptr, min_samples)

        # 3. Build CoreGraph with MRD weights, sorted per row (parallel)
        weights, distances, cg_indices = _build_core_graph_csr(
            X_sym.data, X_sym.indices, X_sym.indptr, core_dists_f64)
        core_graph = CoreGraph(weights, distances, cg_indices, X_sym.indptr)

        # 4. Constrained graph-Boruvka MST (same DSU-check strategy as KD-tree)
        n_components, component_labels, edges = minimum_spanning_tree_constrained(
            core_graph, cl_indptr, cl_indices)

        # 5. Bridge disconnected components with +inf edges
        if n_components > 1:
            edges = bridge_forest_with_inf(edges, component_labels, n_points)

        # 6. Restore float64 precision for MRD weights
        mst_weights = _patch_mst_weights(
            edges, X_sym.data, X_sym.indices, X_sym.indptr, core_dists_f64)
        edges = np.column_stack([edges[:, :2], mst_weights])

        n_rounds = np.int64(0)
        n_viols = np.int64(0)

    else:
        raise ValueError(f"Unsupported metric: {metric!r}. Expected 'euclidean' or 'precomputed'.")

    # -- Bridge MSF -> MST (KD-tree path; precomputed path bridges above) --
    if metric == "euclidean" and edges.shape[0] < n_points - 1:
        _ds = ds_rank_create(n_points)
        for _ei in range(edges.shape[0]):
            _a, _b = int(edges[_ei, 0]), int(edges[_ei, 1])
            _ra, _rb = int(ds_find(_ds, _a)), int(ds_find(_ds, _b))
            if _ra != _rb:
                ds_union_by_rank(_ds, _ra, _rb)
        component_labels = np.array(
            [int(ds_find(_ds, i)) for i in range(n_points)], dtype=np.int32)
        _, component_labels = np.unique(component_labels, return_inverse=True)
        component_labels = component_labels.astype(np.int32)
        edges = bridge_forest_with_inf(edges, component_labels, n_points)

    # -- Cluster extraction (same pipeline as kruskal.py) --
    labels, probs, *_ = clusters_from_spanning_tree(
        edges, min_cluster_size=min_cluster_size)

    if return_metadata:
        return labels, probs, edges, {
            "n_rounds": int(n_rounds),
            "violations_detected_mst": int(n_viols)}
    return labels, probs, edges
