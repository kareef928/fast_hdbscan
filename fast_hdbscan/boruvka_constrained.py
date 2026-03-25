"""Constrained Boruvka MST -- KD-tree + precomputed paths with CL enforcement.

Supports two metric pathways:
  - euclidean via KD-tree  -> parallel_boruvka()
  - precomputed via sparse CSR -> minimum_spanning_tree_constrained()

CL (cannot-link) enforcement uses a single unified strategy on both paths:
  1. L1 -- Direct CL pair skip during candidate selection.  Uses a *bitset*
     for O(1) lookup instead of scanning CSR neighbor lists.
  2. Preventive merge check -- Before each merge, walk the linked-list members
     of both components and use bitset lookups to verify that no *transitive*
     CL violation would be created.  If blocked, the merge is skipped and
     the pair is recorded.
  3. CL expansion -- Blocked pairs are recorded in the bitset for future
     L1 lookups and merge checks.  O(1) per pair.
  4. Bridge with +inf -- After the MSF is built, disconnected components are
     bridged with np.inf edges via bridge_forest_with_inf.  The downstream
     cluster_trees.py lambda=0 logic prevents cross-component cluster merging.

Data structures for CL storage:
  - cl_bitset : uint64[:, :]  -- Single source of truth for all CL pairs.
      Shape (n_points, ceil(n_points/64)).  Bit (i, j) is set iff i and j
      are a cannot-link pair (direct or transitively discovered).
      Used for both L1 filtering (O(1) pair check) and merge violation
      detection (cross-component walk with O(1) per pair check).

Performance characteristics:
  - L1 check: O(1) via bitset
  - CL expansion: O(1) per blocked pair (set two symmetric bits)
  - Merge check: O(|comp_small| * |comp_large|) with O(1) per pair check
      (cache-friendly bitset access; in practice comp sizes are small early
      and CL checks are cheap relative to KD-tree queries)

No L2 (fix_violations) or L3 (posthoc cleanup) layers are needed -- the
preventive merge check eliminates all violations before they enter the MST.

Functions whose Numba JIT signatures match the unconstrained base modules are
imported directly.  Only those gaining CL parameters are re-declared here.
"""

import numba
import numpy as np

from .disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
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
    _symmetrize_min_csr,
    _core_distances_csr,
    _build_core_graph_csr,
    _patch_mst_weights,
)

# -- Pynndescent (arbitrary metric) imports --
from .nndescent import (
    _check_pynndescent_available,
    build_knn_graph,
    _knn_graph_to_sparse,
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
# CL data structure builders (Numba JIT)
# ===========================================================================

@numba.njit(cache=NUMBA_CACHE)
def build_cl_bitset(n_points, cl_indptr, cl_indices):
    """Build a bitset from the CSR CL graph for O(1) lookups.

    The bitset is the SINGLE source of truth for all CL pairs -- both
    original (from cl_indptr/cl_indices) and dynamically discovered
    (from blocked merges).  Used for L1 filtering and merge checks.

    Returns
    -------
    cl_bitset : uint64[:, :], shape (n_points, ceil(n_points / 64))
    """
    n_words = (n_points + 63) >> 6  # ceil(n_points / 64)
    bitset = np.zeros((n_points, n_words), dtype=np.uint64)
    for i in range(n_points):
        for k in range(cl_indptr[i], cl_indptr[i + 1]):
            j = np.int64(cl_indices[k])
            bitset[i, j >> 6] |= np.uint64(1) << np.uint64(j & 63)
    return bitset


# ===========================================================================
# CL helpers (Numba JIT -- shared by both KD-tree and precomputed paths)
# ===========================================================================

@numba.njit(cache=NUMBA_CACHE, inline="always")
def points_have_cannot_link(point_a, point_b, cl_bitset):
    """Return True if (point_a, point_b) is a CL pair.  O(1) via bitset.

    Replaces the former O(degree) CSR scan.  A single bit-test on the
    precomputed bitset array.
    """
    word = cl_bitset[point_a, np.int64(point_b) >> 6]
    bit = np.uint64(1) << np.uint64(np.int64(point_b) & 63)
    return (word & bit) != np.uint64(0)


@numba.njit(cache=NUMBA_CACHE)
def check_merge_violates(root_small, root_large, head, next_node, cl_bitset):
    """Return True if merging root_small into root_large would create a CL violation.

    Walks the linked-list members of BOTH components and checks every
    (member_small, member_large) pair via the bitset.  This is simpler and
    uses a single data structure (bitset) instead of CSR + overflow.

    Cost: O(|comp_small| * |comp_large|) with O(1) per pair (bitset check).
    In practice this is fast because:
      - Early Boruvka rounds have tiny components (1-4 members)
      - Late rounds have fewer components to check
      - Bitset checks are cache-friendly sequential memory access
      - Merge checking is only ~10-13% of total loop time (query dominates)

    This is the *preventive* check -- it runs BEFORE the union, so no
    violation ever enters the MST.
    """
    # Collect members of the large component for cache-friendly inner loop.
    n_large = np.int32(0)
    node = head[root_large]
    while node != -1:
        n_large += np.int32(1)
        node = next_node[node]

    large_members = np.empty(n_large, dtype=np.int32)
    idx = np.int32(0)
    node = head[root_large]
    while node != -1:
        large_members[idx] = node
        idx += np.int32(1)
        node = next_node[node]

    # Walk small component; for each member, check bitset against all large members.
    node_s = head[root_small]
    while node_s != -1:
        for k in range(n_large):
            if points_have_cannot_link(node_s, large_members[k], cl_bitset):
                return True
        node_s = next_node[node_s]
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
# Dynamic CL expansion (bitset-only -- replaces grow_cl_csr)
# ===========================================================================

@numba.njit(cache=NUMBA_CACHE)
def expand_cl(cl_bitset, blocked_u, blocked_v, n_blocked):
    """Record blocked pairs in the bitset.

    For each blocked pair (u, v), sets the symmetric bits in cl_bitset
    so that future L1 checks and merge-violation checks see the pair.

    O(n_blocked) per call -- just two bit-set operations per pair.
    """
    for k in range(n_blocked):
        u = np.int32(blocked_u[k])
        v = np.int32(blocked_v[k])
        cl_bitset[u, np.int64(v) >> 6] |= np.uint64(1) << np.uint64(np.int64(v) & 63)
        cl_bitset[v, np.int64(u) >> 6] |= np.uint64(1) << np.uint64(np.int64(u) & 63)


# ===========================================================================
# KD-tree path -- CL-extended Numba functions
# ===========================================================================
# These functions mirror their boruvka.py counterparts but add CL parameters.
# L1 filtering and merge checking both use the bitset.

@numba.njit(locals={"i": numba.types.int32, "result_idx": numba.types.int32,
                    "blocked_idx": numba.types.int32},
            cache=NUMBA_CACHE)
def merge_components(disjoint_set, candidate_neighbors, candidate_neighbor_distances,
                     point_components, head, tail, next_node,
                     cl_bitset):
    """Mini-Kruskal merge with multi-candidate fallback.

    Accepts 2-D candidate arrays (n_points × K) where K is the number of
    fallback candidates per point from the KD-tree query.  Candidates are
    sorted per-point by distance so that when a merge is CL-blocked, the
    pointer advances to the next-cheapest candidate *without* a full
    re-query of the KD-tree.

    On each pass:
      1. For each point, find its cheapest *unused* cross-component candidate
         (scanning from ``cand_ptr[i]``).
      2. Collect per-component bests, ARGSORT BY WEIGHT (mini-Kruskal).
      3. Process lightest-first: preventive CL check, accept or block.
      4. Blocked → advance ``cand_ptr[src]`` so next pass uses next candidate.

    Returns
    -------
    merged_edges : float64[:, 3]  -- accepted MST edges
    blocked_u, blocked_v : int32[:] -- endpoints of blocked merges (for CL expansion)
    n_blocked : int32
    """
    n_points = candidate_neighbors.shape[0]
    num_candidates = candidate_neighbors.shape[1]

    # Sort each point's K candidates by distance ascending (heap is max-heap)
    sorted_dist = np.empty((n_points, num_candidates), dtype=np.float32)
    sorted_idx  = np.empty((n_points, num_candidates), dtype=np.int32)
    for i in range(n_points):
        order = np.argsort(candidate_neighbor_distances[i])
        for k in range(num_candidates):
            sorted_dist[i, k] = candidate_neighbor_distances[i, order[k]]
            sorted_idx[i, k]  = candidate_neighbors[i, order[k]]

    # Per-point pointer: index of next candidate to consider
    cand_ptr = np.zeros(n_points, dtype=np.int32)

    component_done = np.zeros(n_points, dtype=numba.boolean)
    result = np.empty((n_points, 3), dtype=np.float64)
    blocked_u = np.empty(n_points * num_candidates, dtype=np.int32)
    blocked_v = np.empty(n_points * num_candidates, dtype=np.int32)
    result_idx = np.int32(0)
    blocked_idx = np.int32(0)

    # Flat arrays indexed by component ID
    best_dist = np.empty(n_points, dtype=np.float32)
    best_src = np.empty(n_points, dtype=np.int32)
    best_dst = np.empty(n_points, dtype=np.int32)

    # Proposal arrays for sorting
    prop_wt = np.empty(n_points, dtype=np.float32)
    prop_src = np.empty(n_points, dtype=np.int32)
    prop_dst = np.empty(n_points, dtype=np.int32)
    prop_comp = np.empty(n_points, dtype=np.int32)

    max_passes = np.int32(5)
    for _pass in range(max_passes):
        # Reset per-component best
        for i in range(n_points):
            best_dist[i] = np.float32(np.inf)
            best_src[i] = np.int32(-1)

        # Find cheapest valid cross-component candidate per component.
        # Each point scans from its cand_ptr to the first valid candidate.
        for i in range(n_points):
            comp = point_components[i]
            if component_done[comp]:
                continue
            for kk in range(cand_ptr[i], num_candidates):
                j = sorted_idx[i, kk]
                d = sorted_dist[i, kk]
                if j < 0 or d >= np.float32(np.inf):
                    break  # no more valid candidates for this point
                if point_components[j] == comp:
                    continue  # same component after earlier merge; skip
                # First valid cross-component candidate for this point
                if d < best_dist[comp]:
                    best_dist[comp] = d
                    best_src[comp] = np.int32(i)
                    best_dst[comp] = np.int32(j)
                break  # only use first valid candidate per point

        # Collect active proposals into flat arrays
        n_proposals = np.int32(0)
        for comp in range(n_points):
            if best_src[comp] < 0:
                continue
            prop_wt[n_proposals] = best_dist[comp]
            prop_src[n_proposals] = best_src[comp]
            prop_dst[n_proposals] = best_dst[comp]
            prop_comp[n_proposals] = np.int32(comp)
            n_proposals += np.int32(1)

        if n_proposals == 0:
            break

        # ── MINI-KRUSKAL STEP: sort proposals by weight ascending ──
        sorted_order = np.argsort(prop_wt[:n_proposals])

        # Process in lightest-first order
        had_block = False
        for si in range(n_proposals):
            pi = sorted_order[si]
            comp = prop_comp[pi]
            src = prop_src[pi]
            dst = prop_dst[pi]

            if component_done[comp]:
                continue

            from_comp = ds_find(disjoint_set, src)
            to_comp = ds_find(disjoint_set, dst)
            if from_comp == to_comp:
                continue

            # -- Preventive CL check (bitset cross-walk) --
            if cl_bitset.shape[0] > 0:
                if check_merge_violates(from_comp, to_comp,
                                        head, next_node, cl_bitset):
                    blocked_u[blocked_idx] = src
                    blocked_v[blocked_idx] = dst
                    blocked_idx += 1
                    cand_ptr[src] += np.int32(1)  # advance to next candidate
                    had_block = True
                    continue

            # -- Accept merge (no redundant ds_find -- already checked above) --
            result[result_idx] = (np.float64(src), np.float64(dst),
                                  np.float64(prop_wt[pi]))
            result_idx += 1
            component_done[comp] = True
            ds_union_by_rank(disjoint_set, from_comp, to_comp)
            new_root = ds_find(disjoint_set, from_comp)
            loser = to_comp if new_root == from_comp else from_comp
            linked_list_merge(head, tail, next_node, new_root, loser)

        if not had_block:
            break  # no blocked merges -- no retries needed

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
        query_point_index, cl_bitset):
    """KD-tree nearest-neighbor search with component awareness and L1 CL filtering.

    Uses cl_bitset for O(1) CL pair checks at the leaf level.

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
    # Case 2: leaf node -- scan points with L1 CL filtering via bitset
    elif is_leaf:
        for i in range(idx_start, idx_end):
            idx = tree.idx_array[i]
            if point_components[idx] == current_component:
                continue
            if core_distances[idx] >= component_nearest_neighbor_dist[0]:
                continue
            # -- L1 filter: O(1) bitset check --
            if cl_bitset.shape[0] > 0 and points_have_cannot_link(
                    query_point_index, idx, cl_bitset):
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
                query_point_index, cl_bitset)
            component_aware_query_recursion(
                tree, right, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_right, component_nearest_neighbor_dist,
                query_point_index, cl_bitset)
        else:
            component_aware_query_recursion(
                tree, right, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_right, component_nearest_neighbor_dist,
                query_point_index, cl_bitset)
            component_aware_query_recursion(
                tree, left, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_left, component_nearest_neighbor_dist,
                query_point_index, cl_bitset)
    return


@numba.njit(
    locals={"i": numba.types.int32, "distance_lower_bound": numba.types.float32,
            "current_component": numba.types.int32},
    parallel=True, cache=NUMBA_CACHE, fastmath=True,
)
def boruvka_tree_query(tree, node_components, point_components, core_distances,
                       cl_bitset, num_candidates):
    """Parallel KD-tree query: find each point's top-K cross-component neighbors.

    Returns 2-D arrays of shape (n_points, num_candidates) so that the merge
    step has fallback candidates when the best is CL-blocked.  This avoids
    the expensive full re-query that previously caused ~27 000 rounds.
    """
    n_points = tree.data.shape[0]
    candidate_distances = np.full((n_points, num_candidates), np.inf, dtype=np.float32)
    candidate_indices = np.full((n_points, num_candidates), np.int32(-1), dtype=np.int32)
    component_nearest_neighbor_dist = np.full(n_points, np.inf, dtype=np.float32)
    data = tree.data.astype(np.float32)
    for i in numba.prange(n_points):
        distance_lower_bound = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i])
        heap_p = candidate_distances[i]   # 1-D slice of size K (max-heap)
        heap_i = candidate_indices[i]     # 1-D slice of size K
        component_aware_query_recursion(
            tree, 0, data[i], heap_p, heap_i,
            core_distances[i], core_distances,
            point_components[i], node_components, point_components,
            distance_lower_bound,
            component_nearest_neighbor_dist[point_components[i]:point_components[i]+1],
            i, cl_bitset)
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
                                    core_distances, block_size, cl_bitset,
                                    num_candidates):
    """Block-based reproducible KD-tree query returning top-K candidates.

    Same as boruvka_tree_query but processes points in blocks to avoid
    race conditions on the shared component_nearest_neighbor_dist array.
    Returns 2-D arrays of shape (n_points, num_candidates).
    """
    n_points = tree.data.shape[0]
    candidate_distances = np.full((n_points, num_candidates), np.inf, dtype=np.float32)
    candidate_indices = np.full((n_points, num_candidates), np.int32(-1), dtype=np.int32)
    component_nearest_neighbor_dist = np.full(n_points, np.inf, dtype=np.float32)
    data = tree.data.astype(np.float32)
    max_block_component_bounds = np.full(block_size, np.inf, dtype=np.float32)
    for block_start in range(0, n_points, block_size):
        block_end = min(block_start + block_size, n_points)
        block_size_actual = block_end - block_start
        max_block_component_bounds[:block_size_actual] = np.inf
        for i in numba.prange(block_start, block_end):
            distance_lower_bound = point_to_node_lower_bound_rdist(
                tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i])
            heap_p = candidate_distances[i]   # 1-D slice of size K
            heap_i = candidate_indices[i]     # 1-D slice of size K
            current_component = point_components[i]
            local_component_bound = component_nearest_neighbor_dist[
                current_component:current_component+1]
            component_aware_query_recursion(
                tree, 0, data[i], heap_p, heap_i,
                core_distances[i], core_distances,
                point_components[i], node_components, point_components,
                distance_lower_bound, local_component_bound,
                i, cl_bitset)
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
                                cl_bitset):
    """Bootstrap MST from KNN edges with CL enforcement.

    For each point, finds the first KNN neighbor that:
      - is not a direct CL pair (L1 via bitset -- O(1))
      - has a lower core distance (standard Boruvka direction tie-break)
    Then merges sequentially, checking each merge with check_merge_violates
    (preventive cross-component bitset walk) before accepting.

    The parallel prange computes candidate edges; the sequential loop merges.
    """
    component_edges = np.full((knn_indices.shape[0], 3), -1, dtype=np.float64)
    for i in numba.prange(knn_indices.shape[0]):
        for j in range(1, knn_indices.shape[1]):
            k = np.int32(knn_indices[i, j])
            # L1: O(1) bitset check
            if cl_bitset.shape[0] > 0 and points_have_cannot_link(
                    i, k, cl_bitset):
                continue
            if core_distances[i] >= core_distances[k]:
                edge_weight = max(core_distances[i], knn_distances[i, j])
                component_edges[i, 0] = np.float64(i)
                component_edges[i, 1] = np.float64(k)
                component_edges[i, 2] = np.float64(edge_weight)
                break

    # Sequential merge with preventive CL check (bitset cross-walk)
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
        if cl_bitset.shape[0] > 0:
            if check_merge_violates(from_component, to_component,
                                    head, next_node, cl_bitset):
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
                     reproducible=False, cl_indptr=None, cl_indices=None,
                     num_candidates=16):
    """Build constrained MST via parallel Boruvka on a KD-tree.

    This is the main entry point for the KD-tree (euclidean) path.
    CL enforcement uses:
      - Bitset for O(1) L1 filtering and merge checks
      - Append-only CL expansion (set symmetric bits, no CSR rebuild)

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
    num_candidates : int
        Number of fallback candidates per point in each KD-tree query
        round.  Higher values reduce the number of expensive re-query
        rounds when candidates are CL-blocked, at the cost of slightly
        wider tree exploration per round.  Default 16.

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

    # -- Build CL data structures --
    cl_bitset = build_cl_bitset(n_points, cl_indptr, cl_indices)

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
        head, tail, next_node,
        cl_bitset)
    update_component_vectors(
        tree, components_disjoint_set, node_components, point_components)

    all_edges = initial_edges
    round_number = np.int64(0)
    total_blocked = np.int64(0)
    n_components = len(np.unique(point_components))

    # -- Main Boruvka loop --
    while n_components > 1:
        round_number += np.int64(1)

        # Step 1: KD-tree query (L1 filters CL pairs via bitset)
        if reproducible:
            block_size = calculate_block_size(n_components, n_points, n_threads)
            candidate_distances, candidate_indices = boruvka_tree_query_reproducible(
                tree, node_components, point_components, core_distances,
                block_size, cl_bitset, num_candidates)
        else:
            candidate_distances, candidate_indices = boruvka_tree_query(
                tree, node_components, point_components, core_distances,
                cl_bitset, num_candidates)

        # Step 2: merge with preventive CL check + multi-pass fallback
        (new_edges, blocked_u, blocked_v,
         n_blocked) = merge_components(
            components_disjoint_set, candidate_indices, candidate_distances,
            point_components, head, tail, next_node,
            cl_bitset)

        if new_edges.shape[0] == 0 and n_blocked == 0:
            break  # no candidates at all -- remaining components are CL-separated

        if new_edges.shape[0] > 0:
            all_edges = np.vstack((all_edges, new_edges))

        # Step 3: expand CL with blocked pairs (bitset only)
        if n_blocked > 0:
            total_blocked += np.int64(n_blocked)
            expand_cl(cl_bitset, blocked_u, blocked_v, n_blocked)

        # Step 4: update component vectors for next round
        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components)
        n_components = len(np.unique(point_components))

    # Convert from squared distances (rdist) to euclidean distances
    all_edges[:, 2] = np.sqrt(all_edges.T[2])
    return (all_edges, neighbors[:, 1:], np.sqrt(core_distances),
            round_number, total_blocked)


# ===========================================================================
# Precomputed path -- K-candidate constrained graph-Boruvka functions
# ===========================================================================
# These mirror core_graph.py's select_components / merge_components /
# boruvka_mst but add CL enforcement with the SAME preventive DSU check
# strategy as the KD-tree path above.  Unlike the KD-tree path's
# single-candidate-per-component approach, these functions collect K
# candidates per point and process them in a single global mini-Kruskal
# pass, eliminating the inner retry loop.

@numba.njit(locals={"parent": numba.types.int32}, parallel=True, cache=NUMBA_CACHE)
def select_k_candidates_constrained(distances, indices, indptr, point_components,
                                     cl_bitset, K):
    """Collect up to K valid cross-component non-CL candidates per point.

    Each point scans its CSR row from the beginning, collecting up to K
    cross-component neighbors that pass L1 CL filtering.  Since CSR rows are
    sorted by MRD weight ascending (from ``_build_core_graph_csr``), the K
    candidates are already in ascending weight order -- no per-point argsort.

    Runs in parallel across all points via ``numba.prange``.

    Returns
    -------
    cand_weights : float32[n_points, K] -- candidate MRD weights (inf = empty)
    cand_dsts    : int32[n_points, K]   -- candidate destinations (-1 = empty)
    cand_count   : int32[n_points]      -- number of valid candidates per point
    """
    n_points = len(point_components)
    cand_weights = np.full((n_points, K), np.float32(np.inf), dtype=np.float32)
    cand_dsts = np.full((n_points, K), np.int32(-1), dtype=np.int32)
    cand_count = np.zeros(n_points, dtype=np.int32)

    for parent in numba.prange(n_points):
        from_component = point_components[parent]
        start = indptr[parent]
        end = indptr[parent + 1] if parent + 1 < len(indptr) else len(indices)
        k_found = np.int32(0)
        for idx in range(start, end):
            if k_found >= K:
                break
            if indices[idx] == -1:
                break
            neighbor = indices[idx]
            distance = distances[idx]
            # Skip same-component neighbors (may appear within a round
            # after earlier merges; not pruned until end of round)
            if point_components[neighbor] == from_component:
                continue
            # L1: O(1) bitset check
            if cl_bitset.shape[0] > 0 and points_have_cannot_link(
                    np.int32(parent), neighbor, cl_bitset):
                continue
            # Valid candidate -- store in slot k_found
            cand_weights[parent, k_found] = np.float32(distance)
            cand_dsts[parent, k_found] = neighbor
            k_found += np.int32(1)
        cand_count[parent] = k_found
    return cand_weights, cand_dsts, cand_count


@numba.njit(cache=NUMBA_CACHE)
def merge_per_component(disjoint_set,
                        cand_weights, cand_dsts, cand_count,
                        point_components,
                        head, tail, next_node,
                        cl_bitset):
    """Per-component merge with K-fallback and both-sides guard.

    Phase A: Build component→members mapping from *point_components*.
    Phase B: For each component, gather its members' candidates and sort
             by weight.  Record each component's lightest candidate weight.
    Phase C: Sort components by lightest candidate weight (O(C log C)).
    Phase D: Process components lightest-first.  For each component, scan
             its sorted candidate list:
               - Skip if source or target already merged this round
                 (both-sides *comp_merged* guard).
               - CL check via *check_merge_violates*.
               - On block: record blocked pair, K-fallback to next candidate.
               - On valid: accept edge, union, mark *comp_merged*, BREAK.

    Returns
    -------
    merged_edges : float64[:, 3]  -- accepted MST edges (src, dst, weight)
    blocked_u, blocked_v : int32[:] -- endpoints of blocked merges
    n_blocked : int32
    """
    n_points = cand_weights.shape[0]
    K = cand_weights.shape[1]

    # ── Phase A: Build component → members mapping (O(N)) ────────────
    # Count points per component
    comp_size = np.zeros(n_points, dtype=np.int32)
    for i in range(n_points):
        comp_size[point_components[i]] += np.int32(1)

    # Collect unique component IDs
    n_unique = np.int32(0)
    for i in range(n_points):
        if comp_size[i] > 0:
            n_unique += np.int32(1)
    unique_comps = np.empty(n_unique, dtype=np.int32)
    ui = np.int32(0)
    for i in range(n_points):
        if comp_size[i] > 0:
            unique_comps[ui] = np.int32(i)
            ui += np.int32(1)

    # Prefix sum for offsets
    comp_start = np.zeros(n_points, dtype=np.int32)
    running = np.int32(0)
    for i in range(n_points):
        comp_start[i] = running
        running += comp_size[i]

    # Fill comp_members (flat array, each component's points packed contiguously)
    comp_fill = np.zeros(n_points, dtype=np.int32)  # per-comp fill cursor
    comp_members = np.empty(n_points, dtype=np.int32)
    for i in range(n_points):
        c = point_components[i]
        comp_members[comp_start[c] + comp_fill[c]] = np.int32(i)
        comp_fill[c] += np.int32(1)

    # ── Phase B: Per-component candidate gathering + sorting ─────────
    # Count total candidates across all points (for pre-allocation)
    n_total = np.int32(0)
    for i in range(n_points):
        n_total += cand_count[i]

    if n_total == 0:
        empty_edges = np.empty((0, 3), dtype=np.float64)
        empty_blocked = np.empty(0, dtype=np.int32)
        return empty_edges, empty_blocked, empty_blocked, np.int32(0)

    # For each component: gather candidates, argsort, store sorted list.
    # We use a single flat buffer for all per-component sorted candidates.
    all_cand_wt = np.empty(n_total, dtype=np.float32)
    all_cand_src = np.empty(n_total, dtype=np.int32)
    all_cand_dst = np.empty(n_total, dtype=np.int32)
    # Per-component: start index into all_cand_* and count
    pc_start = np.zeros(n_unique, dtype=np.int32)
    pc_count = np.zeros(n_unique, dtype=np.int32)
    # Lightest candidate weight per component (for Phase C sorting)
    comp_lightest = np.full(n_unique, np.float32(np.inf), dtype=np.float32)

    fill = np.int32(0)
    for ci in range(n_unique):
        comp_id = unique_comps[ci]
        m_start = comp_start[comp_id]
        m_count = comp_size[comp_id]

        # Count total candidates for this component's members
        comp_n_cand = np.int32(0)
        for mi in range(m_count):
            pt = comp_members[m_start + mi]
            comp_n_cand += cand_count[pt]

        pc_start[ci] = fill
        pc_count[ci] = comp_n_cand

        if comp_n_cand == 0:
            continue

        # Gather into temp arrays for sorting
        tmp_wt = np.empty(comp_n_cand, dtype=np.float32)
        tmp_src = np.empty(comp_n_cand, dtype=np.int32)
        tmp_dst = np.empty(comp_n_cand, dtype=np.int32)
        ti = np.int32(0)
        for mi in range(m_count):
            pt = comp_members[m_start + mi]
            for k in range(cand_count[pt]):
                tmp_wt[ti] = cand_weights[pt, k]
                tmp_src[ti] = np.int32(pt)
                tmp_dst[ti] = cand_dsts[pt, k]
                ti += np.int32(1)

        # Sort by weight within this component
        order = np.argsort(tmp_wt)
        for si in range(comp_n_cand):
            oi = order[si]
            all_cand_wt[fill + si] = tmp_wt[oi]
            all_cand_src[fill + si] = tmp_src[oi]
            all_cand_dst[fill + si] = tmp_dst[oi]

        comp_lightest[ci] = all_cand_wt[fill]  # lightest after sort
        fill += comp_n_cand

    # ── Phase C: Sort components by lightest candidate weight ────────
    comp_order = np.argsort(comp_lightest)

    # ── Phase D: Process components lightest-first ───────────────────
    comp_merged = np.zeros(n_points, dtype=numba.boolean)

    # Preallocate result arrays (upper bound: n_unique merges)
    result = np.empty((n_unique, 3), dtype=np.float64)
    blocked_u = np.empty(n_total, dtype=np.int32)
    blocked_v = np.empty(n_total, dtype=np.int32)
    result_idx = np.int32(0)
    blocked_idx = np.int32(0)

    for oi in range(n_unique):
        ci = comp_order[oi]
        if pc_count[ci] == 0:
            continue

        # Check if this component was already absorbed as a target
        comp_id = unique_comps[ci]
        root_c = ds_find(disjoint_set, comp_id)
        if comp_merged[root_c]:
            continue

        # Scan this component's sorted candidate list (K-fallback)
        cstart = pc_start[ci]
        ccount = pc_count[ci]
        for si in range(ccount):
            src = all_cand_src[cstart + si]
            dst = all_cand_dst[cstart + si]
            wt = all_cand_wt[cstart + si]

            root_src = ds_find(disjoint_set, src)
            root_dst = ds_find(disjoint_set, dst)

            # Same component now (merged by earlier processing)
            if root_src == root_dst:
                continue

            # Both-sides guard: target already merged this round
            if comp_merged[root_dst]:
                continue

            # Preventive CL check (bitset cross-walk)
            if cl_bitset.shape[0] > 0:
                if check_merge_violates(root_src, root_dst,
                                        head, next_node, cl_bitset):
                    blocked_u[blocked_idx] = src
                    blocked_v[blocked_idx] = dst
                    blocked_idx += np.int32(1)
                    continue  # K-fallback: try next candidate

            # Accept merge
            result[result_idx] = (np.float64(src), np.float64(dst),
                                  np.float64(wt))
            result_idx += np.int32(1)
            ds_union_by_rank(disjoint_set, root_src, root_dst)
            new_root = ds_find(disjoint_set, root_src)
            loser = root_dst if new_root == root_src else root_src
            linked_list_merge(head, tail, next_node, new_root, loser)
            comp_merged[new_root] = True
            break  # early stop: this component is done for the round

    return (result[:result_idx],
            blocked_u[:blocked_idx], blocked_v[:blocked_idx], blocked_idx)


@numba.njit(cache=NUMBA_CACHE)
def minimum_spanning_tree_constrained(graph, cl_indptr, cl_indices,
                                      overwrite=False, K=8):
    """Constrained graph-Borůvka MST on a CoreGraph (precomputed path).

    Uses K-candidate per-point selection with per-component merge,
    K-fallback, and both-sides comp_merged guard.  Each round:
      1. Parallel select: Collect K cross-component non-CL candidates per
         point (numba.prange).
      2. Per-component merge: Group candidates by component, sort within
         each component, process components lightest-first.  Each component
         gets ONE merge attempt with K-fallback on CL blocks.
      3. expand_cl: Record blocked pairs in bitset.
      4. Compact CSR rows (prune same-component edges).

    CL enforcement:
      1. L1 via bitset in select_k_candidates_constrained
      2. Preventive cross-component bitset walk in merge_per_component
      3. CL expansion via expand_cl (set symmetric bits in bitset)

    Parameters
    ----------
    graph : CoreGraph namedtuple (weights, distances, indices, indptr)
    cl_indptr : int64[:] -- CSR row pointers for CL graph
    cl_indices : int32[:] -- CSR column indices for CL graph
    overwrite : bool -- if True, modify graph arrays in-place
    K : int -- number of fallback candidates per point (default 8)

    Returns
    -------
    n_components : int
    point_components : int32[:]
    edges : float64[:, 3] -- MST/MSF edges
    n_rounds : int -- number of Borůvka rounds executed
    """
    distances = graph.weights
    indices = graph.indices
    indptr = graph.indptr
    n_points = len(indptr) - 1
    if not overwrite:
        indices = indices.copy()
        distances = distances.copy()

    # -- Build CL data structures --
    cl_bitset = build_cl_bitset(n_points, cl_indptr, cl_indices)

    # -- Initialise DSU + linked-list structures --
    disjoint_set = ds_rank_create(n_points)
    point_components = np.arange(n_points, dtype=np.int32)
    n_components = n_points
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)

    edges_list = [np.empty((0, 3), dtype=np.float64) for _ in range(0)]

    max_rounds = n_points  # safety bound

    rounds_done = np.int32(0)

    # -- Main Borůvka loop (per-component merge, K-fallback) --
    for _round in range(max_rounds):
        if n_components <= 1:
            break

        # Step 1: Collect K candidates per point in parallel
        #         (CSR rows are weight-sorted, so the K candidates are
        #         already in ascending weight order per point)
        (cand_weights, cand_dsts,
         cand_count) = select_k_candidates_constrained(
            distances, indices, indptr, point_components,
            cl_bitset, K)

        # Step 2: Per-component merge with K-fallback + both-sides guard
        (new_edges, blocked_u, blocked_v,
         n_blocked) = merge_per_component(
            disjoint_set,
            cand_weights, cand_dsts, cand_count,
            point_components,
            head, tail, next_node,
            cl_bitset)

        round_merged = np.int32(new_edges.shape[0])
        if round_merged > 0:
            edges_list.append(new_edges)

        # Step 3: Expand CL with blocked pairs (bitset only)
        if n_blocked > 0:
            expand_cl(cl_bitset, blocked_u, blocked_v, n_blocked)

        if round_merged == 0:
            break  # no merges this round → remaining components CL-separated

        # Step 4: Update component labels and prune same-component edges
        update_point_components(disjoint_set, point_components)
        update_graph_components(distances, indices, indptr, point_components)
        n_components -= round_merged
        rounds_done += np.int32(1)

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
    return n_components, point_components, result, rounds_done


# ===========================================================================
# High-level entry point
# ===========================================================================

def constrained_hdbscan_from_boruvka(
        tree_or_X, n_threads, cl_indptr, cl_indices, *,
        min_samples=10, min_cluster_size=10,
        sample_weights=None, reproducible=False, return_metadata=False,
        metric="euclidean", metric_kwds=None, knn_k=None,
        num_candidates=16, merge_k=8):
    """Run constrained HDBSCAN: Boruvka MST -> bridge -> cluster extraction.

    Builds a constrained MST/MSF using:
      - Bitset for O(1) L1 CL filtering and merge-violation detection
      - Preventive cross-component bitset walk for transitive violation prevention
      - Append-only CL expansion (set symmetric bits, no CSR rebuild)

    Clustering follows the same pipeline as kruskal.py:
      1. Build MST/MSF with CL enforcement
      2. Bridge disconnected components with bridge_forest_with_inf
      3. Pass n-1 edges to clusters_from_spanning_tree for extraction

    Parameters
    ----------
    tree_or_X : NumbaKDTree, scipy.sparse matrix, or ndarray
        metric='euclidean': NumbaKDTree from build_kdtree.
        metric='precomputed': scipy sparse CSR distance matrix.
        Any other metric: dense feature matrix (n_samples, n_features).
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
        'euclidean' (KD-tree), 'precomputed' (sparse CSR), or any metric
        supported by pynndescent (e.g. 'cosine', 'manhattan', 'minkowski',
        'haversine', 'hamming', 'jaccard', 'correlation', etc.).
    metric_kwds : dict or None
        Additional keyword arguments for the distance metric (pynndescent only).
        For example, ``metric_kwds={'p': 3}`` for Minkowski.
    knn_k : int or None
        Number of neighbors for the KNN graph (pynndescent metrics only).
        If None, defaults to max(3 * min_samples, 15).
    num_candidates : int
        Number of KD-tree query candidates (KD-tree path only).
    merge_k : int
        Number of fallback candidates per point in the K-candidate merge
        (precomputed/pynndescent paths only).  Higher values reduce the
        number of outer Borůvka rounds at the cost of more per-round work.
        Default 8.

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
            cl_indptr=cl_indptr, cl_indices=cl_indices,
            num_candidates=num_candidates)

    elif metric == "precomputed":
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
        n_components, component_labels, edges, n_rounds_mst = minimum_spanning_tree_constrained(
            core_graph, cl_indptr, cl_indices, K=merge_k)

        # 5. Bridge disconnected components with +inf edges
        if n_components > 1:
            edges = bridge_forest_with_inf(edges, component_labels, n_points)

        # 6. Restore float64 precision for MRD weights
        mst_weights = _patch_mst_weights(
            edges, X_sym.data, X_sym.indices, X_sym.indptr, core_dists_f64)
        edges = np.column_stack([edges[:, :2], mst_weights])

        n_rounds = np.int64(n_rounds_mst)
        n_viols = np.int64(0)

    else:
        # ----- Arbitrary metric via pynndescent KNN graph -----
        # Build a KNN graph using pynndescent, convert to sparse CSR,
        # then feed into the precomputed constrained Boruvka path.
        _check_pynndescent_available()

        X = np.asarray(tree_or_X)
        n_points = X.shape[0]

        # Determine KNN k
        _knn_k = knn_k if knn_k is not None else max(3 * min_samples, 15)
        _knn_k = max(_knn_k, min_samples)

        # Build approximate KNN graph via pynndescent
        _nnd_index, knn_indices, knn_distances = build_knn_graph(
            X, _knn_k, metric=metric, metric_kwds=metric_kwds)

        # Convert KNN graph to symmetric sparse CSR distance matrix
        X_sym = _knn_graph_to_sparse(knn_indices, knn_distances, n_points)

        # Core distances and nearest-neighbor indices
        nbrs, core_dists_f64 = _core_distances_csr(
            X_sym.data, X_sym.indices, X_sym.indptr, min_samples)

        # Build CoreGraph with MRD weights, sorted per row
        weights, distances, cg_indices = _build_core_graph_csr(
            X_sym.data, X_sym.indices, X_sym.indptr, core_dists_f64)
        core_graph = CoreGraph(weights, distances, cg_indices, X_sym.indptr)

        # Constrained graph-Boruvka MST
        n_components, component_labels, edges, n_rounds_mst = minimum_spanning_tree_constrained(
            core_graph, cl_indptr, cl_indices, K=merge_k)

        # Bridge disconnected components with +inf edges
        if n_components > 1:
            edges = bridge_forest_with_inf(edges, component_labels, n_points)

        # Restore float64 precision for MRD weights
        mst_weights = _patch_mst_weights(
            edges, X_sym.data, X_sym.indices, X_sym.indptr, core_dists_f64)
        edges = np.column_stack([edges[:, :2], mst_weights])

        n_rounds = np.int64(n_rounds_mst)
        n_viols = np.int64(0)

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
