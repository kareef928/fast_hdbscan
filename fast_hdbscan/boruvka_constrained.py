"""Constrained Borůvka MST — KD-tree + precomputed paths with CL enforcement.

Supports two metric pathways (euclidean via KD-tree, precomputed via sparse CSR)
using a unified 3-layer cannot-link (CL) enforcement model:
  L1 — Direct CL skip during candidate selection.
  L2 — fix_violations after each Borůvka round (transitive violation repair).
  L3 — Post-hoc label splitting via greedy graph-coloring (_apply_posthoc_cleanup).

Functions whose Numba JIT signatures match the unconstrained base modules are
imported directly. Only those gaining CL parameters are re-declared here.

See boruvka_constrained_verbose.py for the annotated reference implementation.
"""

import numba
import numpy as np

from .disjoint_set import ds_rank_create, ds_find, ds_find_readonly, ds_union_by_rank
from .variables import NUMBA_CACHE
from .boruvka import (
    select_components, update_component_vectors, calculate_block_size,
    update_component_bounds_from_block, sample_weight_core_distance,
)
from .numba_kdtree import (
    parallel_tree_query, rdist, point_to_node_lower_bound_rdist,
    NumbaKDTree, build_kdtree,
)
from .core_graph import CoreGraph, update_point_components, update_graph_components
from .precomputed import (
    validate_precomputed_sparse_graph, extract_undirected_min_edges,
    build_adjacency_lists, compute_sparse_core_distances,
    apply_mutual_reachability, to_core_graph_arrays,
)
from .hdbscan import clusters_from_spanning_tree


# ---------------------------------------------------------------------------
# Constraint validation (pure Python)
# ---------------------------------------------------------------------------

def validate_constraints(n_points, cl_indptr, cl_indices):
    """Validate CSR cannot-link arrays (shapes, dtypes, symmetry, no self-links)."""
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


# ---------------------------------------------------------------------------
# CL helpers (Numba JIT — shared by both paths)
# ---------------------------------------------------------------------------

@numba.njit(cache=NUMBA_CACHE, inline="always")
def points_have_cannot_link(point_a, point_b, cl_indptr, cl_indices):
    """Return True if (point_a, point_b) is a direct CL pair (O(degree) scan)."""
    for k in range(cl_indptr[point_a], cl_indptr[point_a + 1]):
        if cl_indices[k] == point_b:
            return True
    return False


@numba.njit(cache=NUMBA_CACHE)
def check_merge_violates(root_small, root_large, parent, head, next_node,
                         cl_indptr, cl_indices):
    """Return True if merging root_small into root_large would create a CL violation."""
    node = head[root_small]
    while node != -1:
        for k in range(np.int64(cl_indptr[node]), np.int64(cl_indptr[node + 1])):
            if ds_find_readonly(parent, np.int32(cl_indices[k])) == root_large:
                return True
        node = next_node[node]
    return False


@numba.njit(cache=NUMBA_CACHE)
def component_has_violation(root, parent, head, next_node, cl_indptr, cl_indices):
    """Return True if component *root* contains an internal CL violation."""
    node = head[root]
    while node != -1:
        for k in range(np.int64(cl_indptr[node]), np.int64(cl_indptr[node + 1])):
            if ds_find_readonly(parent, np.int32(cl_indices[k])) == root:
                return True
        node = next_node[node]
    return False


# ---------------------------------------------------------------------------
# Linked-list DSU helpers (shared by both paths)
# ---------------------------------------------------------------------------

@numba.njit(cache=NUMBA_CACHE, inline="always")
def linked_list_merge(head, tail, next_node, winner, loser):
    """Splice loser's member list onto winner's linked-list chain."""
    next_node[tail[winner]] = head[loser]
    tail[winner] = tail[loser]


@numba.njit(cache=NUMBA_CACHE)
def rebuild_linked_list_dsu(n_points, mst_edges, n_edges):
    """Rebuild full DSU + linked-list from scratch by replaying mst_edges[0:n_edges]."""
    parent = np.arange(n_points, dtype=np.int32)
    rank = np.zeros(n_points, dtype=np.int32)
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)
    for e in range(n_edges):
        u = np.int32(mst_edges[e, 0])
        v = np.int32(mst_edges[e, 1])
        ru = u
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]
        rv = v
        while parent[rv] != rv:
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
        next_node[tail[ru]] = head[rv]
        tail[ru] = tail[rv]
    return parent, rank, head, tail, next_node


# ---------------------------------------------------------------------------
# L2 — fix_violations (shared by both paths)
# ---------------------------------------------------------------------------

@numba.njit(cache=NUMBA_CACHE)
def fix_violations(n_points, mst_edges, n_added, round_edges_u, round_edges_v,
                   round_edges_w, round_count, cl_indptr, cl_indices,
                   parent, head, tail, next_node):
    """Detect and repair CL violations from the last merge round (Layer 2).

    Finds violating components, removes their heaviest round-edge, rebuilds
    the DSU, and repeats until clean.
    """
    removed_edges = np.empty((round_count, 3), dtype=np.float64)
    n_removed = np.int32(0)
    round_edge_removed = np.zeros(round_count, dtype=numba.boolean)

    # Build initial worklist of violating component roots
    worklist = np.empty(n_points, dtype=np.int32)
    worklist_count = np.int32(0)
    for r in range(n_points):
        if parent[r] == r:
            if component_has_violation(r, parent, head, next_node, cl_indptr, cl_indices):
                worklist[worklist_count] = r
                worklist_count += 1

    max_iters = round_count + 1
    iters = 0
    while worklist_count > 0 and iters < max_iters:
        iters += 1
        worklist_count -= 1
        current_root = worklist[worklist_count]

        # Re-find root (may have changed after prior rebuild)
        cr = current_root
        while parent[cr] != cr:
            cr = parent[cr]
        current_root = cr

        if not component_has_violation(current_root, parent, head, next_node,
                                       cl_indptr, cl_indices):
            continue

        # Find the heaviest round-edge inside this component
        best_idx = np.int32(-1)
        best_w = -np.inf
        for ri in range(round_count):
            if round_edge_removed[ri]:
                continue
            eu, ev, ew = round_edges_u[ri], round_edges_v[ri], round_edges_w[ri]
            eru = eu
            while parent[eru] != eru:
                eru = parent[eru]
            erv = ev
            while parent[erv] != erv:
                erv = parent[erv]
            if eru == current_root and erv == current_root and ew > best_w:
                best_w = ew
                best_idx = ri

        if best_idx == -1:
            continue

        round_edge_removed[best_idx] = True
        removed_u = round_edges_u[best_idx]
        removed_v = round_edges_v[best_idx]
        removed_w = round_edges_w[best_idx]
        removed_edges[n_removed, 0] = np.float64(removed_u)
        removed_edges[n_removed, 1] = np.float64(removed_v)
        removed_edges[n_removed, 2] = removed_w
        n_removed += 1

        # Compact mst_edges: remove all round_edge_removed entries
        temp_mst = np.empty((n_added, 3), dtype=np.float64)
        temp_n = np.int32(0)
        for e in range(n_added):
            eu2 = np.int32(mst_edges[e, 0])
            ev2 = np.int32(mst_edges[e, 1])
            ew2 = mst_edges[e, 2]
            is_removed = False
            for rr in range(round_count):
                if round_edge_removed[rr]:
                    if ((eu2 == round_edges_u[rr] and ev2 == round_edges_v[rr])
                            or (eu2 == round_edges_v[rr] and ev2 == round_edges_u[rr])) \
                            and ew2 == round_edges_w[rr]:
                        is_removed = True
                        break
            if not is_removed:
                temp_mst[temp_n, 0] = mst_edges[e, 0]
                temp_mst[temp_n, 1] = mst_edges[e, 1]
                temp_mst[temp_n, 2] = mst_edges[e, 2]
                temp_n += 1

        # Rebuild DSU from surviving edges
        new_parent, new_rank, new_head, new_tail, new_next_node = \
            rebuild_linked_list_dsu(n_points, temp_mst, temp_n)
        for i in range(n_points):
            parent[i] = new_parent[i]
            head[i] = new_head[i]
            tail[i] = new_tail[i]
            next_node[i] = new_next_node[i]
        for e in range(temp_n):
            mst_edges[e, 0] = temp_mst[e, 0]
            mst_edges[e, 1] = temp_mst[e, 1]
            mst_edges[e, 2] = temp_mst[e, 2]
        n_added = temp_n

        # Check sub-components for remaining violations
        sub_a = removed_u
        while parent[sub_a] != sub_a:
            sub_a = parent[sub_a]
        sub_b = removed_v
        while parent[sub_b] != sub_b:
            sub_b = parent[sub_b]
        if component_has_violation(sub_a, parent, head, next_node, cl_indptr, cl_indices):
            worklist[worklist_count] = sub_a
            worklist_count += 1
        if sub_a != sub_b and component_has_violation(sub_b, parent, head, next_node,
                                                       cl_indptr, cl_indices):
            worklist[worklist_count] = sub_b
            worklist_count += 1

    return parent, head, tail, next_node, mst_edges, n_added, removed_edges, n_removed


# ---------------------------------------------------------------------------
# KD-tree path — re-declared functions (CL-extended signatures)
# ---------------------------------------------------------------------------

@numba.njit(locals={"i": numba.types.int64}, cache=NUMBA_CACHE)
def merge_components(disjoint_set, candidate_neighbors, candidate_neighbor_distances,
                     point_components, head, tail, next_node, cl_indptr, cl_indices):
    """Select best cross-component edge per component and merge (KD-tree path).

    Merges are naive; fix_violations (L2) repairs transitive CL violations afterward.
    """
    n_points = candidate_neighbors.shape[0]
    component_edges = {
        np.int64(0): (np.int64(0), np.int64(1), np.float32(0.0)) for i in range(0)
    }
    for i in range(n_points):
        from_component = np.int64(point_components[i])
        if from_component in component_edges and not np.isnan(candidate_neighbor_distances[i]):
            if candidate_neighbor_distances[i] < component_edges[from_component][2]:
                component_edges[from_component] = (
                    np.int64(i), np.int64(candidate_neighbors[i]),
                    candidate_neighbor_distances[i])
        else:
            component_edges[from_component] = (
                np.int64(i), np.int64(candidate_neighbors[i]),
                candidate_neighbor_distances[i])

    result = np.empty((len(component_edges), 3), dtype=np.float64)
    round_edges_u = np.empty(len(component_edges), dtype=np.int32)
    round_edges_v = np.empty(len(component_edges), dtype=np.int32)
    round_edges_w = np.empty(len(component_edges), dtype=np.float64)
    result_idx = np.int32(0)

    for edge in component_edges.values():
        from_component = ds_find(disjoint_set, np.int32(edge[0]))
        to_component = ds_find(disjoint_set, np.int32(edge[1]))
        if from_component == to_component:
            continue
        result[result_idx] = (np.float64(edge[0]), np.float64(edge[1]), np.float64(edge[2]))
        round_edges_u[result_idx] = np.int32(edge[0])
        round_edges_v[result_idx] = np.int32(edge[1])
        round_edges_w[result_idx] = np.float64(edge[2])
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

    return (result[:result_idx], round_edges_u[:result_idx],
            round_edges_v[:result_idx], round_edges_w[:result_idx], result_idx)


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
    """KD-tree NN search with component awareness and L1 CL filtering."""
    is_leaf = tree.is_leaf[node]
    idx_start = tree.idx_start[node]
    idx_end = tree.idx_end[node]

    if dist_lower_bound > heap_p[0]:
        return
    elif (dist_lower_bound > component_nearest_neighbor_dist[0]
          or current_core_distance > component_nearest_neighbor_dist[0]):
        return
    elif node_components[node] == current_component:
        return
    elif is_leaf:
        for i in range(idx_start, idx_end):
            idx = tree.idx_array[i]
            if point_components[idx] == current_component:
                continue
            if core_distances[idx] >= component_nearest_neighbor_dist[0]:
                continue
            if cl_indptr.shape[0] > 1 and points_have_cannot_link(
                    query_point_index, idx, cl_indptr, cl_indices):
                continue
            d = max(rdist(point, tree.data[idx]), current_core_distance, core_distances[idx])
            if d < heap_p[0]:
                heap_p[0] = d
                heap_i[0] = idx
                if d < component_nearest_neighbor_dist[0]:
                    component_nearest_neighbor_dist[0] = d
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
    """Parallel KD-tree query: find each point's cheapest cross-component neighbor (L1)."""
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
    """Block-based reproducible KD-tree query with L1 CL filtering."""
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
    """Bootstrap MST from KNN edges, skipping direct CL pairs (L1)."""
    component_edges = np.full((knn_indices.shape[0], 3), -1, dtype=np.float64)
    for i in numba.prange(knn_indices.shape[0]):
        for j in range(1, knn_indices.shape[1]):
            k = np.int32(knn_indices[i, j])
            if cl_indptr.shape[0] > 1 and points_have_cannot_link(
                    i, k, cl_indptr, cl_indices):
                continue
            if core_distances[i] >= core_distances[k]:
                edge_weight = max(core_distances[i], knn_distances[i, j])
                component_edges[i] = (np.float64(i), np.float64(k), np.float64(edge_weight))
                break

    result = np.empty((len(component_edges), 3), dtype=np.float64)
    result_idx = 0
    for edge in component_edges:
        if edge[0] < 0:
            continue
        from_component = ds_find(disjoint_set, np.int32(edge[0]))
        to_component = ds_find(disjoint_set, np.int32(edge[1]))
        if from_component == to_component:
            continue
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
    """Build MST via parallel Borůvka with CL support (KD-tree path).

    Returns (edges, knn_neighbors, core_distances, n_rounds, n_violations_removed).
    """
    n_points = tree.data.shape[0]
    if cl_indptr is None:
        cl_indptr = np.zeros(n_points + 1, dtype=np.int64)
    if cl_indices is None:
        cl_indices = np.empty(0, dtype=np.int32)

    components_disjoint_set = ds_rank_create(n_points)
    point_components = np.arange(n_points)
    node_components = np.full(tree.idx_start.shape[0], -1)
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)

    # KNN initialisation — compute distances, neighbors, core_distances
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

    initial_edges = initialize_boruvka_from_knn(
        neighbors, distances, core_distances, components_disjoint_set,
        head, tail, next_node, cl_indptr, cl_indices)
    update_component_vectors(
        tree, components_disjoint_set, node_components, point_components)

    # Fix violations from KNN initialisation round (L2)
    all_edges = initial_edges
    round_number = np.int64(0)
    total_violations_removed = np.int64(0)

    if cl_indptr.shape[0] > 1 and all_edges.shape[0] > 0:
        n_init = all_edges.shape[0]
        init_round_u = np.empty(n_init, dtype=np.int32)
        init_round_v = np.empty(n_init, dtype=np.int32)
        init_round_w = np.empty(n_init, dtype=np.float64)
        for _ie in range(n_init):
            init_round_u[_ie] = np.int32(all_edges[_ie, 0])
            init_round_v[_ie] = np.int32(all_edges[_ie, 1])
            init_round_w[_ie] = all_edges[_ie, 2]
        (parent_out, head_out, tail_out, next_out, all_edges, n_init_out,
         _removed, _n_rem_init) = fix_violations(
            n_points, all_edges, np.intp(n_init),
            init_round_u, init_round_v, init_round_w, np.int32(n_init),
            cl_indptr, cl_indices, components_disjoint_set.parent,
            head, tail, next_node)
        for _i in range(n_points):
            components_disjoint_set.parent[_i] = parent_out[_i]
            head[_i] = head_out[_i]
            tail[_i] = tail_out[_i]
            next_node[_i] = next_out[_i]
        all_edges = all_edges[:n_init_out]
        total_violations_removed += np.int64(_n_rem_init)
        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components)

    n_components = len(np.unique(point_components))

    # Main Borůvka loop
    while n_components > 1:
        round_number += np.int64(1)

        # Step 1: find cheapest cross-component edges via KD-tree (L1)
        if reproducible:
            block_size = calculate_block_size(n_components, n_points, n_threads)
            candidate_distances, candidate_indices = boruvka_tree_query_reproducible(
                tree, node_components, point_components, core_distances,
                block_size, cl_indptr, cl_indices)
        else:
            candidate_distances, candidate_indices = boruvka_tree_query(
                tree, node_components, point_components, core_distances,
                cl_indptr, cl_indices)

        # Step 2: merge components
        (new_edges, round_edges_u, round_edges_v, round_edges_w,
         round_count) = merge_components(
            components_disjoint_set, candidate_indices, candidate_distances,
            point_components, head, tail, next_node, cl_indptr, cl_indices)
        if len(new_edges) == 0:
            break

        # Step 3: fix violations from this round (L2)
        n_edges_before = all_edges.shape[0]
        if cl_indptr.shape[0] > 1 and round_count > 0:
            n_added = np.intp(n_edges_before + new_edges.shape[0])
            combined = np.empty((n_added, 3), dtype=np.float64)
            for e in range(n_edges_before):
                combined[e, 0] = all_edges[e, 0]
                combined[e, 1] = all_edges[e, 1]
                combined[e, 2] = all_edges[e, 2]
            for e in range(new_edges.shape[0]):
                combined[n_edges_before + e, 0] = new_edges[e, 0]
                combined[n_edges_before + e, 1] = new_edges[e, 1]
                combined[n_edges_before + e, 2] = new_edges[e, 2]
            (parent_out, head_out, tail_out, next_out, combined, n_added_out,
             _removed_edges, _n_removed_round) = fix_violations(
                n_points, combined, n_added,
                round_edges_u, round_edges_v, round_edges_w, round_count,
                cl_indptr, cl_indices, components_disjoint_set.parent,
                head, tail, next_node)
            total_violations_removed += np.int64(_n_removed_round)
            for i in range(n_points):
                head[i] = head_out[i]
                tail[i] = tail_out[i]
                next_node[i] = next_out[i]
                components_disjoint_set.parent[i] = parent_out[i]
            all_edges = combined[:n_added_out]
            if n_added_out <= n_edges_before:
                update_component_vectors(
                    tree, components_disjoint_set, node_components, point_components)
                break
        else:
            all_edges = np.vstack((all_edges, new_edges))

        # Step 4: update component vectors
        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components)
        n_components = len(np.unique(point_components))

    all_edges[:, 2] = np.sqrt(all_edges.T[2])
    return (all_edges, neighbors[:, 1:], np.sqrt(core_distances),
            round_number, total_violations_removed)


# ---------------------------------------------------------------------------
# Precomputed path — constrained graph-Borůvka functions
# ---------------------------------------------------------------------------

@numba.njit(locals={"parent": numba.types.int32}, cache=NUMBA_CACHE)
def select_components_constrained(distances, indices, indptr, point_components,
                                  cl_indptr, cl_indices):
    """Find cheapest cross-component edge per component, skipping CL pairs (L1)."""
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
            if point_components[neighbor] == from_component:
                continue
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
                                 head, tail, next_node):
    """Merge components from cheapest edges, maintaining linked-list DSU."""
    result = np.empty((len(component_edges), 3), dtype=np.float64)
    round_edges_u = np.empty(len(component_edges), dtype=np.int32)
    round_edges_v = np.empty(len(component_edges), dtype=np.int32)
    round_edges_w = np.empty(len(component_edges), dtype=np.float64)
    result_idx = np.int32(0)
    for edge in component_edges.values():
        from_component = ds_find(disjoint_set, edge[0])
        to_component = ds_find(disjoint_set, edge[1])
        if from_component == to_component:
            continue
        result[result_idx] = (np.float64(edge[0]), np.float64(edge[1]), np.float64(edge[2]))
        round_edges_u[result_idx] = np.int32(edge[0])
        round_edges_v[result_idx] = np.int32(edge[1])
        round_edges_w[result_idx] = np.float64(edge[2])
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
    return (result[:result_idx], round_edges_u[:result_idx],
            round_edges_v[:result_idx], round_edges_w[:result_idx], result_idx)


@numba.njit(cache=NUMBA_CACHE)
def minimum_spanning_tree_constrained(graph, cl_indptr, cl_indices, overwrite=False):
    """Constrained graph-Borůvka MST on a CoreGraph with L1+L2 CL enforcement."""
    distances = graph.weights
    indices = graph.indices
    indptr = graph.indptr
    n_points = len(indptr) - 1
    if not overwrite:
        indices = indices.copy()
        distances = distances.copy()

    disjoint_set = ds_rank_create(n_points)
    point_components = np.arange(n_points, dtype=np.int32)
    n_components = n_points
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)
    edges_list = [np.empty((0, 3), dtype=np.float64) for _ in range(0)]
    total_edges = np.int32(0)

    while n_components > 1:
        comp_edges = select_components_constrained(
            distances, indices, indptr, point_components, cl_indptr, cl_indices)
        (new_edges, round_edges_u, round_edges_v, round_edges_w,
         round_count) = merge_components_constrained(
            disjoint_set, comp_edges, head, tail, next_node)
        if new_edges.shape[0] == 0:
            break
        edges_list.append(new_edges)

        # L2: fix violations from this round
        if cl_indptr.shape[0] > 1 and round_count > 0:
            n_prev = total_edges
            n_added_total = np.intp(n_prev + new_edges.shape[0])
            combined = np.empty((n_added_total, 3), dtype=np.float64)
            comb_idx = np.int32(0)
            for el in edges_list:
                for e in range(el.shape[0]):
                    combined[comb_idx, 0] = el[e, 0]
                    combined[comb_idx, 1] = el[e, 1]
                    combined[comb_idx, 2] = el[e, 2]
                    comb_idx += 1
            (parent_out, head_out, tail_out, next_out, combined, n_added_out,
             _removed, _n_removed) = fix_violations(
                n_points, combined, n_added_total,
                round_edges_u, round_edges_v, round_edges_w, round_count,
                cl_indptr, cl_indices, disjoint_set.parent, head, tail, next_node)
            for i in range(n_points):
                disjoint_set.parent[i] = parent_out[i]
                head[i] = head_out[i]
                tail[i] = tail_out[i]
                next_node[i] = next_out[i]
            edges_list = [combined[:n_added_out]]
            total_edges = n_added_out
            if n_added_out <= n_prev:
                update_point_components(disjoint_set, point_components)
                update_graph_components(distances, indices, indptr, point_components)
                break
        else:
            total_edges += new_edges.shape[0]

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


# ---------------------------------------------------------------------------
# Entry-point utilities
# ---------------------------------------------------------------------------

def _pad_spanning_forest(edges, n_points):
    """Pad a spanning forest to n-1 edges with inf-weight bridges."""
    n_edges = edges.shape[0]
    needed = n_points - 1
    if n_edges >= needed:
        return edges
    parent = np.arange(n_points)

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n_edges):
        _union(int(edges[i, 0]), int(edges[i, 1]))
    roots = sorted({_find(i) for i in range(n_points)})
    bridges = [[float(roots[0]), float(roots[k]), np.inf] for k in range(1, len(roots))]
    if bridges:
        edges = np.vstack([edges, np.array(bridges, dtype=edges.dtype)])
    return edges


@numba.njit(cache=NUMBA_CACHE)
def _has_cl_violation_csr(labels, cl_indptr, cl_indices, noise_label):
    """Return True if any CL pair (i,j) has labels[i]==labels[j]!=noise_label."""
    n = labels.shape[0]
    for i in range(n):
        li = labels[i]
        if li == noise_label:
            continue
        for k in range(cl_indptr[i], cl_indptr[i + 1]):
            j = cl_indices[k]
            if j != i and labels[j] == li:
                return True
    return False


def _pair_cannot_link_csr(i, j, cl_indptr, cl_indices):
    """Return True if (i,j) is a CL pair in the CSR graph (either direction)."""
    for k in range(int(cl_indptr[i]), int(cl_indptr[i + 1])):
        if int(cl_indices[k]) == j:
            return True
    for k in range(int(cl_indptr[j]), int(cl_indptr[j + 1])):
        if int(cl_indices[k]) == i:
            return True
    return False


def _split_clusters_greedy(labels, cl_indptr, cl_indices, noise_label=-1):
    """Greedy graph-coloring split for clusters with CL violations.

    For each cluster, builds a conflict adjacency among its members,
    sorts by descending conflict degree, and greedy-colors. Points with
    color > 0 are assigned fresh labels.
    """
    labels_out = labels.astype(np.int64, copy=True)
    next_label = int(labels_out.max()) + 1

    for label_id in sorted(np.unique(labels_out).tolist()):
        if label_id == noise_label:
            continue
        idx_points = np.flatnonzero(labels_out == label_id)
        if idx_points.size <= 1:
            continue

        # Build conflict adjacency for this cluster
        idx_list = idx_points.tolist()
        adjacency = {i: [] for i in idx_list}
        for a_i in range(len(idx_list)):
            pi = idx_list[a_i]
            for a_j in range(a_i + 1, len(idx_list)):
                pj = idx_list[a_j]
                if _pair_cannot_link_csr(pi, pj, cl_indptr, cl_indices):
                    adjacency[pi].append(pj)
                    adjacency[pj].append(pi)

        if not any(len(v) > 0 for v in adjacency.values()):
            continue

        # Sort by descending conflict degree, then by index
        order = sorted(idx_list, key=lambda n: (-len(adjacency[n]), n))

        # Greedy graph-coloring
        color_of = {}
        for node in order:
            used = {color_of[nbr] for nbr in adjacency[node] if nbr in color_of}
            c = 0
            while c in used:
                c += 1
            color_of[node] = c

        colors = sorted(set(color_of.values()))
        if len(colors) <= 1:
            continue

        for color in colors:
            if color == 0:
                continue
            nodes = [n for n, c in color_of.items() if c == color]
            labels_out[np.asarray(nodes, dtype=np.int64)] = next_label
            next_label += 1

    return labels_out


def _apply_posthoc_cleanup(labels, n_points, cl_indptr, cl_indices):
    """L3 post-hoc CL label splitting via greedy graph-coloring."""
    labels_in = np.asarray(labels, dtype=np.int64)
    cl_indptr_i64 = np.asarray(cl_indptr, dtype=np.int64)
    cl_indices_i64 = np.asarray(cl_indices, dtype=np.int64)

    if not _has_cl_violation_csr(labels_in, cl_indptr_i64, cl_indices_i64, -1):
        return labels_in

    return _split_clusters_greedy(labels_in, cl_indptr, cl_indices, noise_label=-1)


def constrained_hdbscan_from_boruvka(
        tree_or_X, n_threads, cl_indptr, cl_indices, *,
        min_samples=10, min_cluster_size=10, posthoc_cleanup=True,
        sample_weights=None, reproducible=False, return_metadata=False,
        metric="euclidean"):
    """Run constrained Borůvka MST → HDBSCAN with L1+L2+L3 CL enforcement.

    metric='euclidean': tree_or_X is a NumbaKDTree → parallel_boruvka.
    metric='precomputed': tree_or_X is a scipy sparse CSR → graph-Borůvka.
    Returns (labels, probs, edges) or (labels, probs, edges, metadata).
    """
    if metric == "euclidean":
        tree = tree_or_X
        n_points = tree.data.shape[0]
        edges, nbrs, core_dists, n_rounds, n_viols = parallel_boruvka(
            tree, n_threads, min_samples=min_samples,
            sample_weights=sample_weights, reproducible=reproducible,
            cl_indptr=cl_indptr, cl_indices=cl_indices)
    elif metric == "precomputed":
        X = tree_or_X
        validate_precomputed_sparse_graph(X)
        n_points = X.shape[0]
        undirected_edges = extract_undirected_min_edges(X)
        adjacency = build_adjacency_lists(n_points, undirected_edges)
        neighbors, core_dists_raw = compute_sparse_core_distances(adjacency, min_samples)
        mrd_edges = apply_mutual_reachability(undirected_edges, core_dists_raw)
        core_graph = to_core_graph_arrays(n_points, mrd_edges)
        n_comp, comp_labels, edges = minimum_spanning_tree_constrained(
            core_graph, cl_indptr, cl_indices)
        n_rounds, n_viols = 0, 0
    else:
        raise ValueError(f"Unsupported metric: {metric!r}. Expected 'euclidean' or 'precomputed'.")

    # Pad spanning forest → full spanning tree
    edges = _pad_spanning_forest(edges, n_points)

    # Replace inf with large finite penalty for cluster extraction
    finite_vals = edges[:, 2][np.isfinite(edges[:, 2])]
    if finite_vals.size > 0:
        penalty = float(np.percentile(finite_vals, 99.9) * 1e6 + 1.0)
    else:
        penalty = 1e6
    bad = ~np.isfinite(edges[:, 2])
    if np.any(bad):
        edges = edges.copy()
        edges[bad, 2] = penalty

    labels, probs, *_ = clusters_from_spanning_tree(
        edges, min_cluster_size=min_cluster_size)

    if posthoc_cleanup and cl_indptr.shape[0] > 1:
        labels = _apply_posthoc_cleanup(labels, n_points, cl_indptr, cl_indices)

    if return_metadata:
        return labels, probs, edges, {
            "n_rounds": int(n_rounds),
            "violations_detected_mst": int(n_viols)}
    return labels, probs, edges
