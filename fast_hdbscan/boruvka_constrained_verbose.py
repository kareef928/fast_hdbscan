"""
Constrained Parallel Borůvka MST (KD-tree based)
=================================================

KD-tree-accelerated Borůvka's algorithm for Minimum Spanning Trees /
Forests with optional cannot-link constraints.  This is an extension of
the original ``boruvka.py`` that threads CSR-format constraints through
every stage of the algorithm:

 * KNN initialisation  – direct CL pairs are skipped.
 * KD-tree leaf search – direct CL pairs are skipped.
 * Component merge     – full component-level CL check before union.
 * Post-merge fix      – detect & remove heaviest violating edge.

When no constraints are supplied the code path is identical to the
unconstrained version (empty CSR → all checks are no-ops).
"""
import numba
import numpy as np

from .disjoint_set import (
    ds_rank_create,
    ds_find,
    ds_find_readonly,
    ds_union_by_rank,
)
from .numba_kdtree import (
    parallel_tree_query,
    rdist,
    point_to_node_lower_bound_rdist,
    NumbaKDTree,
)
from .variables import NUMBA_CACHE


# =====================================================================
# Constraint helpers (all operate on raw arrays, not DisjointSet tuples)
# =====================================================================


@numba.njit(cache=NUMBA_CACHE, inline="always")
def points_have_cannot_link(point_a, point_b, cl_indptr, cl_indices):
    """Return True if *point_a* and *point_b* are directly connected by a
    cannot-link constraint.  O(degree-of-point_a) scan of one CSR row."""
    for k in range(cl_indptr[point_a], cl_indptr[point_a + 1]):
        if cl_indices[k] == point_b:
            return True
    return False


@numba.njit(cache=NUMBA_CACHE)
def check_merge_violates(
    root_small,
    root_large,
    parent,
    head,
    next_node,
    cl_indptr,
    cl_indices,
):
    """Check whether merging *root_small* into *root_large* would create a
    cannot-link violation.  Walks the linked-list of the smaller component
    and, for every CL partner of every member, checks whether that partner
    already belongs to the larger component (via ``ds_find_readonly`` so
    that parallel callers don't race on path compression)."""
    node = head[root_small]
    while node != -1:
        start = np.int64(cl_indptr[node])
        end = np.int64(cl_indptr[node + 1])
        for k in range(start, end):
            partner = np.int32(cl_indices[k])
            if ds_find_readonly(parent, partner) == root_large:
                return True
        node = next_node[node]
    return False


@numba.njit(cache=NUMBA_CACHE)
def component_has_violation(
    root,
    parent,
    head,
    next_node,
    cl_indptr,
    cl_indices,
):
    """Return True if *root*'s component contains an internal CL violation
    (two members that share a cannot-link edge)."""
    node = head[root]
    while node != -1:
        start = np.int64(cl_indptr[node])
        end = np.int64(cl_indptr[node + 1])
        for k in range(start, end):
            partner = np.int32(cl_indices[k])
            if ds_find_readonly(parent, partner) == root:
                return True
        node = next_node[node]
    return False


# =====================================================================
# Linked-list DSU helpers
# =====================================================================


@numba.njit(cache=NUMBA_CACHE, inline="always")
def linked_list_merge(head, tail, next_node, winner, loser):
    """Append *loser*'s member list onto *winner*'s.
    After this call every member of *loser* is reachable via *winner*'s
    ``head → next_node → … → tail`` chain."""
    next_node[tail[winner]] = head[loser]
    tail[winner] = tail[loser]


@numba.njit(cache=NUMBA_CACHE)
def rebuild_linked_list_dsu(n_points, mst_edges, n_edges):
    """Re-create the full DSU (parent/rank + head/tail/next_node) from
    scratch by replaying *n_edges* rows of *mst_edges*.  Used by
    ``fix_violations`` after removing an edge."""
    parent = np.arange(n_points, dtype=np.int32)
    rank = np.zeros(n_points, dtype=np.int32)
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)

    for e in range(n_edges):
        u = np.int32(mst_edges[e, 0])
        v = np.int32(mst_edges[e, 1])
        # find roots (with path compression – single-threaded here)
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
        # union by rank
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
        # splice linked lists
        next_node[tail[ru]] = head[rv]
        tail[ru] = tail[rv]

    return parent, rank, head, tail, next_node


# =====================================================================
# Original helpers (unchanged)
# =====================================================================


@numba.njit(locals={"parent": numba.types.int32}, cache=NUMBA_CACHE)
def select_components(candidate_distances, candidate_neighbors, point_components):
    component_edges = {
        np.int64(0): (np.int32(0), np.int32(1), np.float32(0.0)) for i in range(0)
    }

    # Find the best edges from each component
    for parent, (distance, neighbor, from_component) in enumerate(
        zip(candidate_distances, candidate_neighbors, point_components)
    ):
        if from_component in component_edges:
            if distance < component_edges[np.int64(from_component)][2]:
                component_edges[np.int64(from_component)] = (
                    parent,
                    neighbor,
                    distance,
                )
        else:
            component_edges[np.int64(from_component)] = (
                parent,
                neighbor,
                distance,
            )

    return component_edges


# =====================================================================
# merge_components – now with constraint checking + linked-list DSU
# =====================================================================


@numba.njit(locals={"i": numba.types.int64}, cache=NUMBA_CACHE)
def merge_components(
    disjoint_set,
    candidate_neighbors,
    candidate_neighbor_distances,
    point_components,
    # ---- constraint / linked-list parameters ----
    head,
    tail,
    next_node,
    cl_indptr,
    cl_indices,
):
    """Select the best cross-component edge per component and merge.

    Before every union the function checks ``check_merge_violates``.
    After a successful union the linked lists are spliced so that
    ``head[winner]`` walks the full merged component.

    Returns
    -------
    result : ndarray, shape (n_merged, 3)
        Rows ``[source, target, weight]`` for edges actually added.
    round_edges_u, round_edges_v, round_edges_w : ndarray
        Same edges in separate arrays (needed by ``fix_violations``).
    round_count : int
        Number of edges in the round arrays.
    """
    n_points = candidate_neighbors.shape[0]

    component_edges = {
        np.int64(0): (np.int64(0), np.int64(1), np.float32(0.0)) for i in range(0)
    }

    # 1. Find the best candidate edge per component
    for i in range(n_points):
        from_component = np.int64(point_components[i])
        if from_component in component_edges and not np.isnan(
            candidate_neighbor_distances[i]
        ):
            if candidate_neighbor_distances[i] < component_edges[from_component][2]:
                component_edges[from_component] = (
                    np.int64(i),
                    np.int64(candidate_neighbors[i]),
                    candidate_neighbor_distances[i],
                )
        else:
            component_edges[from_component] = (
                np.int64(i),
                np.int64(candidate_neighbors[i]),
                candidate_neighbor_distances[i],
            )

    result = np.empty((len(component_edges), 3), dtype=np.float64)
    round_edges_u = np.empty(len(component_edges), dtype=np.int32)
    round_edges_v = np.empty(len(component_edges), dtype=np.int32)
    round_edges_w = np.empty(len(component_edges), dtype=np.float64)
    result_idx = np.int32(0)

    # 2. Merge naively – fix_violations will repair CL violations afterward
    for edge in component_edges.values():
        from_component = ds_find(disjoint_set, np.int32(edge[0]))
        to_component = ds_find(disjoint_set, np.int32(edge[1]))
        if from_component == to_component:
            continue

        # --- record edge ---
        result[result_idx] = (
            np.float64(edge[0]),
            np.float64(edge[1]),
            np.float64(edge[2]),
        )
        round_edges_u[result_idx] = np.int32(edge[0])
        round_edges_v[result_idx] = np.int32(edge[1])
        round_edges_w[result_idx] = np.float64(edge[2])
        result_idx += 1

        # --- union ---
        # Re-find after potential prior merges in this loop
        from_component = ds_find(disjoint_set, np.int32(edge[0]))
        to_component = ds_find(disjoint_set, np.int32(edge[1]))
        if from_component == to_component:
            # already merged by an earlier iteration
            result_idx -= 1
            continue

        ds_union_by_rank(disjoint_set, from_component, to_component)

        # Determine new root after union (it's whichever is the root now)
        new_root = ds_find(disjoint_set, from_component)
        loser = to_component if new_root == from_component else from_component
        linked_list_merge(head, tail, next_node, new_root, loser)

    return (
        result[:result_idx],
        round_edges_u[:result_idx],
        round_edges_v[:result_idx],
        round_edges_w[:result_idx],
        result_idx,
    )


# =====================================================================
# update_component_vectors – unchanged from boruvka.py
# =====================================================================


@numba.njit(
    locals={
        "i": numba.types.int32,
        "j": numba.types.int32,
        "idx": numba.types.int32,
        "left": numba.types.int32,
        "right": numba.types.int32,
        "candidate_component": numba.types.int32,
    },
    parallel=True,
    cache=NUMBA_CACHE,
    fastmath=True,
)
def update_component_vectors(tree, disjoint_set, node_components, point_components):
    for i in numba.prange(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))

    for i in range(tree.idx_start.shape[0] - 1, -1, -1):
        is_leaf = tree.is_leaf[i]
        idx_start = tree.idx_start[i]
        idx_end = tree.idx_end[i]

        if is_leaf:
            candidate_component = point_components[tree.idx_array[idx_start]]
            for j in range(idx_start + 1, idx_end):
                idx = tree.idx_array[j]
                if point_components[idx] != candidate_component:
                    break
            else:
                node_components[i] = candidate_component
        else:
            left = 2 * i + 1
            right = left + 1
            if node_components[left] == node_components[right]:
                node_components[i] = node_components[left]


# =====================================================================
# component_aware_query_recursion – with CL filtering at leaf nodes
# =====================================================================


@numba.njit(
    locals={
        "i": numba.types.int32,
        "idx": numba.types.int32,
        "left": numba.types.int32,
        "right": numba.types.int32,
        "d": numba.types.float32,
        "dist_lower_bound_left": numba.types.float32,
        "dist_lower_bound_right": numba.types.float32,
    },
    cache=NUMBA_CACHE,
    fastmath=True,
)
def component_aware_query_recursion(
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
    # ---- constraint parameters ----
    query_point_index,
    cl_indptr,
    cl_indices,
):
    is_leaf = tree.is_leaf[node]
    idx_start = tree.idx_start[node]
    idx_end = tree.idx_end[node]

    # ----------------------------------------------------------------
    # Case 1a: query point is outside node radius – trim
    if dist_lower_bound > heap_p[0]:
        return

    # ----------------------------------------------------------------
    # Case 1b: can't improve on best distance for this component – trim
    elif (
        dist_lower_bound > component_nearest_neighbor_dist[0]
        or current_core_distance > component_nearest_neighbor_dist[0]
    ):
        return

    # ----------------------------------------------------------------
    # Case 1c: node contains only points in same component – trim
    elif node_components[node] == current_component:
        return

    # ----------------------------------------------------------------
    # Case 2: leaf node – update set of nearby points
    elif is_leaf:
        for i in range(idx_start, idx_end):
            idx = tree.idx_array[i]
            if point_components[idx] == current_component:
                continue
            if core_distances[idx] >= component_nearest_neighbor_dist[0]:
                continue
            # --- direct CL filter ---
            if cl_indptr.shape[0] > 1 and points_have_cannot_link(
                query_point_index, idx, cl_indptr, cl_indices
            ):
                continue
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

    # ----------------------------------------------------------------
    # Case 3: internal node – recurse into children closest-first
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
            component_aware_query_recursion(
                tree, left, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_left, component_nearest_neighbor_dist,
                query_point_index, cl_indptr, cl_indices,
            )
            component_aware_query_recursion(
                tree, right, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_right, component_nearest_neighbor_dist,
                query_point_index, cl_indptr, cl_indices,
            )
        else:
            component_aware_query_recursion(
                tree, right, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_right, component_nearest_neighbor_dist,
                query_point_index, cl_indptr, cl_indices,
            )
            component_aware_query_recursion(
                tree, left, point, heap_p, heap_i,
                current_core_distance, core_distances,
                current_component, node_components, point_components,
                dist_lower_bound_left, component_nearest_neighbor_dist,
                query_point_index, cl_indptr, cl_indices,
            )

    return


# =====================================================================
# boruvka_tree_query – forwards CL arrays to recursion
# =====================================================================


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
def boruvka_tree_query(
    tree, node_components, point_components, core_distances,
    cl_indptr, cl_indices,
):
    candidate_distances = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    candidate_indices = np.full(tree.data.shape[0], -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(
        tree.data.shape[0], np.inf, dtype=np.float32
    )

    data = tree.data.astype(np.float32)

    for i in numba.prange(tree.data.shape[0]):
        distance_lower_bound = point_to_node_lower_bound_rdist(
            tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i]
        )
        heap_p, heap_i = candidate_distances[i : i + 1], candidate_indices[i : i + 1]
        component_aware_query_recursion(
            tree, 0, data[i],
            heap_p, heap_i,
            core_distances[i], core_distances,
            point_components[i],
            node_components, point_components,
            distance_lower_bound,
            component_nearest_neighbor_dist[
                point_components[i] : point_components[i] + 1
            ],
            i, cl_indptr, cl_indices,
        )

    return candidate_distances, candidate_indices


# =====================================================================
# Block-size calculation (unchanged)
# =====================================================================


@numba.njit(inline="always", cache=NUMBA_CACHE)
def calculate_block_size(n_components, n_points, num_threads):
    """Calculate adaptive block size based on component sizes."""
    if n_components == 0:
        points_per_component = n_points
    else:
        points_per_component = n_points / n_components

    if points_per_component < 10:
        block_size = num_threads * 512
    elif points_per_component < 100:
        block_size = num_threads * 128
    elif points_per_component < 1000:
        block_size = num_threads * 32
    else:
        block_size = num_threads * 8

    block_size = max(num_threads, min(block_size, n_points // 4 + 1))
    return int(block_size)


# =====================================================================
# update_component_bounds_from_block (unchanged)
# =====================================================================


@numba.njit(
    [
        "void(float32[:], float32[:], int32[:], int32, int32)",
        "void(float64[:], float64[:], int64[:], int64, int64)",
    ],
    locals={
        "i": numba.types.int32,
        "component": numba.types.int32,
        "block_bound": numba.types.float32,
    },
    cache=NUMBA_CACHE,
    fastmath=True,
    inline="always",
)
def update_component_bounds_from_block(
    component_nearest_neighbor_dist,
    block_component_bounds,
    point_components,
    block_start,
    block_end,
):
    """Update global component bounds from block results."""
    for i in range(block_start, block_end):
        component = point_components[i]
        block_bound = block_component_bounds[i - block_start]
        if block_bound < component_nearest_neighbor_dist[component]:
            component_nearest_neighbor_dist[component] = block_bound


# =====================================================================
# boruvka_tree_query_reproducible – forwards CL arrays
# =====================================================================


@numba.njit(
    locals={
        "block_start": numba.types.int32,
        "block_end": numba.types.int32,
        "block_size_actual": numba.types.int32,
        "i": numba.types.int32,
        "distance_lower_bound": numba.types.float32,
        "current_component": numba.types.int32,
    },
    parallel=True,
    cache=NUMBA_CACHE,
    fastmath=True,
)
def boruvka_tree_query_reproducible(
    tree, node_components, point_components, core_distances, block_size,
    cl_indptr, cl_indices,
):
    """Reproducible version using block-based processing to avoid race conditions."""
    candidate_distances = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    candidate_indices = np.full(tree.data.shape[0], -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(
        tree.data.shape[0], np.inf, dtype=np.float32
    )

    data = tree.data.astype(np.float32)

    max_block_component_bounds = np.full(block_size, np.inf, dtype=np.float32)

    for block_start in range(0, tree.data.shape[0], block_size):
        block_end = min(block_start + block_size, tree.data.shape[0])
        block_size_actual = block_end - block_start

        max_block_component_bounds[:block_size_actual] = np.inf

        for i in numba.prange(block_start, block_end):
            distance_lower_bound = point_to_node_lower_bound_rdist(
                tree.node_bounds[0, 0], tree.node_bounds[1, 0], tree.data[i]
            )
            heap_p, heap_i = (
                candidate_distances[i : i + 1],
                candidate_indices[i : i + 1],
            )

            current_component = point_components[i]
            local_component_bound = component_nearest_neighbor_dist[
                current_component : current_component + 1
            ]

            component_aware_query_recursion(
                tree, 0, data[i],
                heap_p, heap_i,
                core_distances[i], core_distances,
                point_components[i],
                node_components, point_components,
                distance_lower_bound,
                local_component_bound,
                i, cl_indptr, cl_indices,
            )

            max_block_component_bounds[i - block_start] = local_component_bound[0]

        update_component_bounds_from_block(
            component_nearest_neighbor_dist,
            max_block_component_bounds,
            point_components,
            block_start,
            block_end,
        )

    return candidate_distances, candidate_indices


# =====================================================================
# fix_violations – post-merge constraint repair
# =====================================================================


@numba.njit(cache=NUMBA_CACHE)
def fix_violations(
    n_points,
    mst_edges,
    n_added,
    round_edges_u,
    round_edges_v,
    round_edges_w,
    round_count,
    cl_indptr,
    cl_indices,
    parent,
    head,
    tail,
    next_node,
):
    """Detect and repair constraint violations created during the last
    merge round.

    Algorithm
    ---------
    1.  Scan all component roots for internal CL violations.
    2.  For each violating component find the *heaviest* edge from
        ``round_edges`` that lies inside it and remove that edge.
    3.  Rebuild the DSU from the surviving MST edges.
    4.  Repeat until no violations remain.

    Returns
    -------
    parent, head, tail, next_node : updated DSU arrays
    mst_edges : updated (some rows may have been shifted)
    n_added   : new count of MST edges
    removed_edges : ndarray (n_removed, 3)
    n_removed     : int
    """
    removed_edges = np.empty((round_count, 3), dtype=np.float64)
    n_removed = np.int32(0)
    round_edge_removed = np.zeros(round_count, dtype=numba.boolean)

    # Build initial worklist
    worklist = np.empty(n_points, dtype=np.int32)
    worklist_count = np.int32(0)
    for r in range(n_points):
        if parent[r] == r:
            if component_has_violation(
                r, parent, head, next_node, cl_indptr, cl_indices
            ):
                worklist[worklist_count] = r
                worklist_count += 1

    max_iters = round_count + 1  # at most we can remove round_count edges
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

        if not component_has_violation(
            current_root, parent, head, next_node, cl_indptr, cl_indices
        ):
            continue

        # Find the heaviest round-edge inside this component
        best_idx = np.int32(-1)
        best_w = -np.inf
        for ri in range(round_count):
            if round_edge_removed[ri]:
                continue
            eu = round_edges_u[ri]
            ev = round_edges_v[ri]
            ew = round_edges_w[ri]
            # check both endpoints belong to current_root
            eru = eu
            while parent[eru] != eru:
                eru = parent[eru]
            erv = ev
            while parent[erv] != erv:
                erv = parent[erv]
            if eru == current_root and erv == current_root:
                if ew > best_w:
                    best_w = ew
                    best_idx = ri

        if best_idx == -1:
            continue  # no removable edge found

        round_edge_removed[best_idx] = True
        removed_u = round_edges_u[best_idx]
        removed_v = round_edges_v[best_idx]
        removed_w = round_edges_w[best_idx]

        # Record removed edge
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
                    ru2 = round_edges_u[rr]
                    rv2 = round_edges_v[rr]
                    rw2 = round_edges_w[rr]
                    if (
                        (eu2 == ru2 and ev2 == rv2)
                        or (eu2 == rv2 and ev2 == ru2)
                    ) and ew2 == rw2:
                        is_removed = True
                        break
            if not is_removed:
                temp_mst[temp_n, 0] = mst_edges[e, 0]
                temp_mst[temp_n, 1] = mst_edges[e, 1]
                temp_mst[temp_n, 2] = mst_edges[e, 2]
                temp_n += 1

        # Rebuild DSU from surviving edges
        (
            new_parent,
            new_rank,
            new_head,
            new_tail,
            new_next_node,
        ) = rebuild_linked_list_dsu(n_points, temp_mst, temp_n)

        # Copy back
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
        sub_a_root = removed_u
        while parent[sub_a_root] != sub_a_root:
            sub_a_root = parent[sub_a_root]
        sub_b_root = removed_v
        while parent[sub_b_root] != sub_b_root:
            sub_b_root = parent[sub_b_root]

        if component_has_violation(
            sub_a_root, parent, head, next_node, cl_indptr, cl_indices
        ):
            worklist[worklist_count] = sub_a_root
            worklist_count += 1
        if sub_a_root != sub_b_root and component_has_violation(
            sub_b_root, parent, head, next_node, cl_indptr, cl_indices
        ):
            worklist[worklist_count] = sub_b_root
            worklist_count += 1

    return parent, head, tail, next_node, mst_edges, n_added, removed_edges, n_removed


# =====================================================================
# initialize_boruvka_from_knn – with CL filtering + linked-list DSU
# =====================================================================


@numba.njit(
    locals={
        "i": numba.types.int32,
        "j": numba.types.int32,
        "k": numba.types.int32,
        "result_idx": numba.types.int32,
        "from_component": numba.types.int32,
        "to_component": numba.types.int32,
    },
    parallel=True,
    cache=NUMBA_CACHE,
)
def initialize_boruvka_from_knn(
    knn_indices,
    knn_distances,
    core_distances,
    disjoint_set,
    # ---- constraint / linked-list parameters ----
    head,
    tail,
    next_node,
    cl_indptr,
    cl_indices,
):
    component_edges = np.full((knn_indices.shape[0], 3), -1, dtype=np.float64)

    for i in numba.prange(knn_indices.shape[0]):
        for j in range(1, knn_indices.shape[1]):
            k = np.int32(knn_indices[i, j])
            # --- direct CL filter ---
            if cl_indptr.shape[0] > 1 and points_have_cannot_link(
                i, k, cl_indptr, cl_indices
            ):
                continue
            if core_distances[i] >= core_distances[k]:
                edge_weight = max(core_distances[i], knn_distances[i, j])
                component_edges[i] = (
                    np.float64(i),
                    np.float64(k),
                    np.float64(edge_weight),
                )
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

        result[result_idx] = (
            np.float64(edge[0]),
            np.float64(edge[1]),
            np.float64(edge[2]),
        )
        result_idx += 1

        # Re-find in case prior iterations changed roots
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


# =====================================================================
# sample_weight_core_distance (unchanged)
# =====================================================================


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def sample_weight_core_distance(distances, neighbors, sample_weights, min_samples):
    core_distances = np.zeros(distances.shape[0], dtype=np.float32)
    for i in numba.prange(distances.shape[0]):
        total_weight = 0.0
        j = 0
        while total_weight < min_samples and j < neighbors.shape[1]:
            total_weight += sample_weights[neighbors[i, j]]
            j += 1
        core_distances[i] = distances[i, j - 1]
    return core_distances


# =====================================================================
# parallel_boruvka – top-level orchestrator with constraint support
# =====================================================================


@numba.njit(cache=NUMBA_CACHE)
def parallel_boruvka(
    tree,
    n_threads,
    min_samples=10,
    sample_weights=None,
    reproducible=False,
    cl_indptr=None,
    cl_indices=None,
):
    """Build an MST via parallel Borůvka, optionally respecting cannot-link
    constraints given in CSR format (``cl_indptr``, ``cl_indices``).

    Parameters
    ----------
    tree : NumbaKDTree
    n_threads : int
    min_samples : int
    sample_weights : ndarray or None
    reproducible : bool
    cl_indptr : ndarray (n_points+1,) int64 or None
        CSR row-pointer array for cannot-link constraints.
    cl_indices : ndarray int32 or None
        CSR column-index array for cannot-link constraints.

    Returns
    -------
    all_edges : ndarray (n_edges, 3)  – MST edges [u, v, weight]
    neighbors : ndarray               – kNN neighbor indices (minus self)
    core_distances : ndarray           – per-point core distances (sqrt'd)
    """
    n_points = tree.data.shape[0]

    # ---- default empty constraints ----
    if cl_indptr is None:
        cl_indptr = np.zeros(n_points + 1, dtype=np.int64)
    if cl_indices is None:
        cl_indices = np.empty(0, dtype=np.int32)

    # ---- DSU ----
    components_disjoint_set = ds_rank_create(n_points)
    point_components = np.arange(n_points)
    node_components = np.full(tree.idx_start.shape[0], -1)

    # ---- linked-list DSU for constraint checking ----
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)

    # ---- KNN initialisation ----
    if sample_weights is not None:
        mean_sample_weight = np.mean(sample_weights)
        expected_neighbors = min_samples / mean_sample_weight
        distances, neighbors = parallel_tree_query(
            tree, tree.data, k=int(2 * expected_neighbors)
        )
        core_distances = sample_weight_core_distance(
            distances, neighbors, sample_weights, min_samples
        )
        initial_edges = initialize_boruvka_from_knn(
            neighbors, distances, core_distances, components_disjoint_set,
            head, tail, next_node, cl_indptr, cl_indices,
        )
        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components
        )
    else:
        if min_samples > 1:
            distances, neighbors = parallel_tree_query(
                tree, tree.data, k=min_samples + 1, output_rdist=True
            )
            core_distances = distances.T[-1]
            initial_edges = initialize_boruvka_from_knn(
                neighbors, distances, core_distances, components_disjoint_set,
                head, tail, next_node, cl_indptr, cl_indices,
            )
            update_component_vectors(
                tree, components_disjoint_set, node_components, point_components
            )
        else:
            core_distances = np.zeros(n_points, dtype=np.float32)
            distances, neighbors = parallel_tree_query(
                tree, tree.data, k=2, output_rdist=True
            )
            initial_edges = initialize_boruvka_from_knn(
                neighbors, distances, core_distances, components_disjoint_set,
                head, tail, next_node, cl_indptr, cl_indices,
            )
            update_component_vectors(
                tree, components_disjoint_set, node_components, point_components
            )

    # ---- fix violations from the KNN initialisation round ----
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

        (
            parent_out,
            head_out,
            tail_out,
            next_out,
            all_edges,
            n_init_out,
            _removed,
            _n_rem_init,
        ) = fix_violations(
            n_points,
            all_edges,
            np.intp(n_init),
            init_round_u,
            init_round_v,
            init_round_w,
            np.int32(n_init),
            cl_indptr,
            cl_indices,
            components_disjoint_set.parent,
            head,
            tail,
            next_node,
        )

        for _i in range(n_points):
            components_disjoint_set.parent[_i] = parent_out[_i]
            head[_i] = head_out[_i]
            tail[_i] = tail_out[_i]
            next_node[_i] = next_out[_i]

        all_edges = all_edges[:n_init_out]
        total_violations_removed += np.int64(_n_rem_init)

        # Refresh component vectors after potential splits
        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components
        )

    n_components = len(np.unique(point_components))

    while n_components > 1:
        round_number += np.int64(1)
        # -- Step 1: find cheapest cross-component edges via KD-tree --
        if reproducible:
            block_size = calculate_block_size(
                n_components, n_points, n_threads
            )
            candidate_distances, candidate_indices = boruvka_tree_query_reproducible(
                tree, node_components, point_components, core_distances,
                block_size, cl_indptr, cl_indices,
            )
        else:
            candidate_distances, candidate_indices = boruvka_tree_query(
                tree, node_components, point_components, core_distances,
                cl_indptr, cl_indices,
            )

        # -- Step 2: merge components --
        (
            new_edges,
            round_edges_u,
            round_edges_v,
            round_edges_w,
            round_count,
        ) = merge_components(
            components_disjoint_set,
            candidate_indices,
            candidate_distances,
            point_components,
            head, tail, next_node,
            cl_indptr, cl_indices,
        )

        if len(new_edges) == 0:
            break  # no progress – no cross-component edges found

        # -- Step 3: fix any violations created by this round's merges --
        n_edges_before = all_edges.shape[0]

        if cl_indptr.shape[0] > 1 and round_count > 0:
            n_added = np.intp(n_edges_before + new_edges.shape[0])
            # Build combined MST for fix_violations
            combined = np.empty((n_added, 3), dtype=np.float64)
            for e in range(n_edges_before):
                combined[e, 0] = all_edges[e, 0]
                combined[e, 1] = all_edges[e, 1]
                combined[e, 2] = all_edges[e, 2]
            for e in range(new_edges.shape[0]):
                combined[n_edges_before + e, 0] = new_edges[e, 0]
                combined[n_edges_before + e, 1] = new_edges[e, 1]
                combined[n_edges_before + e, 2] = new_edges[e, 2]

            (
                parent_out,
                head_out,
                tail_out,
                next_out,
                combined,
                n_added_out,
                _removed_edges,
                _n_removed_round,
            ) = fix_violations(
                n_points,
                combined,
                n_added,
                round_edges_u,
                round_edges_v,
                round_edges_w,
                round_count,
                cl_indptr,
                cl_indices,
                components_disjoint_set.parent,
                head,
                tail,
                next_node,
            )
            total_violations_removed += np.int64(_n_removed_round)

            # Update linked-list arrays
            for i in range(n_points):
                head[i] = head_out[i]
                tail[i] = tail_out[i]
                next_node[i] = next_out[i]
                components_disjoint_set.parent[i] = parent_out[i]

            # Replace all_edges with the fixed combined set
            all_edges = combined[:n_added_out]

            # If fix_violations removed every edge we added this round
            # (or more), we made zero net progress – stop to avoid an
            # infinite loop where the same edges are proposed and removed
            # repeatedly.
            if n_added_out <= n_edges_before:
                update_component_vectors(
                    tree, components_disjoint_set, node_components, point_components
                )
                break
        else:
            # No constraints or no round edges – just append
            all_edges = np.vstack((all_edges, new_edges))

        update_component_vectors(
            tree, components_disjoint_set, node_components, point_components
        )

        # Recount components
        n_components = len(np.unique(point_components))

    all_edges[:, 2] = np.sqrt(all_edges.T[2])
    return all_edges, neighbors[:, 1:], np.sqrt(core_distances), round_number, total_violations_removed


# =====================================================================
# High-level wrapper: Borůvka MST → HDBSCAN labels with posthoc cleanup
# =====================================================================


def _pad_spanning_forest(edges, n_points):
    """Pad a spanning forest into a full spanning tree (n-1 edges).

    Disconnected components are bridged with ``np.inf``-weight sentinel
    edges so that ``clusters_from_spanning_tree`` sizes its internal
    arrays correctly.  The infinite weight ensures these bridges are cut
    first during condensation.
    """
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
    bridges = []
    for k in range(1, len(roots)):
        bridges.append([float(roots[0]), float(roots[k]), np.inf])

    if bridges:
        edges = np.vstack([edges, np.array(bridges, dtype=edges.dtype)])
    return edges


def constrained_hdbscan_from_boruvka(
    tree,
    n_threads,
    cl_indptr,
    cl_indices,
    *,
    min_samples=10,
    min_cluster_size=10,
    posthoc_cleanup=True,
    sample_weights=None,
    reproducible=False,
    return_metadata=False,
):
    """Run constrained Borůvka MST → HDBSCAN labels, with optional
    post-hoc cleanup to guarantee zero cannot-link violations.

    Parameters
    ----------
    tree : NumbaKDTree
        Pre-built KD-tree.
    n_threads : int
        Number of Numba threads.
    cl_indptr, cl_indices : ndarray
        CSR-format cannot-link constraint graph.
    min_samples, min_cluster_size : int
        HDBSCAN parameters.
    posthoc_cleanup : bool
        If True, split any cluster that still violates a cannot-link
        constraint after the standard HDBSCAN label extraction.
    sample_weights : ndarray or None
    reproducible : bool
    return_metadata : bool
        If True, return a fourth element: a dict with
        ``n_rounds`` and ``violations_detected_mst``.

    Returns
    -------
    labels : ndarray (n_points,) int64
    probs  : ndarray (n_points,) float64
    edges  : ndarray (n_edges, 3) float64 – the (padded) MST
    metadata : dict (only when *return_metadata* is True)
    """
    from .hdbscan import clusters_from_spanning_tree

    n_points = tree.data.shape[0]

    edges, nbrs, core_dists, n_rounds, n_viols = parallel_boruvka(
        tree, n_threads,
        min_samples=min_samples,
        sample_weights=sample_weights,
        reproducible=reproducible,
        cl_indptr=cl_indptr,
        cl_indices=cl_indices,
    )

    edges = _pad_spanning_forest(edges, n_points)

    # Replace inf with large finite penalty (same logic as hdbscan_cannotLink_orig)
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
        edges, min_cluster_size=min_cluster_size,
    )

    if posthoc_cleanup and cl_indptr.shape[0] > 1:
        from .hdbscan_cannotLink_orig import (
            MergeConstraint,
            _maybe_split_labels_if_cannot_link_violated,
        )
        # Rebuild dense constraint matrix from CSR arrays
        constraint_matrix = np.zeros((n_points, n_points), dtype=bool)
        for row in range(n_points):
            for k in range(cl_indptr[row], cl_indptr[row + 1]):
                col = int(cl_indices[k])
                constraint_matrix[row, col] = True
                constraint_matrix[col, row] = True

        mc = MergeConstraint.from_cannot_link_matrix(
            constraint_matrix, n_points=n_points,
        )
        labels = _maybe_split_labels_if_cannot_link_violated(
            labels, merge_constraint=mc, distances=None, noise_label=-1,
        )

    if return_metadata:
        metadata = {
            "n_rounds": int(n_rounds),
            "violations_detected_mst": int(n_viols),
        }
        return labels, probs, edges, metadata
    return labels, probs, edges
