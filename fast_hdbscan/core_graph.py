import numba
import numpy as np
from collections import namedtuple

from .disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
from .hdbscan import clusters_from_spanning_tree
from .cluster_trees import empty_condensed_tree
from .variables import NUMBA_CACHE

CoreGraph = namedtuple("CoreGraph", ["weights", "distances", "indices", "indptr"])


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def update_point_components(disjoint_set, point_components):
    for i in numba.prange(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def knn_mst_union(neighbors, core_distances, min_spanning_tree, lens_values):
    # List of dictionaries of child: (weight, distance)
    graph = [
        {np.int32(0): (np.float64(0.0), np.float64(0.0)) for _ in range(0)}
        for _ in range(neighbors.shape[0])
    ]

    # Add knn edges
    for point in numba.prange(len(core_distances)):
        children = graph[point]
        parent_lens = lens_values[point]
        parent_dist = core_distances[point]
        for child in neighbors[point]:
            if child < 0:
                continue
            children[child] = (
                max(parent_lens, lens_values[child]),
                max(parent_dist, core_distances[child]),
            )

    # Add non-knn mst edges
    for parent, child, distance in min_spanning_tree:
        parent = np.int32(parent)
        child = np.int32(child)
        children = graph[parent]
        if child in children:
            continue
        children[child] = (max(lens_values[parent], lens_values[child]), distance)

    return graph


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def flatten_to_csr(graph):
    # Count children to form indptr
    num_points = len(graph)
    indptr = np.empty(num_points + 1, dtype=np.int32)
    indptr[0] = 0
    for i, children in enumerate(graph):
        indptr[i + 1] = indptr[i] + len(children)

    # Flatten children to form indices, weights, and distances
    weights = np.empty(indptr[-1], dtype=np.float32)
    distances = np.empty(indptr[-1], dtype=np.float32)
    indices = np.empty(indptr[-1], dtype=np.int32)
    for point in numba.prange(num_points):
        start = indptr[point]
        children = graph[point]
        for j, (child, (weight, distance)) in enumerate(children.items()):
            weights[start + j] = weight
            distances[start + j] = distance
            indices[start + j] = child

    # Return as named csr tuple
    return CoreGraph(weights, distances, indices, indptr)


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def sort_by_lens(graph):
    new_weights = np.empty_like(graph.weights)
    new_distances = np.empty_like(graph.distances)
    new_indices = np.empty_like(graph.indices)
    for point in numba.prange(len(graph.indptr) - 1):
        start = graph.indptr[point]
        end = graph.indptr[point + 1]

        row_weights = graph.weights[start:end]
        row_distances = graph.distances[start:end]
        row_indices = graph.indices[start:end]

        order = np.argsort(row_weights)
        new_weights[start:end] = row_weights[order]
        new_distances[start:end] = row_distances[order]
        new_indices[start:end] = row_indices[order]
    return CoreGraph(new_weights, new_distances, new_indices, graph.indptr)


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def apply_lens(core_graph, lens_values):
    # Apply new lens to the graph
    for point in numba.prange(len(lens_values)):
        point_lens = lens_values[point]
        start = core_graph.indptr[point]
        end = core_graph.indptr[point + 1]
        for idx, child in enumerate(core_graph.indices[start:end]):
            core_graph.weights[start + idx] = max(point_lens, lens_values[child])
    return sort_by_lens(core_graph)


@numba.njit(locals={"parent": numba.types.int32}, cache=NUMBA_CACHE)
def select_components(distances, indices, indptr, point_components):
    component_edges = {
        np.int64(0): (np.int32(0), np.int32(1), np.float32(0.0)) for _ in range(0)
    }

    # Find the best edges from each component
    for parent, from_component in enumerate(point_components):
        start = indptr[parent]
        if start == len(indices) or indices[start] == -1:
            continue

        neighbor = indices[start]
        distance = distances[start]
        if from_component in component_edges:
            if distance < component_edges[from_component][2]:
                component_edges[from_component] = (parent, neighbor, distance)
        else:
            component_edges[from_component] = (parent, neighbor, distance)

    return component_edges


@numba.njit(cache=NUMBA_CACHE)
def merge_components(disjoint_set, component_edges):
    result = np.empty((len(component_edges), 3), dtype=np.float64)
    result_idx = 0

    # Add the best edges to the edge set and merge the relevant components
    for edge in component_edges.values():
        from_component = ds_find(disjoint_set, edge[0])
        to_component = ds_find(disjoint_set, edge[1])
        if from_component != to_component:
            result[result_idx] = (
                np.float64(edge[0]),
                np.float64(edge[1]),
                np.float64(edge[2]),
            )
            result_idx += 1

            ds_union_by_rank(disjoint_set, from_component, to_component)

    return result[:result_idx]


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def update_graph_components(distances, indices, indptr, point_components):
    for point in numba.prange(len(point_components)):
        counter = 0
        start = indptr[point]
        end = indptr[point + 1]
        for idx in range(start, end):
            child = indices[idx]
            if child == -1:
                break
            if point_components[child] != point_components[point]:
                indices[start + counter] = indices[idx]
                distances[start + counter] = distances[idx]
                counter += 1
        indices[start + counter : end] = -1
        distances[start + counter : end] = np.inf


@numba.njit(cache=NUMBA_CACHE)
def boruvka_mst(graph, overwrite=False):
    """
    Implements Boruvka on lod-style graph with multiple connected components.
    """
    distances = graph.weights
    indices = graph.indices
    indptr = graph.indptr
    if not overwrite:
        indices = indices.copy()
        distances = distances.copy()

    disjoint_set = ds_rank_create(len(indptr) - 1)
    point_components = np.arange(len(indptr) - 1)
    n_components = len(point_components)

    edges_list = [np.empty((0, 3), dtype=np.float64) for _ in range(0)]
    while n_components > 1:
        new_edges = merge_components(
            disjoint_set,
            select_components(distances, indices, indptr, point_components),
        )
        if new_edges.shape[0] == 0:
            break

        edges_list.append(new_edges)
        update_point_components(disjoint_set, point_components)
        update_graph_components(distances, indices, indptr, point_components)
        n_components -= new_edges.shape[0]

    counter = 0
    num_edges = sum([edges.shape[0] for edges in edges_list])
    result = np.empty((num_edges, 3), dtype=np.float64)
    for edges in edges_list:
        result[counter : counter + edges.shape[0]] = edges
        counter += edges.shape[0]
    return n_components, point_components, result


@numba.njit(cache=NUMBA_CACHE)
def core_graph_spanning_tree(neighbors, core_distances, min_spanning_tree, lens):
    graph = sort_by_lens(
        flatten_to_csr(
            knn_mst_union(neighbors, core_distances, min_spanning_tree, lens)
        )
    )
    return (*boruvka_mst(graph), graph)


def core_graph_clusters(
    lens,
    neighbors,
    core_distances,
    min_spanning_tree,
    **kwargs,
):
    num_components, component_labels, lensed_mst, graph = core_graph_spanning_tree(
        neighbors, core_distances, min_spanning_tree, lens
    )
    if num_components > 1:
        for i, label in enumerate(np.unique(component_labels)):
            component_labels[component_labels == label] = i
        return (
            component_labels,
            np.ones(len(component_labels), dtype=np.float32),
            np.empty((0, 4)),
            empty_condensed_tree(),
            lensed_mst,
            graph,
        )

    return (
        *clusters_from_spanning_tree(lensed_mst, **kwargs),
        graph,
    )


# ---------------------------------------------------------------------------
# CL-constrained Borůvka helpers
# ---------------------------------------------------------------------------

@numba.njit(cache=NUMBA_CACHE)
def _init_cl_pool(cl_indices, cl_indptr, n_verts):
    """
    Build the linked-list pool for per-component conflict tracking.

    Same data structure as _kruskal_core_constrained: each component gets a
    singly-linked list of its CL partner vertices.

    Returns
    -------
    pool_vertex : int32[M]  — vertex id for each constraint entry
    pool_next   : int32[M]  — next pointer (-1 = end of list)
    comp_head   : int32[n]  — head of conflict list per component
    comp_tail   : int32[n]  — tail of conflict list per component
    comp_csize  : int32[n]  — constraint count per component
    """
    M = len(cl_indices)
    pool_vertex = np.empty(M, dtype=np.int32)
    pool_next = np.full(M, -1, dtype=np.int32)

    comp_head = np.full(n_verts, -1, dtype=np.int32)
    comp_tail = np.full(n_verts, -1, dtype=np.int32)
    comp_csize = np.zeros(n_verts, dtype=np.int32)

    pool_idx = 0
    for i in range(n_verts):
        for p in range(cl_indptr[i], cl_indptr[i + 1]):
            pool_vertex[pool_idx] = cl_indices[p]
            if comp_tail[i] >= 0:
                pool_next[comp_tail[i]] = np.int32(pool_idx)
            else:
                comp_head[i] = np.int32(pool_idx)
            comp_tail[i] = np.int32(pool_idx)
            comp_csize[i] += 1
            pool_idx += 1

    return pool_vertex, pool_next, comp_head, comp_tail, comp_csize


@numba.njit(cache=NUMBA_CACHE)
def _check_cl_conflict(root_a, root_b, comp_head, comp_csize,
                        pool_vertex, pool_next, predecessors):
    """
    Check if merging components root_a and root_b would create a CL violation.

    Scans the smaller component's conflict list and checks if any vertex
    resolves to the other root.  Returns True if conflict exists.
    """
    if comp_csize[root_a] == 0 or comp_csize[root_b] == 0:
        return False

    if comp_csize[root_a] <= comp_csize[root_b]:
        small_root = root_a
        big_root = root_b
    else:
        small_root = root_b
        big_root = root_a

    cur = comp_head[small_root]
    while cur >= 0:
        v_cl = pool_vertex[cur]
        # Read-only find (no path compression)
        root_cl = v_cl
        while predecessors[root_cl] != root_cl:
            root_cl = predecessors[root_cl]
        if root_cl == big_root:
            return True
        cur = pool_next[cur]
    return False


@numba.njit(cache=NUMBA_CACHE)
def _merge_cl_lists(new_root, old_root, comp_head, comp_tail, comp_csize,
                     pool_next):
    """
    O(1) concatenation of old_root's conflict list onto new_root's list.
    """
    if comp_head[old_root] >= 0:
        if comp_tail[new_root] >= 0:
            pool_next[comp_tail[new_root]] = comp_head[old_root]
        else:
            comp_head[new_root] = comp_head[old_root]
        comp_tail[new_root] = comp_tail[old_root]
        comp_csize[new_root] += comp_csize[old_root]

    comp_head[old_root] = np.int32(-1)
    comp_tail[old_root] = np.int32(-1)
    comp_csize[old_root] = 0


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _select_per_vertex_cl(distances, indices, indptr, point_components,
                           comp_head, comp_csize, pool_vertex, pool_next,
                           predecessors, vert_best_dst, vert_best_wt):
    """
    Phase 1 (parallel): each vertex finds its cheapest CL-safe outgoing edge.

    Writes to per-vertex arrays vert_best_dst[v] and vert_best_wt[v].
    -1 / inf means no valid edge found.  All inputs are read-only.
    """
    n = len(point_components)
    for v in numba.prange(n):
        from_component = point_components[v]
        start = indptr[v]
        end = indptr[v + 1]

        for idx in range(start, end):
            neighbor = indices[idx]
            if neighbor == -1:
                break
            distance = distances[idx]
            to_component = point_components[neighbor]
            if to_component == from_component:
                continue

            # O(1) root lookup via compressed snapshot
            root_from = predecessors[from_component]
            root_to = predecessors[to_component]

            if _check_cl_conflict(root_from, root_to, comp_head, comp_csize,
                                   pool_vertex, pool_next, predecessors):
                continue

            # First valid = cheapest (rows sorted ascending)
            vert_best_dst[v] = neighbor
            vert_best_wt[v] = distance
            break


@numba.njit(cache=NUMBA_CACHE)
def select_components_cl(distances, indices, indptr, point_components,
                          comp_head, comp_csize, pool_vertex, pool_next,
                          predecessors):
    """
    CL-aware edge selection: for each component, find the cheapest outgoing
    edge that does not violate a cannot-link constraint.

    Two-phase approach:
      Phase 1 (parallel): each vertex finds its cheapest CL-safe outgoing edge.
      Phase 2 (serial O(n)): reduce per-vertex results to per-component minimum.

    Returns flat arrays (src, dst, wt, n_edges) instead of a typed dict.
    """
    n = len(point_components)

    # Phase 1: parallel per-vertex scan
    vert_best_dst = np.full(n, -1, dtype=np.int32)
    vert_best_wt = np.full(n, np.inf, dtype=np.float32)
    _select_per_vertex_cl(distances, indices, indptr, point_components,
                           comp_head, comp_csize, pool_vertex, pool_next,
                           predecessors, vert_best_dst, vert_best_wt)

    # Phase 2: reduce to per-component minimum (serial, O(n))
    comp_best_src = np.full(n, -1, dtype=np.int32)
    comp_best_dst = np.full(n, -1, dtype=np.int32)
    comp_best_wt = np.full(n, np.inf, dtype=np.float32)

    for v in range(n):
        if vert_best_dst[v] < 0:
            continue
        c = point_components[v]
        if vert_best_wt[v] < comp_best_wt[c]:
            comp_best_src[c] = np.int32(v)
            comp_best_dst[c] = vert_best_dst[v]
            comp_best_wt[c] = vert_best_wt[v]

    # Pack into flat output arrays
    n_edges = 0
    for c in range(n):
        if comp_best_src[c] >= 0:
            n_edges += 1

    out_src = np.empty(n_edges, dtype=np.int32)
    out_dst = np.empty(n_edges, dtype=np.int32)
    out_wt = np.empty(n_edges, dtype=np.float32)
    j = 0
    for c in range(n):
        if comp_best_src[c] >= 0:
            out_src[j] = comp_best_src[c]
            out_dst[j] = comp_best_dst[c]
            out_wt[j] = comp_best_wt[c]
            j += 1

    return out_src, out_dst, out_wt, n_edges


@numba.njit(cache=NUMBA_CACHE)
def _bfs_heaviest_edge(comp_u, comp_v, tree_src, tree_dst, tree_wt,
                        tree_alive, n_tree_edges, adj_head, adj_next,
                        adj_edge_idx, bfs_queue, bfs_parent_edge, bfs_visited):
    """
    BFS in the merge tree from comp_u to comp_v.  Returns the index of the
    heaviest alive edge on the path, or -1 if no path exists.

    adj_head/adj_next/adj_edge_idx form a flat adjacency list built from
    alive tree edges.
    """
    # Reset visited
    bfs_visited[comp_u] = True
    bfs_parent_edge[comp_u] = -1

    q_front = 0
    q_back = 0
    bfs_queue[q_back] = comp_u
    q_back += 1

    found = False
    while q_front < q_back:
        node = bfs_queue[q_front]
        q_front += 1

        # Iterate adjacency list
        e = adj_head[node]
        while e >= 0:
            ei = adj_edge_idx[e]
            if not tree_alive[ei]:
                e = adj_next[e]
                continue
            # Determine neighbor
            if tree_src[ei] == node:
                nbr = tree_dst[ei]
            else:
                nbr = tree_src[ei]

            if not bfs_visited[nbr]:
                bfs_visited[nbr] = True
                bfs_parent_edge[nbr] = ei
                bfs_queue[q_back] = nbr
                q_back += 1
                if nbr == comp_v:
                    found = True
                    break
            e = adj_next[e]
        if found:
            break

    if not found:
        # Clean up visited
        for i in range(q_back):
            bfs_visited[bfs_queue[i]] = False
        return -1

    # Backtrack from comp_v to comp_u, find heaviest edge
    heaviest_idx = -1
    heaviest_wt = np.float32(-1.0)
    cur = comp_v
    while cur != comp_u:
        ei = bfs_parent_edge[cur]
        if tree_wt[ei] > heaviest_wt:
            heaviest_wt = tree_wt[ei]
            heaviest_idx = ei
        if tree_src[ei] == cur:
            cur = tree_dst[ei]
        else:
            cur = tree_src[ei]

    # Clean up visited
    for i in range(q_back):
        bfs_visited[bfs_queue[i]] = False

    return heaviest_idx


@numba.njit(cache=NUMBA_CACHE)
def validate_and_prune_merges(candidate_src, candidate_dst, candidate_wt,
                               n_candidates, cl_indices, cl_indptr,
                               point_components, n_verts,
                               _adj_head, _adj_next, _adj_edge_idx,
                               _bfs_queue, _bfs_parent_edge, _bfs_visited,
                               _temp_parent):
    """
    Steps (b)+(c): tentative merge then cleanup of transitive CL violations.

    Builds a temporary DSU from candidate edges, then scans CL pairs for
    violations (scoped to vertices whose components participate in merges).
    For each violation, BFS in the merge tree finds the heaviest edge on the
    path between the two violating pre-round components and marks it for
    removal.  Repeats until clean.

    Scratch arrays (_adj_head, etc.) are pre-allocated by the caller to avoid
    repeated allocation across rounds.

    Returns
    -------
    surviving_src, surviving_dst, surviving_wt : arrays of surviving edges
    n_surviving : int
    """
    if n_candidates == 0:
        return candidate_src[:0], candidate_dst[:0], candidate_wt[:0], 0

    # Merge tree edge arrays
    tree_src = np.empty(n_candidates, dtype=np.int32)
    tree_dst = np.empty(n_candidates, dtype=np.int32)
    tree_wt = np.empty(n_candidates, dtype=np.float32)
    tree_alive = np.ones(n_candidates, dtype=numba.boolean)

    # Map edges to component space and collect involved components
    involved_comps = np.empty(2 * n_candidates, dtype=np.int32)
    n_involved = 0
    for i in range(n_candidates):
        s = point_components[candidate_src[i]]
        d = point_components[candidate_dst[i]]
        tree_src[i] = s
        tree_dst[i] = d
        tree_wt[i] = candidate_wt[i]
        involved_comps[n_involved] = s
        involved_comps[n_involved + 1] = d
        n_involved += 2

    # Deduplicate involved components
    involved_comps = involved_comps[:n_involved]
    involved_comps.sort()
    n_unique = 0
    for i in range(n_involved):
        if i == 0 or involved_comps[i] != involved_comps[i - 1]:
            involved_comps[n_unique] = involved_comps[i]
            n_unique += 1
    involved_comps = involved_comps[:n_unique]

    # Build a lookup: is_involved[comp] = True for scoped CL scan
    is_involved = _bfs_visited  # reuse bool array (will reset before BFS)
    for i in range(n_unique):
        is_involved[involved_comps[i]] = True

    # Collect vertices whose component is involved (for scoped CL scan)
    scan_verts = np.empty(n_verts, dtype=np.int32)
    n_scan = 0
    for u in range(n_verts):
        if is_involved[point_components[u]]:
            scan_verts[n_scan] = u
            n_scan += 1

    # Clear is_involved
    for i in range(n_unique):
        is_involved[involved_comps[i]] = False

    max_iters = n_candidates

    for _iteration in range(max_iters):
        # Reset temp DSU only for involved components
        for i in range(n_unique):
            c = involved_comps[i]
            _temp_parent[c] = c

        # Reset adjacency only for involved components
        for i in range(n_unique):
            _adj_head[involved_comps[i]] = -1

        adj_ptr = 0
        for i in range(n_candidates):
            if not tree_alive[i]:
                continue
            s = tree_src[i]
            d = tree_dst[i]

            rs = s
            while _temp_parent[rs] != rs:
                rs = _temp_parent[rs]
            rd = d
            while _temp_parent[rd] != rd:
                rd = _temp_parent[rd]

            if rs != rd:
                _temp_parent[rd] = rs

                _adj_next[adj_ptr] = _adj_head[s]
                _adj_edge_idx[adj_ptr] = np.int32(i)
                _adj_head[s] = np.int32(adj_ptr)
                adj_ptr += 1

                _adj_next[adj_ptr] = _adj_head[d]
                _adj_edge_idx[adj_ptr] = np.int32(i)
                _adj_head[d] = np.int32(adj_ptr)
                adj_ptr += 1

        # Path compression only for involved components
        for i in range(n_unique):
            c = involved_comps[i]
            root = _temp_parent[c]
            while _temp_parent[root] != root:
                root = _temp_parent[root]
            curr = c
            while curr != root:
                nxt = _temp_parent[curr]
                _temp_parent[curr] = root
                curr = nxt

        # Scoped CL scan: only vertices whose component is involved
        edges_to_remove = np.full(n_candidates, False, dtype=numba.boolean)
        found_violation = False

        for si in range(n_scan):
            u = scan_verts[si]
            for p in range(cl_indptr[u], cl_indptr[u + 1]):
                v = cl_indices[p]
                if v <= u:
                    continue

                comp_u = point_components[u]
                comp_v = point_components[v]
                if comp_u == comp_v:
                    continue

                if _temp_parent[comp_u] != _temp_parent[comp_v]:
                    continue

                heaviest = _bfs_heaviest_edge(
                    comp_u, comp_v, tree_src, tree_dst, tree_wt,
                    tree_alive, n_candidates, _adj_head, _adj_next,
                    _adj_edge_idx, _bfs_queue, _bfs_parent_edge, _bfs_visited
                )
                if heaviest >= 0:
                    edges_to_remove[heaviest] = True
                    found_violation = True

        if not found_violation:
            break

        for i in range(n_candidates):
            if edges_to_remove[i]:
                tree_alive[i] = False

    # Collect surviving edges
    n_surviving = 0
    for i in range(n_candidates):
        if tree_alive[i]:
            n_surviving += 1

    surv_src = np.empty(n_surviving, dtype=np.int32)
    surv_dst = np.empty(n_surviving, dtype=np.int32)
    surv_wt = np.empty(n_surviving, dtype=np.float32)
    j = 0
    for i in range(n_candidates):
        if tree_alive[i]:
            surv_src[j] = candidate_src[i]
            surv_dst[j] = candidate_dst[i]
            surv_wt[j] = candidate_wt[i]
            j += 1

    return surv_src, surv_dst, surv_wt, n_surviving


@numba.njit(cache=NUMBA_CACHE)
def boruvka_mst_cl(graph, cl_indices, cl_indptr, band_fraction=np.inf,
                    overwrite=False):
    """
    Borůvka MST with cannot-link constraints -> Minimum Spanning Forest.

    3-step inner loop per round:
    (a) Select: CL-safe cheapest outgoing edge per component
    (b) Merge: Tentative merge with temporary DSU
    (c) Cleanup: Detect and repair transitive CL violations by cutting
        the heaviest edge on the path between violating components

    When band_fraction < inf, only edges within band_fraction of the minimum
    candidate weight are allowed per round.  This produces a more Kruskal-like
    (sequential) merge order which improves CL accuracy at the cost of more
    rounds.

    Parameters
    ----------
    graph : CoreGraph namedtuple (weights, distances, indices, indptr)
    cl_indices : int32[:], CSR column indices of symmetric CL graph
    cl_indptr  : int32[:], CSR row pointers (length n+1)
    band_fraction : float
        Fraction of the minimum weight that defines the band upper bound.
        band_hi = w_min * (1 + band_fraction).
        np.inf = no banding (standard Borůvka).
        0.05  = 5% band (close to Kruskal ordering).
    overwrite : bool

    Returns
    -------
    n_components : int
    point_components : int32[:] shape (n,)
    mst_edges : float64[:, 3]
    """
    distances = graph.weights
    indices = graph.indices
    indptr = graph.indptr
    if not overwrite:
        indices = indices.copy()
        distances = distances.copy()

    n = len(indptr) - 1
    disjoint_set = ds_rank_create(n)
    point_components = np.arange(n, dtype=np.int32)
    n_components = n

    # Initialize CL linked-list pool
    pool_vertex, pool_next, comp_head, comp_tail, comp_csize = _init_cl_pool(
        cl_indices, cl_indptr, n
    )

    edges_list = [np.empty((0, 3), dtype=np.float64) for _ in range(0)]
    use_banding = band_fraction < 1e30  # avoid inf comparisons

    # Pre-allocate scratch arrays for validate_and_prune_merges (reused each round)
    max_adj = 2 * n  # upper bound: at most n/2 candidates per round
    _adj_head = np.full(n, -1, dtype=np.int32)
    _adj_next = np.empty(max_adj, dtype=np.int32)
    _adj_edge_idx = np.empty(max_adj, dtype=np.int32)
    _bfs_queue = np.empty(n, dtype=np.int32)
    _bfs_parent_edge = np.full(n, -1, dtype=np.int32)
    _bfs_visited = np.zeros(n, dtype=numba.boolean)
    _temp_parent = np.arange(n, dtype=np.int32)

    # Compressed root snapshot for parallel select (avoids deep tree traversal)
    compressed_roots = np.arange(n, dtype=np.int32)

    while n_components > 1:
        # Snapshot compressed roots for O(1) lookups in parallel select
        for i in range(n):
            root = disjoint_set.parent[i]
            while disjoint_set.parent[root] != root:
                root = disjoint_set.parent[root]
            compressed_roots[i] = root

        # Step (a): CL-safe edge selection (parallel phase 1 + serial phase 2)
        cand_src, cand_dst, cand_wt, n_cand = select_components_cl(
            distances, indices, indptr, point_components,
            comp_head, comp_csize, pool_vertex, pool_next,
            compressed_roots
        )
        if n_cand == 0:
            break

        # Band filtering: keep only edges within band_fraction of the minimum
        if use_banding:
            w_min = cand_wt[0]
            for i in range(1, n_cand):
                if cand_wt[i] < w_min:
                    w_min = cand_wt[i]
            band_hi = np.float64(w_min) * (1.0 + band_fraction)
            n_in_band = 0
            for i in range(n_cand):
                if np.float64(cand_wt[i]) <= band_hi:
                    cand_src[n_in_band] = cand_src[i]
                    cand_dst[n_in_band] = cand_dst[i]
                    cand_wt[n_in_band] = cand_wt[i]
                    n_in_band += 1
            n_cand = n_in_band

        # Steps (b)+(c): tentative merge + cleanup
        surv_src, surv_dst, surv_wt, n_surviving = validate_and_prune_merges(
            cand_src, cand_dst, cand_wt, n_cand,
            cl_indices, cl_indptr, point_components, n,
            _adj_head, _adj_next, _adj_edge_idx,
            _bfs_queue, _bfs_parent_edge, _bfs_visited, _temp_parent
        )

        if n_surviving == 0:
            break

        # Commit surviving edges to main DSU + CL lists
        new_edges = np.empty((n_surviving, 3), dtype=np.float64)
        n_added = 0
        for i in range(n_surviving):
            src = surv_src[i]
            dst = surv_dst[i]

            root_src = ds_find(disjoint_set, src)
            root_dst = ds_find(disjoint_set, dst)
            if root_src != root_dst:
                new_edges[n_added, 0] = np.float64(src)
                new_edges[n_added, 1] = np.float64(dst)
                new_edges[n_added, 2] = np.float64(surv_wt[i])
                n_added += 1

                # Union by rank
                if disjoint_set.rank[root_src] > disjoint_set.rank[root_dst]:
                    new_root = root_src
                    old_root = root_dst
                elif disjoint_set.rank[root_src] < disjoint_set.rank[root_dst]:
                    new_root = root_dst
                    old_root = root_src
                else:
                    new_root = root_src
                    old_root = root_dst
                    disjoint_set.rank[new_root] += 1
                disjoint_set.parent[old_root] = new_root

                # Merge CL lists
                _merge_cl_lists(new_root, old_root, comp_head, comp_tail,
                                comp_csize, pool_next)

        if n_added == 0:
            break

        edges_list.append(new_edges[:n_added])
        update_point_components(disjoint_set, point_components)
        update_graph_components(distances, indices, indptr, point_components)
        n_components -= n_added

    counter = 0
    num_edges = sum([edges.shape[0] for edges in edges_list])
    result = np.empty((num_edges, 3), dtype=np.float64)
    for edges in edges_list:
        result[counter : counter + edges.shape[0]] = edges
        counter += edges.shape[0]
    return n_components, point_components, result


def core_graph_to_rec_array(graph):
    result = np.empty(
        graph.indptr[-1],
        dtype=[
            ("parent", np.int32),
            ("child", np.int32),
            ("weight", np.float32),
            ("distance", np.float32),
        ],
    )
    result["parent"] = np.repeat(
        np.arange(len(graph.indptr) - 1), np.diff(graph.indptr)
    )
    result["child"] = graph.indices
    result["weight"] = graph.weights
    result["distance"] = graph.distances
    return result


def core_graph_to_edge_list(graph):
    result = np.empty((graph.indptr[-1], 4), dtype=np.float64)
    result[:, 0] = np.repeat(np.arange(len(graph.indptr) - 1), np.diff(graph.indptr))
    result[:, 1] = graph.indices
    result[:, 2] = graph.weights
    result[:, 3] = graph.distances
    return result
