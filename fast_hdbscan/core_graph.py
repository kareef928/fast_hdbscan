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
    resolves to the other root.  Returns (conflict_found, n_walk_steps) where
    n_walk_steps counts linked-list nodes visited in the outer pool walk.
    """
    if comp_csize[root_a] == 0 or comp_csize[root_b] == 0:
        return False, np.int64(0)

    if comp_csize[root_a] <= comp_csize[root_b]:
        small_root = root_a
        big_root = root_b
    else:
        small_root = root_b
        big_root = root_a

    n_walk_steps = np.int64(0)
    cur = comp_head[small_root]
    while cur >= 0:
        n_walk_steps += np.int64(1)
        v_cl = pool_vertex[cur]
        # Read-only find (no path compression)
        root_cl = v_cl
        while predecessors[root_cl] != root_cl:
            root_cl = predecessors[root_cl]
        if root_cl == big_root:
            return True, n_walk_steps
        cur = pool_next[cur]
    return False, n_walk_steps


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


# ---------------------------------------------------------------------------
# Sorted-array CL conflict structure (Fix 2 / Fix 4)
# ---------------------------------------------------------------------------
# Replacement for the linked-list pool above.  Stores per-component, sorted,
# deduplicated arrays of partner *component IDs* (not vertex IDs as in the
# legacy pool).  Conflict checks become O(log M) binary searches; merges
# rewrite-on-merge with sorted-merge dedup (small-to-larger amortizes to
# O(M log n) total work over the whole Borůvka run).
#
# Storage: flat int32 buffer ``cl_data`` with per-row slots laid out
# CSR-style.  Each row's slot has a fixed start (``cl_arr_start[c]``) and
# capacity (``cl_arr_capacity[c]``).  Live size is tracked separately
# (``comp_csize_arr[c]``).  When a merge would overflow the destination's
# slot, we relocate the merged row to the tail of ``cl_data`` (controlled by
# ``next_free_arr[0]``); the old slot becomes a permanent gap.
#
# The buffer is pre-allocated with generous padding; the merge function
# raises a RuntimeError if the buffer is exhausted, surfacing a sizing bug
# loudly rather than corrupting state.

@numba.njit(cache=NUMBA_CACHE)
def _init_cl_arrays(cl_indices, cl_indptr, n_verts):
    """
    Build the per-component sorted/deduped partner-ID arrays.

    At t=0 every vertex is its own component, so the input
    ``cl_indices[cl_indptr[i]:cl_indptr[i+1]]`` already lists partner
    component IDs for component ``i``.  This routine:

      1. Sorts and dedups each row.
      2. Strips self-references.
      3. Lays out the rows in a flat ``int32`` buffer with extra capacity
         per slot, so subsequent in-place rewrites and (limited) merge
         absorption stay within their original slots most of the time.
      4. Allocates extra tail space for relocation when an in-place merge
         would overflow.

    Returns
    -------
    cl_data : int32[total_cap]
        Flat data buffer; per-row entries live at
        ``cl_data[cl_arr_start[c] : cl_arr_start[c] + comp_csize_arr[c]]``.
    cl_arr_start : int32[n_verts]
        Start index of each row's slot inside ``cl_data``.
    cl_arr_capacity : int32[n_verts]
        Slot capacity (max live size before relocation) per row.
    comp_csize_arr : int32[n_verts]
        Live count of partner IDs per row.
    next_free_arr : int32[1]
        Next free index in ``cl_data`` for tail relocations.  Single-element
        array to allow in-place mutation from JIT helpers.
    """
    M = np.int64(len(cl_indices))
    n_verts_i64 = np.int64(n_verts)

    ## Per-row pad: 2× the original row size (rounded up to >=4) so that
    ## modest in-place absorption fits without relocation.
    ## Total buffer pad: 8× M plus n_verts*4 for first-row pads.  Generous
    ## but bounded; at user's scale (M=1.25M, n=50k) this is ~40MB.
    ## ``next_free_arr`` starts at the post-initial-rows offset.
    pad_per_row = np.int32(4)
    init_total = np.int64(0)
    for i in range(n_verts):
        row_len = cl_indptr[i + 1] - cl_indptr[i]
        cap = np.int32(2) * row_len
        if cap < pad_per_row:
            cap = pad_per_row
        init_total += np.int64(cap)

    ## Tail pad — extra space for relocations during merges.
    tail_pad = np.int64(8) * M + n_verts_i64 * np.int64(4) + np.int64(1024)
    total_cap = init_total + tail_pad

    cl_data = np.zeros(total_cap, dtype=np.int32)
    cl_arr_start = np.empty(n_verts, dtype=np.int32)
    cl_arr_capacity = np.empty(n_verts, dtype=np.int32)
    comp_csize_arr = np.zeros(n_verts, dtype=np.int32)

    ## Layout slots and copy/sort/dedup each row.
    cursor = np.int64(0)
    for i in range(n_verts):
        row_lo = cl_indptr[i]
        row_hi = cl_indptr[i + 1]
        row_len = row_hi - row_lo
        cap = np.int32(2) * row_len
        if cap < pad_per_row:
            cap = pad_per_row

        cl_arr_start[i] = np.int32(cursor)
        cl_arr_capacity[i] = cap

        if row_len > 0:
            ## Copy row, then sort and dedup in place (skip self-refs).
            ## Use np.sort on a temporary so JIT can dispatch a numeric sort.
            tmp = np.empty(row_len, dtype=np.int32)
            for k in range(row_len):
                tmp[k] = cl_indices[row_lo + k]
            tmp.sort()
            ## Dedup + strip self-refs in one pass.
            n_live = np.int32(0)
            prev = np.int32(-1)
            self_ref = np.int32(i)
            for k in range(row_len):
                v = tmp[k]
                if v == self_ref:
                    continue
                if n_live > 0 and v == prev:
                    continue
                cl_data[cursor + np.int64(n_live)] = v
                prev = v
                n_live += np.int32(1)
            comp_csize_arr[i] = n_live

        cursor += np.int64(cap)

    next_free_arr = np.empty(1, dtype=np.int64)
    next_free_arr[0] = cursor

    return cl_data, cl_arr_start, cl_arr_capacity, comp_csize_arr, next_free_arr


@numba.njit(cache=NUMBA_CACHE)
def _check_cl_conflict_sorted(root_a, root_b, cl_data, cl_arr_start,
                                comp_csize_arr):
    """
    Sorted-array variant of ``_check_cl_conflict``.

    Binary-searches the smaller component's partner array for the other
    root.  Returns ``(conflict_bool, n_walk_steps)`` where ``n_walk_steps``
    counts comparisons (~``log2(size)``).  The counter shape mirrors the
    legacy linked-list helper for instrumentation parity (the pyloop's
    ``vert_n_cl_walk_steps`` accumulator expects an int64).
    """
    size_a = comp_csize_arr[root_a]
    size_b = comp_csize_arr[root_b]
    if size_a == 0 or size_b == 0:
        return False, np.int64(0)

    if size_a <= size_b:
        small_root = root_a
        big_root = root_b
        small_size = size_a
    else:
        small_root = root_b
        big_root = root_a
        small_size = size_b

    ## Branch-free binary search over the smaller side's slot.
    start = cl_arr_start[small_root]
    target = np.int32(big_root)
    lo = np.int32(0)
    hi = small_size  # exclusive
    n_steps = np.int64(0)
    while lo < hi:
        n_steps += np.int64(1)
        mid = (lo + hi) >> np.int32(1)
        v = cl_data[start + mid]
        if v == target:
            return True, n_steps
        if v < target:
            lo = mid + np.int32(1)
        else:
            hi = mid
    return False, n_steps


@numba.njit(cache=NUMBA_CACHE)
def _sorted_array_replace(start, size, old_val, new_val, cl_data):
    """
    In-place replacement of ``old_val`` with ``new_val`` inside the sorted
    slot ``cl_data[start : start + size]``.

    This helper is the per-partner step of rewrite-on-merge.  It is
    careful to preserve sorted order even when the new value is far from
    the old value's position.

    Returns
    -------
    new_size : int32
        Updated live size of the slot.  Equal to ``size`` when ``new_val``
        was absent (or when ``old_val`` was absent — defensive no-op).
        Equal to ``size - 1`` when ``new_val`` was already present and
        ``old_val`` was dropped to keep the slot deduped.
    """
    ## Locate old_val via binary search.  If absent, nothing to do (defensive).
    lo = np.int32(0)
    hi = size
    old_pos = np.int32(-1)
    while lo < hi:
        mid = (lo + hi) >> np.int32(1)
        v = cl_data[start + mid]
        if v == old_val:
            old_pos = mid
            break
        if v < old_val:
            lo = mid + np.int32(1)
        else:
            hi = mid
    if old_pos < 0:
        return size  # not present; nothing to do

    ## Locate new_val (or its insertion point) ignoring the old slot.
    ## Binary-search for new_val across the whole slot.
    lo2 = np.int32(0)
    hi2 = size
    new_pos = np.int32(-1)
    while lo2 < hi2:
        mid = (lo2 + hi2) >> np.int32(1)
        v = cl_data[start + mid]
        if v == new_val:
            new_pos = mid
            break
        if v < new_val:
            lo2 = mid + np.int32(1)
        else:
            hi2 = mid

    if new_pos >= 0:
        ## new_val already present → remove old_pos to dedupe.
        ## Shift left starting at old_pos+1.
        for k in range(int(old_pos), int(size) - 1):
            cl_data[start + np.int32(k)] = cl_data[start + np.int32(k + 1)]
        return size - np.int32(1)

    ## new_val absent — write at insertion position (lo2), shifting elements
    ## to keep sorted order.  Algorithm:
    ##   * remove old_val (shift left from old_pos+1)
    ##   * compute insertion position in the now-shrunk array
    ##   * insert new_val (shift right)
    ## Done as a single in-place slide for efficiency.
    ins_pos = lo2  # insertion index in the original (size-element) array
    ## After removing old_val, indices > old_pos shift down by 1.  Adjust.
    if ins_pos > old_pos:
        ins_pos = ins_pos - np.int32(1)

    if ins_pos == old_pos:
        ## Common case: simple in-place overwrite (sorted order preserved).
        cl_data[start + old_pos] = new_val
    elif ins_pos < old_pos:
        ## Shift right: elements [ins_pos .. old_pos-1] move one slot up.
        k = old_pos
        while k > ins_pos:
            cl_data[start + k] = cl_data[start + np.int32(k - 1)]
            k -= np.int32(1)
        cl_data[start + ins_pos] = new_val
    else:
        ## Shift left: elements [old_pos+1 .. ins_pos] move one slot down.
        k = old_pos
        while k < ins_pos:
            cl_data[start + k] = cl_data[start + np.int32(k + 1)]
            k += np.int32(1)
        cl_data[start + ins_pos] = new_val

    return size  # count unchanged


@numba.njit(cache=NUMBA_CACHE)
def _merge_cl_arrays(new_root, old_root, cl_data, cl_arr_start,
                       cl_arr_capacity, comp_csize_arr, next_free_arr,
                       scratch):
    """
    Sorted-array variant of ``_merge_cl_lists``.

    Steps:
      1. Determine smaller / larger side by live size.
      2. Rewrite each partner P in the smaller side's array: replace
         ``smaller_root`` with ``larger_root`` inside P's array.  If P
         already had ``larger_root``, drop the duplicate.
      3. Sorted-merge the smaller side's array into the larger side's
         array, dropping duplicates and self-references.  Result becomes
         ``new_root``'s array.  Relocate to the tail of ``cl_data`` if the
         merged size exceeds ``new_root``'s slot capacity.
      4. Mark the smaller (and absorbed-into-tail, if any) slots as gaps.
      5. Update ``comp_csize_arr`` for both sides.

    ``scratch`` is a shared int32 work buffer; caller pre-allocates and
    sizes it for the maximum possible merged-row length.

    The legacy ``_merge_cl_lists`` is called serially from the orchestrator
    (one merge at a time); same here, so the shared scratch is safe.

    Raises
    ------
    RuntimeError
        If a relocation would exceed the tail-pad budget of ``cl_data``
        allocated by ``_init_cl_arrays``.  Not observed in the n=1M
        production sweep; surfaces a sizing bug loudly rather than
        corrupting the structure.
    """
    size_new = comp_csize_arr[new_root]
    size_old = comp_csize_arr[old_root]

    if size_old == 0:
        ## No work — old root has no partners.  Belt-and-suspenders cleanup.
        comp_csize_arr[old_root] = np.int32(0)
        cl_arr_capacity[old_root] = np.int32(0)
        return

    new_root_start = cl_arr_start[new_root]
    old_root_start = cl_arr_start[old_root]

    ## ── Step 2: rewrite every partner that referred to ``old_root`` ──────
    ## The DSU survivor is ``new_root``; ``old_root`` is the dying identity
    ## that ``ds_find`` will never return again.  Every component P that
    ## had ``old_root`` as a CL partner must be rewritten to ``new_root``
    ## immediately, otherwise its array goes stale and binary search
    ## misses real conflicts.
    ##
    ## Note: we rewrite the dying side's partners regardless of relative
    ## size.  The amortized cost is still O(M log n) because each CL pair
    ## (u, v) sits in exactly two partner arrays at any moment, and a
    ## given pair's external reference can be rewritten at most O(log n)
    ## times across the whole Borůvka run (component lineage depth).
    for k in range(size_old):
        p = cl_data[old_root_start + np.int32(k)]
        if p == new_root or p == old_root:
            ## Self-reference for the merged component — skip.  Will be
            ## stripped by the sorted-merge step below.
            continue
        new_size_p = _sorted_array_replace(
            cl_arr_start[p], comp_csize_arr[p],
            old_root, new_root, cl_data,
        )
        comp_csize_arr[p] = new_size_p

    ## ── Step 3: sorted-merge old_root's array into new_root's array ──────
    ## Two-pointer merge over the two sorted slots.  Strip entries equal
    ## to ``new_root`` or ``old_root`` (the merged component is its own
    ## self-reference — must drop).  Output goes to ``scratch`` first
    ## then is copied into the destination slot below.
    i_s = np.int32(0)
    i_l = np.int32(0)
    out = np.int32(0)
    smaller_size = size_old
    larger_size = size_new
    smaller_start = old_root_start
    larger_start = new_root_start
    while i_s < smaller_size and i_l < larger_size:
        v_s = cl_data[smaller_start + i_s]
        v_l = cl_data[larger_start + i_l]
        if v_s < v_l:
            chosen = v_s
            i_s += np.int32(1)
        elif v_s > v_l:
            chosen = v_l
            i_l += np.int32(1)
        else:
            chosen = v_s
            i_s += np.int32(1)
            i_l += np.int32(1)
        if chosen == new_root or chosen == old_root:
            continue
        scratch[out] = chosen
        out += np.int32(1)
    while i_s < smaller_size:
        v_s = cl_data[smaller_start + i_s]
        i_s += np.int32(1)
        if v_s == new_root or v_s == old_root:
            continue
        scratch[out] = v_s
        out += np.int32(1)
    while i_l < larger_size:
        v_l = cl_data[larger_start + i_l]
        i_l += np.int32(1)
        if v_l == new_root or v_l == old_root:
            continue
        scratch[out] = v_l
        out += np.int32(1)

    merged_size = out

    ## ── Step 4: write merged row into new_root's slot, relocating if needed ─
    ## We want the merged row to live under ``new_root`` (the DSU survivor).
    ## The most efficient destination is whichever slot has enough capacity.
    ## Prefer ``new_root``'s existing slot if it fits; else relocate to
    ## the tail of ``cl_data``.
    new_root_cap = cl_arr_capacity[new_root]
    if merged_size <= new_root_cap:
        ## Fits in new_root's existing slot.  Write directly.
        dst = cl_arr_start[new_root]
        for k in range(int(merged_size)):
            cl_data[dst + np.int32(k)] = scratch[k]
    else:
        ## Overflow — relocate to tail with 2× pad for future absorption.
        new_cap = np.int32(2) * merged_size
        if new_cap < np.int32(8):
            new_cap = np.int32(8)
        tail = next_free_arr[0]
        ## Numba-friendly capacity check.  If tripped, the caller's pad
        ## estimate (in ``_init_cl_arrays``) was too small; surface loudly.
        if tail + np.int64(new_cap) > np.int64(len(cl_data)):
            raise RuntimeError(
                "fast_hdbscan: cl_data tail buffer exhausted during merge; "
                "increase pad in _init_cl_arrays."
            )
        for k in range(int(merged_size)):
            cl_data[tail + np.int64(k)] = scratch[k]
        cl_arr_start[new_root] = np.int32(tail)
        cl_arr_capacity[new_root] = new_cap
        next_free_arr[0] = tail + np.int64(new_cap)

    comp_csize_arr[new_root] = merged_size
    comp_csize_arr[old_root] = np.int32(0)
    cl_arr_capacity[old_root] = np.int32(0)


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _select_per_vertex_cl_sorted(distances, indices, indptr, point_components,
                                  cl_data, cl_arr_start, comp_csize_arr,
                                  predecessors, vert_best_dst, vert_best_wt,
                                  vert_n_edges_examined, vert_n_cl_checks,
                                  vert_n_cl_walk_steps, vert_n_edges_rejected):
    """
    Sorted-array sibling of ``_select_per_vertex_cl``.

    Same prange / outputs / counter shapes as the legacy variant; only the
    inner conflict-check helper changes.
    """
    n = len(point_components)
    for v in numba.prange(n):
        from_component = point_components[v]
        start = indptr[v]
        end = indptr[v + 1]

        n_examined = np.int64(0)
        n_cl_checks = np.int64(0)
        n_cl_walk = np.int64(0)
        n_rejected = np.int64(0)

        for idx in range(start, end):
            neighbor = indices[idx]
            if neighbor == -1:
                break
            n_examined += np.int64(1)
            distance = distances[idx]
            to_component = point_components[neighbor]
            if to_component == from_component:
                continue

            # O(1) root lookup via compressed snapshot
            root_from = predecessors[from_component]
            root_to = predecessors[to_component]

            n_cl_checks += np.int64(1)
            conflict, walk_steps = _check_cl_conflict_sorted(
                root_from, root_to, cl_data, cl_arr_start, comp_csize_arr,
            )
            n_cl_walk += walk_steps
            if conflict:
                n_rejected += np.int64(1)
                continue

            # First valid = cheapest (rows sorted ascending)
            vert_best_dst[v] = neighbor
            vert_best_wt[v] = distance
            break

        vert_n_edges_examined[v] = n_examined
        vert_n_cl_checks[v] = n_cl_checks
        vert_n_cl_walk_steps[v] = n_cl_walk
        vert_n_edges_rejected[v] = n_rejected


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _select_per_vertex_cl(distances, indices, indptr, point_components,
                           comp_head, comp_csize, pool_vertex, pool_next,
                           predecessors, vert_best_dst, vert_best_wt,
                           vert_n_edges_examined, vert_n_cl_checks,
                           vert_n_cl_walk_steps, vert_n_edges_rejected):
    """
    Phase 1 (parallel): each vertex finds its cheapest CL-safe outgoing edge.

    Writes to per-vertex arrays vert_best_dst[v] and vert_best_wt[v].
    -1 / inf means no valid edge found.  All inputs are read-only.

    Counter arrays (each length n, int64) are written per-vertex (safe for prange):
        vert_n_edges_examined[v]   — CSR neighbors scanned (including same-comp)
        vert_n_cl_checks[v]        — times _check_cl_conflict was called
        vert_n_cl_walk_steps[v]    — linked-list pool steps across all CL checks
        vert_n_edges_rejected[v]   — edges rejected due to CL conflict
    """
    n = len(point_components)
    for v in numba.prange(n):
        from_component = point_components[v]
        start = indptr[v]
        end = indptr[v + 1]

        n_examined = np.int64(0)
        n_cl_checks = np.int64(0)
        n_cl_walk = np.int64(0)
        n_rejected = np.int64(0)

        for idx in range(start, end):
            neighbor = indices[idx]
            if neighbor == -1:
                break
            n_examined += np.int64(1)
            distance = distances[idx]
            to_component = point_components[neighbor]
            if to_component == from_component:
                continue

            # O(1) root lookup via compressed snapshot
            root_from = predecessors[from_component]
            root_to = predecessors[to_component]

            n_cl_checks += np.int64(1)
            conflict, walk_steps = _check_cl_conflict(
                root_from, root_to, comp_head, comp_csize,
                pool_vertex, pool_next, predecessors,
            )
            n_cl_walk += walk_steps
            if conflict:
                n_rejected += np.int64(1)
                continue

            # First valid = cheapest (rows sorted ascending)
            vert_best_dst[v] = neighbor
            vert_best_wt[v] = distance
            break

        vert_n_edges_examined[v] = n_examined
        vert_n_cl_checks[v] = n_cl_checks
        vert_n_cl_walk_steps[v] = n_cl_walk
        vert_n_edges_rejected[v] = n_rejected


@numba.njit(cache=NUMBA_CACHE)
def select_components_cl_sorted(distances, indices, indptr, point_components,
                                  cl_data, cl_arr_start, comp_csize_arr,
                                  predecessors):
    """
    Sorted-array variant of ``select_components_cl``.

    Identical contract; only the per-vertex CL conflict check uses the
    sorted/deduped partner-component arrays instead of the legacy linked
    list pool.  See ``_check_cl_conflict_sorted``.
    """
    n = len(point_components)

    # Phase 1: parallel per-vertex scan
    vert_best_dst = np.full(n, -1, dtype=np.int32)
    vert_best_wt = np.full(n, np.inf, dtype=np.float32)
    _vert_cnt_examined = np.zeros(n, dtype=np.int64)
    _vert_cnt_checks   = np.zeros(n, dtype=np.int64)
    _vert_cnt_walk     = np.zeros(n, dtype=np.int64)
    _vert_cnt_rejected = np.zeros(n, dtype=np.int64)
    _select_per_vertex_cl_sorted(distances, indices, indptr, point_components,
                                   cl_data, cl_arr_start, comp_csize_arr,
                                   predecessors, vert_best_dst, vert_best_wt,
                                   _vert_cnt_examined, _vert_cnt_checks,
                                   _vert_cnt_walk, _vert_cnt_rejected)

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
    # Allocate throwaway counter arrays — select_components_cl is the
    # parity path (used inside boruvka_mst_cl); counters are discarded here.
    _vert_cnt_examined = np.zeros(n, dtype=np.int64)
    _vert_cnt_checks   = np.zeros(n, dtype=np.int64)
    _vert_cnt_walk     = np.zeros(n, dtype=np.int64)
    _vert_cnt_rejected = np.zeros(n, dtype=np.int64)
    _select_per_vertex_cl(distances, indices, indptr, point_components,
                           comp_head, comp_csize, pool_vertex, pool_next,
                           predecessors, vert_best_dst, vert_best_wt,
                           _vert_cnt_examined, _vert_cnt_checks,
                           _vert_cnt_walk, _vert_cnt_rejected)

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
                        adj_edge_idx, bfs_queue, bfs_parent_edge, bfs_visited,
                        bfs_tie_fix):
    """
    BFS in the merge tree from comp_u to comp_v.  Returns the index of the
    heaviest alive edge on the path, or -1 if no path exists, plus the count
    of BFS edges traversed (adjacency-list steps, including dead edges skipped).

    adj_head/adj_next/adj_edge_idx form a flat adjacency list built from
    alive tree edges.

    The ``bfs_tie_fix`` flag controls tie-breaking on equal-weight edges along
    the path:
      * ``0`` — current behavior: strict ``>`` (first tied edge encountered
        during the comp_v→comp_u backtrack wins; direction-sensitive).
      * ``1`` — direction-invariant: on equality, the lowest edge index wins.

    Returns
    -------
    heaviest_idx (int32): index of heaviest edge on path, -1 if not found.
    n_bfs_edges (int64): total adjacency-list steps taken during BFS.
    """
    # Reset visited
    bfs_visited[comp_u] = True
    bfs_parent_edge[comp_u] = -1

    q_front = 0
    q_back = 0
    bfs_queue[q_back] = comp_u
    q_back += 1

    n_bfs_edges = np.int64(0)
    found = False
    while q_front < q_back:
        node = bfs_queue[q_front]
        q_front += 1

        # Iterate adjacency list
        e = adj_head[node]
        while e >= 0:
            n_bfs_edges += np.int64(1)
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
        return np.int32(-1), n_bfs_edges

    # Backtrack from comp_v to comp_u, find heaviest edge
    heaviest_idx = np.int32(-1)
    heaviest_wt = np.float32(-1.0)
    cur = comp_v
    while cur != comp_u:
        ei = bfs_parent_edge[cur]
        ## Selection rule:
        ##   bfs_tie_fix==0 (default) — strict greater-than, first encountered wins on ties.
        ##   bfs_tie_fix==1 — on equality, prefer the lowest edge index (direction-invariant).
        if (tree_wt[ei] > heaviest_wt) or (
            bfs_tie_fix != 0
            and tree_wt[ei] == heaviest_wt
            and (heaviest_idx < 0 or ei < heaviest_idx)
        ):
            heaviest_wt = tree_wt[ei]
            heaviest_idx = np.int32(ei)
        if tree_src[ei] == cur:
            cur = tree_dst[ei]
        else:
            cur = tree_src[ei]

    # Clean up visited
    for i in range(q_back):
        bfs_visited[bfs_queue[i]] = False

    return heaviest_idx, n_bfs_edges


@numba.njit(cache=NUMBA_CACHE)
def validate_and_prune_merges(candidate_src, candidate_dst, candidate_wt,
                               n_candidates, cl_indices, cl_indptr,
                               point_components, n_verts,
                               _adj_head, _adj_next, _adj_edge_idx,
                               _bfs_queue, _bfs_parent_edge, _bfs_visited,
                               _temp_parent, dedup_mode, bfs_tie_fix):
    """
    Steps (b)+(c): tentative merge then cleanup of transitive CL violations.

    Builds a temporary DSU from candidate edges, then scans CL pairs for
    violations (scoped to vertices whose components participate in merges).
    For each violation, BFS in the merge tree finds the heaviest edge on the
    path between the two violating pre-round components and marks it for
    removal.  Repeats until clean.

    Scratch arrays (_adj_head, etc.) are pre-allocated by the caller to avoid
    repeated allocation across rounds.

    Parameters controlling experimental dedup / tie-break behavior
    --------------------------------------------------------------
    dedup_mode : int32
        0 — no dedup (current behavior).
        1 — ordered-pair dedup, key = ``comp_u * (n_verts+1) + comp_v``
            (preserves direction; both ``(A,B)`` and ``(B,A)`` enter BFS).
            Bit-identical to ``dedup_mode=0`` regardless of ``bfs_tie_fix``.
        2 — canonical-pair dedup, key = ``min(comp_u, comp_v) * (n_verts+1)
            + max(comp_u, comp_v)``. *Must* be paired with
            ``bfs_tie_fix=1`` to guarantee bit-identical output to the
            tie-fix-only baseline; otherwise tie-direction sensitivity in
            the BFS produces a different MST.

    bfs_tie_fix : int32
        0 — current behavior (strict ``>`` in BFS heaviest-edge selection;
            direction-sensitive on tied weights).
        1 — direction-invariant tie-break (lowest edge index wins on equality).

    Returns
    -------
    surviving_src, surviving_dst, surviving_wt : arrays of surviving edges
    n_surviving : int
    counters : int64[:] length 7 — [n_scan_iterations, n_cl_pairs_scanned,
        n_violations_found_total, n_bfs_edges_traversed, n_edges_cut,
        n_cl_pairs_pre_filter, n_unique_violation_pairs]
    """
    ## counters[0] = n_scan_iterations
    ## counters[1] = n_cl_pairs_scanned (post-filter, reached BFS-eligibility)
    ## counters[2] = n_violations_found_total
    ## counters[3] = n_bfs_edges_traversed
    ## counters[4] = n_edges_cut
    ## counters[5] = n_cl_pairs_pre_filter (raw CSR walk volume per inner body)
    ## counters[6] = n_unique_violation_pairs (post-dedup; equals counters[1] when dedup_mode==0)
    counters = np.zeros(7, dtype=np.int64)

    if n_candidates == 0:
        return candidate_src[:0], candidate_dst[:0], candidate_wt[:0], 0, counters

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

    ## Dedup dict — allocated once, cleared per outer iteration.
    ## Even when dedup_mode==0 the empty dict is cheap; the inner loop simply
    ## never inserts/looks up.  Keeping the alloc unconditional keeps numba
    ## type-inference simple (Optional[Dict] is awkward).
    pair_stride = np.int64(n_verts) + np.int64(1)
    dedup_seen = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.int8,
    )

    for _iteration in range(max_iters):
        counters[0] += np.int64(1)   ## n_scan_iterations
        if dedup_mode != 0:
            dedup_seen.clear()

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
                counters[5] += np.int64(1)   ## n_cl_pairs_pre_filter (raw CSR walk volume)
                v = cl_indices[p]
                if v <= u:
                    continue

                comp_u = point_components[u]
                comp_v = point_components[v]
                if comp_u == comp_v:
                    continue

                if _temp_parent[comp_u] != _temp_parent[comp_v]:
                    continue

                counters[1] += np.int64(1)   ## n_cl_pairs_scanned (reached BFS-eligibility)

                ## Optional dedup: skip the BFS if this component-pair was
                ## already processed this iteration.  Two key encodings:
                ##  * dedup_mode==1 → ordered (preserves direction; both
                ##    (A,B) and (B,A) get separate BFS calls).
                ##  * dedup_mode==2 → canonical (min,max); requires
                ##    bfs_tie_fix=1 for bit-identity to a stable reference.
                if dedup_mode == 1:
                    pair_key = np.int64(comp_u) * pair_stride + np.int64(comp_v)
                    if pair_key in dedup_seen:
                        continue
                    dedup_seen[pair_key] = np.int8(1)
                elif dedup_mode == 2:
                    if comp_u < comp_v:
                        pair_key = np.int64(comp_u) * pair_stride + np.int64(comp_v)
                    else:
                        pair_key = np.int64(comp_v) * pair_stride + np.int64(comp_u)
                    if pair_key in dedup_seen:
                        continue
                    dedup_seen[pair_key] = np.int8(1)

                counters[6] += np.int64(1)   ## n_unique_violation_pairs (post-dedup BFS triggers)
                heaviest, n_bfs_steps = _bfs_heaviest_edge(
                    comp_u, comp_v, tree_src, tree_dst, tree_wt,
                    tree_alive, n_candidates, _adj_head, _adj_next,
                    _adj_edge_idx, _bfs_queue, _bfs_parent_edge, _bfs_visited,
                    bfs_tie_fix,
                )
                counters[3] += n_bfs_steps   ## n_bfs_edges_traversed
                if heaviest >= 0:
                    edges_to_remove[heaviest] = True
                    found_violation = True
                    counters[2] += np.int64(1)   ## n_violations_found_total

        if not found_violation:
            break

        for i in range(n_candidates):
            if edges_to_remove[i]:
                tree_alive[i] = False
                counters[4] += np.int64(1)   ## n_edges_cut

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

    return surv_src, surv_dst, surv_wt, n_surviving, counters


@numba.njit(cache=NUMBA_CACHE)
def boruvka_mst_cl(graph, cl_indices, cl_indptr, band_fraction=np.inf,
                    overwrite=False, dedup_mode=2, bfs_tie_fix=1,
                    cl_struct_mode=1):
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
    dedup_mode : int (default 2)
        BFS dedup mode — see ``validate_and_prune_merges``.
        0 = no dedup; 1 = ordered-pair dedup (Option D); 2 = canonical-pair
        dedup (Option A — must be paired with ``bfs_tie_fix=1``).
        Default 2 (Option A) ships canonical-pair dedup; toggles 0/1 are
        retained as fallback / benchmark scaffold.
    bfs_tie_fix : int (default 1)
        0 = legacy direction-sensitive ``>`` in BFS; 1 = direction-invariant
        tie-break (lowest edge index wins on equality).  Default 1 pairs
        with ``dedup_mode=2`` for the canonical-pair dedup path.
    cl_struct_mode : int (default 1)
        Selects the CL conflict-tracking data structure used during the
        per-round select phase:
          0 = legacy linked-list pool (``_init_cl_pool`` + ``_check_cl_conflict``).
          1 = sorted-array variant (``_init_cl_arrays`` + ``_check_cl_conflict_sorted``).
        Both paths are functionally equivalent; mode 1 amortizes per-check
        cost from O(M) walk steps to O(log M) binary search at the price
        of small-to-larger rewrite-on-merge.  Default 1 (sorted-array) is
        the production path (~2.6× faster than legacy at n=50k); mode 0
        is retained as fallback / benchmarking.

    Returns
    -------
    n_components : int
    point_components : int32[:] shape (n,)
    mst_edges : float64[:, 3]
    n_rounds : int
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

    ## ── Allocate both CL data structures so JIT can branch at call sites. ──
    ## Whichever is unused is a tiny placeholder (n_init_pool=1).  Numba
    ## requires consistent types across both branches of an ``if`` so we
    ## can't conditionally allocate; the unused side is essentially free.
    if cl_struct_mode == 1:
        ## Real sorted-array structure; placeholder linked-list pool.
        cl_data, cl_arr_start, cl_arr_capacity, comp_csize_arr, next_free_arr = (
            _init_cl_arrays(cl_indices, cl_indptr, n)
        )
        ## Scratch buffer for sorted-merge — sized to the worst-case merged row.
        ## Total live partner IDs across all components <= 2*M (symmetric CL),
        ## and a single merge row's length is bounded by (n - 1).  Use the
        ## smaller of those.  ``len(cl_indices)`` is the upper bound at init.
        scratch_size = np.int64(len(cl_indices)) + np.int64(2)
        if scratch_size < np.int64(2 * n):
            cl_scratch = np.empty(scratch_size, dtype=np.int32)
        else:
            cl_scratch = np.empty(np.int64(2 * n), dtype=np.int32)
        ## Placeholder linked-list pool.
        pool_vertex = np.empty(1, dtype=np.int32)
        pool_next = np.empty(1, dtype=np.int32)
        comp_head = np.empty(1, dtype=np.int32)
        comp_tail = np.empty(1, dtype=np.int32)
        comp_csize = np.empty(1, dtype=np.int32)
    else:
        ## Legacy linked-list pool; placeholder sorted-array structure.
        pool_vertex, pool_next, comp_head, comp_tail, comp_csize = _init_cl_pool(
            cl_indices, cl_indptr, n
        )
        cl_data = np.empty(1, dtype=np.int32)
        cl_arr_start = np.empty(1, dtype=np.int32)
        cl_arr_capacity = np.empty(1, dtype=np.int32)
        comp_csize_arr = np.empty(1, dtype=np.int32)
        next_free_arr = np.empty(1, dtype=np.int64)
        cl_scratch = np.empty(1, dtype=np.int32)

    edges_list = [np.empty((0, 3), dtype=np.float64) for _ in range(0)]
    use_banding = band_fraction < 1e30  # avoid inf comparisons
    n_rounds = np.int32(0)

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
        if cl_struct_mode == 1:
            cand_src, cand_dst, cand_wt, n_cand = select_components_cl_sorted(
                distances, indices, indptr, point_components,
                cl_data, cl_arr_start, comp_csize_arr,
                compressed_roots,
            )
        else:
            cand_src, cand_dst, cand_wt, n_cand = select_components_cl(
                distances, indices, indptr, point_components,
                comp_head, comp_csize, pool_vertex, pool_next,
                compressed_roots,
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
        # counters discarded in the parity-reference JIT path
        surv_src, surv_dst, surv_wt, n_surviving, _cnt = validate_and_prune_merges(
            cand_src, cand_dst, cand_wt, n_cand,
            cl_indices, cl_indptr, point_components, n,
            _adj_head, _adj_next, _adj_edge_idx,
            _bfs_queue, _bfs_parent_edge, _bfs_visited, _temp_parent,
            dedup_mode, bfs_tie_fix,
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

                # Merge CL structure
                if cl_struct_mode == 1:
                    _merge_cl_arrays(
                        new_root, old_root, cl_data, cl_arr_start,
                        cl_arr_capacity, comp_csize_arr, next_free_arr,
                        cl_scratch,
                    )
                else:
                    _merge_cl_lists(new_root, old_root, comp_head, comp_tail,
                                    comp_csize, pool_next)

        if n_added == 0:
            break

        edges_list.append(new_edges[:n_added])
        update_point_components(disjoint_set, point_components)
        update_graph_components(distances, indices, indptr, point_components)
        n_components -= n_added
        n_rounds += np.int32(1)

    counter = 0
    num_edges = sum([edges.shape[0] for edges in edges_list])
    result = np.empty((num_edges, 3), dtype=np.float64)
    for edges in edges_list:
        result[counter : counter + edges.shape[0]] = edges
        counter += edges.shape[0]
    return n_components, point_components, result, n_rounds


# ---------------------------------------------------------------------------
# Pyloop instrumentation helpers  (new — do not modify boruvka_mst_cl above)
# ---------------------------------------------------------------------------

@numba.njit(cache=NUMBA_CACHE)
def _snapshot_compressed_roots(parent, compressed_roots, n):
    """
    Phase 1 helper: take a full-path-compression snapshot of DSU roots.

    Writes into pre-allocated ``compressed_roots[i]`` the compressed root of
    vertex ``i``.  Called once per Borůvka round from the Python orchestrator.

    Args:
        parent (int32[:]): DSU parent array (``disjoint_set.parent``).
        compressed_roots (int32[:]): output array, length n.
        n (int): number of vertices.
    """
    for i in range(n):
        root = parent[i]
        while parent[root] != root:
            root = parent[root]
        compressed_roots[i] = root


@numba.njit(cache=NUMBA_CACHE)
def _reduce_per_component(point_components, vert_best_dst, vert_best_wt, n):
    """
    Phase 2b (serial O(n)): reduce per-vertex select results to per-component minimum.

    Counterpart to the parallel ``_select_per_vertex_cl`` pass.  Together they
    implement ``select_components_cl``; splitting them lets the pyloop orchestrator
    time the parallel prange and serial reduction independently.

    Args:
        point_components (int32[:]): current component label per vertex.
        vert_best_dst (int32[:]): best destination per vertex (-1 = none).
        vert_best_wt (float32[:]): best weight per vertex (inf = none).
        n (int): number of vertices.

    Returns:
        out_src (int32[:]): source vertex of cheapest outgoing edge per component.
        out_dst (int32[:]): destination vertex.
        out_wt  (float32[:]): edge weight.
        n_edges (int): number of components with a valid edge.
    """
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
def _commit_surviving_edges(
    surv_src, surv_dst, surv_wt, n_surviving,
    ds_parent, ds_rank,
    comp_head, comp_tail, comp_csize, pool_next,
):
    """
    Phase 4 helper: commit surviving edges into main DSU and CL lists.

    Iterates over surviving edges, unions the two component roots via
    union-by-rank, merges their CL conflict lists (O(1) pointer surgery),
    and accumulates accepted edges.  Must remain JIT because it's a tight
    find/union loop.

    Args:
        surv_src (int32[:]): source vertices of surviving candidates.
        surv_dst (int32[:]): destination vertices.
        surv_wt  (float32[:]): edge weights.
        n_surviving (int): number of surviving candidates.
        ds_parent (int32[:]): DSU parent array (mutated in-place).
        ds_rank   (int32[:]): DSU rank array (mutated in-place).
        comp_head / comp_tail / comp_csize / pool_next: CL pool arrays.

    Returns:
        new_edges (float64[:, 3]): accepted MSF edges (src, dst, wt).
        n_added (int): number of edges actually added (after duplicate-root skip).
    """
    new_edges = np.empty((n_surviving, 3), dtype=np.float64)
    n_added = 0

    for i in range(n_surviving):
        src = surv_src[i]
        dst = surv_dst[i]

        # Full path-compression find
        root_src = src
        while ds_parent[root_src] != root_src:
            root_src = ds_parent[root_src]
        cur = src
        while cur != root_src:
            nxt = ds_parent[cur]
            ds_parent[cur] = root_src
            cur = nxt

        root_dst = dst
        while ds_parent[root_dst] != root_dst:
            root_dst = ds_parent[root_dst]
        cur = dst
        while cur != root_dst:
            nxt = ds_parent[cur]
            ds_parent[cur] = root_dst
            cur = nxt

        if root_src == root_dst:
            continue

        new_edges[n_added, 0] = np.float64(src)
        new_edges[n_added, 1] = np.float64(dst)
        new_edges[n_added, 2] = np.float64(surv_wt[i])
        n_added += 1

        # Union-by-rank
        if ds_rank[root_src] > ds_rank[root_dst]:
            new_root = root_src
            old_root = root_dst
        elif ds_rank[root_src] < ds_rank[root_dst]:
            new_root = root_dst
            old_root = root_src
        else:
            new_root = root_src
            old_root = root_dst
            ds_rank[new_root] += 1
        ds_parent[old_root] = new_root

        # O(1) CL list merge
        _merge_cl_lists(new_root, old_root, comp_head, comp_tail,
                        comp_csize, pool_next)

    return new_edges, n_added


@numba.njit(cache=NUMBA_CACHE)
def _commit_surviving_edges_sorted(
    surv_src, surv_dst, surv_wt, n_surviving,
    ds_parent, ds_rank,
    cl_data, cl_arr_start, cl_arr_capacity, comp_csize_arr, next_free_arr,
    cl_scratch,
):
    """
    Sorted-array sibling of ``_commit_surviving_edges``.

    Same semantics — iterate surviving edges, do full-path-compression find,
    union-by-rank, and merge CL state.  The only difference is the merge
    helper called: ``_merge_cl_arrays`` instead of ``_merge_cl_lists``.

    Returns
    -------
    new_edges : float64[:, 3]
        Accepted MSF edges (src, dst, wt).  First ``n_added`` rows are valid.
    n_added : int
        Number of edges actually added (after duplicate-root skip).
    """
    new_edges = np.empty((n_surviving, 3), dtype=np.float64)
    n_added = 0

    for i in range(n_surviving):
        src = surv_src[i]
        dst = surv_dst[i]

        # Full path-compression find
        root_src = src
        while ds_parent[root_src] != root_src:
            root_src = ds_parent[root_src]
        cur = src
        while cur != root_src:
            nxt = ds_parent[cur]
            ds_parent[cur] = root_src
            cur = nxt

        root_dst = dst
        while ds_parent[root_dst] != root_dst:
            root_dst = ds_parent[root_dst]
        cur = dst
        while cur != root_dst:
            nxt = ds_parent[cur]
            ds_parent[cur] = root_dst
            cur = nxt

        if root_src == root_dst:
            continue

        new_edges[n_added, 0] = np.float64(src)
        new_edges[n_added, 1] = np.float64(dst)
        new_edges[n_added, 2] = np.float64(surv_wt[i])
        n_added += 1

        # Union-by-rank
        if ds_rank[root_src] > ds_rank[root_dst]:
            new_root = root_src
            old_root = root_dst
        elif ds_rank[root_src] < ds_rank[root_dst]:
            new_root = root_dst
            old_root = root_src
        else:
            new_root = root_src
            old_root = root_dst
            ds_rank[new_root] += 1
        ds_parent[old_root] = new_root

        # Sorted-array CL merge (rewrite-on-merge + dedup)
        _merge_cl_arrays(
            new_root, old_root, cl_data, cl_arr_start, cl_arr_capacity,
            comp_csize_arr, next_free_arr, cl_scratch,
        )

    return new_edges, n_added


def boruvka_mst_cl_pyloop(
    graph,
    cl_indices,
    cl_indptr,
    band_fraction=np.inf,
    overwrite=False,
    collect_timings=False,
    dedup_mode=2,
    bfs_tie_fix=1,
    cl_struct_mode=1,
):
    """
    Python-orchestrated Borůvka MST with cannot-link constraints.

    Mirrors ``boruvka_mst_cl`` exactly but lifts the round loop out of the
    JIT into Python so that per-phase wall-clock can be measured with
    ``time.perf_counter()``.  Each phase is still a JIT call; the Python
    loop body itself does ~zero numeric work.

    When ``collect_timings=True``, returns a timing summary dict alongside
    the normal outputs.  When ``False`` (default), returns the same 3-tuple
    as ``boruvka_mst_cl`` (no overhead from timing infrastructure).

    This function is a *sibling* — it does NOT modify ``boruvka_mst_cl``.

    Args:
        graph (CoreGraph):
            Namedtuple (weights, distances, indices, indptr).
        cl_indices (int32[:]):
            CSR column indices of symmetric cannot-link graph.
        cl_indptr (int32[:]):
            CSR row pointers (length n+1).
        band_fraction (float):
            Passed to band-filter logic.  ``np.inf`` = no banding.
        overwrite (bool):
            If False (default), copies graph arrays before mutation.
        collect_timings (bool):
            If True, accumulate per-phase and per-round timers and return them.

    Returns:
        n_components (int),
        point_components (int32[:]),
        mst_edges (float64[:, 3]),
        timing_summary (dict)  — only present when ``collect_timings=True``.
    """
    import time as _time

    ## ── unpack graph (avoid repeated namedtuple boundary crossings) ────────
    distances = graph.weights
    indices = graph.indices
    indptr = graph.indptr
    if not overwrite:
        indices = indices.copy()
        distances = distances.copy()

    n = len(indptr) - 1

    ## ── DSU + component bookkeeping ────────────────────────────────────────
    disjoint_set = ds_rank_create(n)
    ds_parent = disjoint_set.parent   # int32[:] — arrays are mutable namedtuple fields
    ds_rank = disjoint_set.rank
    point_components = np.arange(n, dtype=np.int32)
    n_components = n

    ## ── CL conflict-tracking structure (mode 0 = legacy linked-list,
    ##     mode 1 = sorted-array). ──
    t_init_pool_start = _time.perf_counter()
    if cl_struct_mode == 1:
        cl_data, cl_arr_start, cl_arr_capacity, comp_csize_arr, next_free_arr = (
            _init_cl_arrays(cl_indices, cl_indptr, n)
        )
        scratch_size = max(int(len(cl_indices)) + 2, 2 * n)
        cl_scratch = np.empty(scratch_size, dtype=np.int32)
        ## Placeholders so the legacy variable names exist when the pyloop
        ## body falls through to non-mode-specific code paths.  Never read
        ## in mode 1.
        pool_vertex = np.empty(0, dtype=np.int32)
        pool_next = np.empty(0, dtype=np.int32)
        comp_head = np.empty(0, dtype=np.int32)
        comp_tail = np.empty(0, dtype=np.int32)
        comp_csize = np.empty(0, dtype=np.int32)
    else:
        pool_vertex, pool_next, comp_head, comp_tail, comp_csize = _init_cl_pool(
            cl_indices, cl_indptr, n
        )
        cl_data = np.empty(0, dtype=np.int32)
        cl_arr_start = np.empty(0, dtype=np.int32)
        cl_arr_capacity = np.empty(0, dtype=np.int32)
        comp_csize_arr = np.empty(0, dtype=np.int32)
        next_free_arr = np.empty(0, dtype=np.int64)
        cl_scratch = np.empty(0, dtype=np.int32)
    t_init_pool = _time.perf_counter() - t_init_pool_start

    use_banding = band_fraction < 1e30

    ## ── scratch arrays for validate_and_prune_merges ───────────────────────
    max_adj = 2 * n
    _adj_head = np.full(n, -1, dtype=np.int32)
    _adj_next = np.empty(max_adj, dtype=np.int32)
    _adj_edge_idx = np.empty(max_adj, dtype=np.int32)
    _bfs_queue = np.empty(n, dtype=np.int32)
    _bfs_parent_edge = np.full(n, -1, dtype=np.int32)
    _bfs_visited = np.zeros(n, dtype=np.bool_)
    _temp_parent = np.arange(n, dtype=np.int32)

    ## ── compressed-root snapshot (reused each round) ───────────────────────
    compressed_roots = np.arange(n, dtype=np.int32)

    ## ── per-vertex scratch for split select ───────────────────────────────
    vert_best_dst = np.full(n, -1, dtype=np.int32)
    vert_best_wt = np.full(n, np.inf, dtype=np.float32)

    ## ── per-vertex counter arrays for _select_per_vertex_cl ───────────────
    ## Allocated once; reset to 0 each round before the prange call.
    ## Each prange iter writes only its own v-th slot → no race condition.
    vert_n_edges_examined = np.zeros(n, dtype=np.int64)
    vert_n_cl_checks      = np.zeros(n, dtype=np.int64)
    vert_n_cl_walk_steps  = np.zeros(n, dtype=np.int64)
    vert_n_edges_rejected = np.zeros(n, dtype=np.int64)

    ## ── timing / counter accumulators ──────────────────────────────────────
    rounds_t_snapshot = []
    rounds_t_select_parallel = []    # _select_per_vertex_cl (prange)
    rounds_t_select_reduce = []      # _reduce_per_component (serial)
    rounds_t_select = []             # total select (sum of above two)
    rounds_t_validate = []
    rounds_t_commit = []
    rounds_t_update = []
    rounds_n_cand = []
    rounds_n_surviving = []
    rounds_n_added = []
    ## select-phase per-round counters (aggregated from per-vertex arrays)
    rounds_n_edges_examined = []
    rounds_n_cl_checks      = []
    rounds_n_cl_walk_steps  = []
    rounds_n_edges_rejected = []
    ## validate-phase per-round counters (from validate_and_prune_merges)
    rounds_n_scan_iterations       = []
    rounds_n_cl_pairs_scanned      = []
    rounds_n_violations_found      = []
    rounds_n_bfs_edges_traversed   = []
    rounds_n_edges_cut             = []
    ## new dedup pilot counters (counters[5], counters[6])
    rounds_n_cl_pairs_pre_filter   = []
    rounds_n_unique_violation_pairs = []

    edges_list = []
    t_wall_start = _time.perf_counter()

    ## ── Borůvka round loop ─────────────────────────────────────────────────
    while n_components > 1:

        ## Phase 1: compressed-root snapshot
        t0 = _time.perf_counter()
        _snapshot_compressed_roots(ds_parent, compressed_roots, n)
        t_snapshot = _time.perf_counter() - t0

        ## Phase 2a: parallel per-vertex CL-safe scan
        vert_best_dst[:] = -1
        vert_best_wt[:] = np.inf
        vert_n_edges_examined[:] = 0
        vert_n_cl_checks[:] = 0
        vert_n_cl_walk_steps[:] = 0
        vert_n_edges_rejected[:] = 0
        t0 = _time.perf_counter()
        if cl_struct_mode == 1:
            _select_per_vertex_cl_sorted(
                distances, indices, indptr, point_components,
                cl_data, cl_arr_start, comp_csize_arr,
                compressed_roots, vert_best_dst, vert_best_wt,
                vert_n_edges_examined, vert_n_cl_checks,
                vert_n_cl_walk_steps, vert_n_edges_rejected,
            )
        else:
            _select_per_vertex_cl(
                distances, indices, indptr, point_components,
                comp_head, comp_csize, pool_vertex, pool_next,
                compressed_roots, vert_best_dst, vert_best_wt,
                vert_n_edges_examined, vert_n_cl_checks,
                vert_n_cl_walk_steps, vert_n_edges_rejected,
            )
        t_select_parallel = _time.perf_counter() - t0
        ## Aggregate per-vertex counters → per-round scalars
        round_n_edges_examined = int(vert_n_edges_examined.sum())
        round_n_cl_checks      = int(vert_n_cl_checks.sum())
        round_n_cl_walk_steps  = int(vert_n_cl_walk_steps.sum())
        round_n_edges_rejected = int(vert_n_edges_rejected.sum())

        ## Phase 2b: serial reduction to per-component minimum
        t0 = _time.perf_counter()
        cand_src, cand_dst, cand_wt, n_cand = _reduce_per_component(
            point_components, vert_best_dst, vert_best_wt, n
        )
        t_select_reduce = _time.perf_counter() - t0
        t_select = t_select_parallel + t_select_reduce

        if n_cand == 0:
            rounds_t_snapshot.append(t_snapshot)
            rounds_t_select_parallel.append(t_select_parallel)
            rounds_t_select_reduce.append(t_select_reduce)
            rounds_t_select.append(t_select)
            rounds_t_validate.append(0.0)
            rounds_t_commit.append(0.0)
            rounds_t_update.append(0.0)
            rounds_n_cand.append(0)
            rounds_n_surviving.append(0)
            rounds_n_added.append(0)
            rounds_n_edges_examined.append(round_n_edges_examined)
            rounds_n_cl_checks.append(round_n_cl_checks)
            rounds_n_cl_walk_steps.append(round_n_cl_walk_steps)
            rounds_n_edges_rejected.append(round_n_edges_rejected)
            rounds_n_scan_iterations.append(0)
            rounds_n_cl_pairs_scanned.append(0)
            rounds_n_violations_found.append(0)
            rounds_n_bfs_edges_traversed.append(0)
            rounds_n_edges_cut.append(0)
            rounds_n_cl_pairs_pre_filter.append(0)
            rounds_n_unique_violation_pairs.append(0)
            break

        ## Phase 3: band filtering (Python; only active when band_fraction < inf)
        if use_banding:
            w_min = float(cand_wt.min())
            band_hi = w_min * (1.0 + band_fraction)
            mask = cand_wt <= band_hi
            cand_src = cand_src[mask]
            cand_dst = cand_dst[mask]
            cand_wt = cand_wt[mask]
            n_cand = int(mask.sum())

        ## Phase 4: tentative merge + CL validation + pruning
        t0 = _time.perf_counter()
        surv_src, surv_dst, surv_wt, n_surviving, validate_counters = validate_and_prune_merges(
            cand_src, cand_dst, cand_wt, n_cand,
            cl_indices, cl_indptr, point_components, n,
            _adj_head, _adj_next, _adj_edge_idx,
            _bfs_queue, _bfs_parent_edge, _bfs_visited, _temp_parent,
            dedup_mode, bfs_tie_fix,
        )
        t_validate = _time.perf_counter() - t0

        if n_surviving == 0:
            rounds_t_snapshot.append(t_snapshot)
            rounds_t_select_parallel.append(t_select_parallel)
            rounds_t_select_reduce.append(t_select_reduce)
            rounds_t_select.append(t_select)
            rounds_t_validate.append(t_validate)
            rounds_t_commit.append(0.0)
            rounds_t_update.append(0.0)
            rounds_n_cand.append(n_cand)
            rounds_n_surviving.append(0)
            rounds_n_added.append(0)
            rounds_n_edges_examined.append(round_n_edges_examined)
            rounds_n_cl_checks.append(round_n_cl_checks)
            rounds_n_cl_walk_steps.append(round_n_cl_walk_steps)
            rounds_n_edges_rejected.append(round_n_edges_rejected)
            rounds_n_scan_iterations.append(int(validate_counters[0]))
            rounds_n_cl_pairs_scanned.append(int(validate_counters[1]))
            rounds_n_violations_found.append(int(validate_counters[2]))
            rounds_n_bfs_edges_traversed.append(int(validate_counters[3]))
            rounds_n_edges_cut.append(int(validate_counters[4]))
            rounds_n_cl_pairs_pre_filter.append(int(validate_counters[5]))
            rounds_n_unique_violation_pairs.append(int(validate_counters[6]))
            break

        ## Phase 5: commit surviving edges to main DSU + CL lists
        t0 = _time.perf_counter()
        if cl_struct_mode == 1:
            new_edges, n_added = _commit_surviving_edges_sorted(
                surv_src, surv_dst, surv_wt, n_surviving,
                ds_parent, ds_rank,
                cl_data, cl_arr_start, cl_arr_capacity,
                comp_csize_arr, next_free_arr, cl_scratch,
            )
        else:
            new_edges, n_added = _commit_surviving_edges(
                surv_src, surv_dst, surv_wt, n_surviving,
                ds_parent, ds_rank,
                comp_head, comp_tail, comp_csize, pool_next,
            )
        t_commit = _time.perf_counter() - t0

        if n_added == 0:
            rounds_t_snapshot.append(t_snapshot)
            rounds_t_select_parallel.append(t_select_parallel)
            rounds_t_select_reduce.append(t_select_reduce)
            rounds_t_select.append(t_select)
            rounds_t_validate.append(t_validate)
            rounds_t_commit.append(t_commit)
            rounds_t_update.append(0.0)
            rounds_n_cand.append(n_cand)
            rounds_n_surviving.append(n_surviving)
            rounds_n_added.append(0)
            rounds_n_edges_examined.append(round_n_edges_examined)
            rounds_n_cl_checks.append(round_n_cl_checks)
            rounds_n_cl_walk_steps.append(round_n_cl_walk_steps)
            rounds_n_edges_rejected.append(round_n_edges_rejected)
            rounds_n_scan_iterations.append(int(validate_counters[0]))
            rounds_n_cl_pairs_scanned.append(int(validate_counters[1]))
            rounds_n_violations_found.append(int(validate_counters[2]))
            rounds_n_bfs_edges_traversed.append(int(validate_counters[3]))
            rounds_n_edges_cut.append(int(validate_counters[4]))
            rounds_n_cl_pairs_pre_filter.append(int(validate_counters[5]))
            rounds_n_unique_violation_pairs.append(int(validate_counters[6]))
            break

        edges_list.append(new_edges[:n_added])

        ## Phase 6: update point and graph components
        t0 = _time.perf_counter()
        update_point_components(disjoint_set, point_components)
        update_graph_components(distances, indices, indptr, point_components)
        t_update = _time.perf_counter() - t0

        n_components -= n_added

        ## Accumulate per-round counters + timers
        rounds_t_snapshot.append(t_snapshot)
        rounds_t_select_parallel.append(t_select_parallel)
        rounds_t_select_reduce.append(t_select_reduce)
        rounds_t_select.append(t_select)
        rounds_t_validate.append(t_validate)
        rounds_t_commit.append(t_commit)
        rounds_t_update.append(t_update)
        rounds_n_cand.append(n_cand)
        rounds_n_surviving.append(n_surviving)
        rounds_n_added.append(n_added)
        rounds_n_edges_examined.append(round_n_edges_examined)
        rounds_n_cl_checks.append(round_n_cl_checks)
        rounds_n_cl_walk_steps.append(round_n_cl_walk_steps)
        rounds_n_edges_rejected.append(round_n_edges_rejected)
        rounds_n_scan_iterations.append(int(validate_counters[0]))
        rounds_n_cl_pairs_scanned.append(int(validate_counters[1]))
        rounds_n_violations_found.append(int(validate_counters[2]))
        rounds_n_bfs_edges_traversed.append(int(validate_counters[3]))
        rounds_n_edges_cut.append(int(validate_counters[4]))
        rounds_n_cl_pairs_pre_filter.append(int(validate_counters[5]))
        rounds_n_unique_violation_pairs.append(int(validate_counters[6]))

    ## ── result assembly ─────────────────────────────────────────────────────
    t_assemble_start = _time.perf_counter()
    counter = 0
    num_edges = sum(e.shape[0] for e in edges_list)
    result = np.empty((num_edges, 3), dtype=np.float64)
    for edges in edges_list:
        result[counter : counter + edges.shape[0]] = edges
        counter += edges.shape[0]
    t_assemble = _time.perf_counter() - t_assemble_start

    t_wall_total = _time.perf_counter() - t_wall_start

    ## ── build timing summary ────────────────────────────────────────────────
    import numpy as _np

    def _sum(lst):
        return float(sum(lst)) if lst else 0.0

    timing_summary = {
        ## boundary timers
        "t_init_pool_s": t_init_pool,
        "t_assemble_s": t_assemble,
        "t_wall_total_s": t_wall_total,
        ## per-phase totals
        "t_snapshot_total_s": _sum(rounds_t_snapshot),
        "t_select_parallel_total_s": _sum(rounds_t_select_parallel),
        "t_select_reduce_total_s": _sum(rounds_t_select_reduce),
        "t_select_total_s": _sum(rounds_t_select),
        "t_validate_total_s": _sum(rounds_t_validate),
        "t_commit_total_s": _sum(rounds_t_commit),
        "t_update_total_s": _sum(rounds_t_update),
        ## per-round lists (list of floats)
        "rounds_t_snapshot_s": rounds_t_snapshot,
        "rounds_t_select_parallel_s": rounds_t_select_parallel,
        "rounds_t_select_reduce_s": rounds_t_select_reduce,
        "rounds_t_select_s": rounds_t_select,
        "rounds_t_validate_s": rounds_t_validate,
        "rounds_t_commit_s": rounds_t_commit,
        "rounds_t_update_s": rounds_t_update,
        ## per-round counters (original)
        "rounds_n_cand": rounds_n_cand,
        "rounds_n_surviving": rounds_n_surviving,
        "rounds_n_added": rounds_n_added,
        ## per-round select-phase counters (_select_per_vertex_cl)
        "rounds_n_edges_examined": rounds_n_edges_examined,
        "rounds_n_cl_checks": rounds_n_cl_checks,
        "rounds_n_cl_walk_steps": rounds_n_cl_walk_steps,
        "rounds_n_edges_rejected": rounds_n_edges_rejected,
        ## per-round validate-phase counters (validate_and_prune_merges)
        "rounds_n_scan_iterations": rounds_n_scan_iterations,
        "rounds_n_cl_pairs_scanned": rounds_n_cl_pairs_scanned,
        "rounds_n_violations_found": rounds_n_violations_found,
        "rounds_n_bfs_edges_traversed": rounds_n_bfs_edges_traversed,
        "rounds_n_edges_cut": rounds_n_edges_cut,
        ## new dedup pilot counters (counters[5], counters[6])
        "rounds_n_cl_pairs_pre_filter": rounds_n_cl_pairs_pre_filter,
        "rounds_n_unique_violation_pairs": rounds_n_unique_violation_pairs,
        ## dedup-mode / tie-fix flags echoed into the summary (for run logs)
        "dedup_mode": int(dedup_mode),
        "bfs_tie_fix": int(bfs_tie_fix),
        ## scalar totals for convenience
        "n_rounds": len(rounds_n_added),
        "total_n_edges_examined": sum(rounds_n_edges_examined),
        "total_n_cl_checks": sum(rounds_n_cl_checks),
        "total_n_cl_walk_steps": sum(rounds_n_cl_walk_steps),
        "total_n_edges_rejected": sum(rounds_n_edges_rejected),
        "total_n_scan_iterations": sum(rounds_n_scan_iterations),
        "total_n_cl_pairs_scanned": sum(rounds_n_cl_pairs_scanned),
        "total_n_violations_found": sum(rounds_n_violations_found),
        "total_n_bfs_edges_traversed": sum(rounds_n_bfs_edges_traversed),
        "total_n_edges_cut": sum(rounds_n_edges_cut),
        "total_n_cl_pairs_pre_filter": sum(rounds_n_cl_pairs_pre_filter),
        "total_n_unique_violation_pairs": sum(rounds_n_unique_violation_pairs),
    }

    if collect_timings:
        return n_components, point_components, result, timing_summary
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
