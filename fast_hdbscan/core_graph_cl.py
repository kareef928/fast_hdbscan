"""Cannot-link constrained Boruvka MST helpers for CoreGraph inputs."""

import numba
import numpy as np

from .core_graph import update_point_components, update_graph_components
from .disjoint_set import ds_rank_create, ds_find
from .variables import NUMBA_CACHE


@numba.njit(cache=NUMBA_CACHE)
def _init_cl_arrays(cl_indices, cl_indptr, n_verts):
    """
    Build sorted, deduplicated cannot-link partner arrays for each component.
    """
    M = np.int64(len(cl_indices))
    n_verts_i64 = np.int64(n_verts)

    pad_per_row = np.int32(4)
    init_total = np.int64(0)
    for i in range(n_verts):
        row_len = cl_indptr[i + 1] - cl_indptr[i]
        cap = np.int32(2) * row_len
        if cap < pad_per_row:
            cap = pad_per_row
        init_total += np.int64(cap)

    tail_pad = np.int64(8) * M + n_verts_i64 * np.int64(4) + np.int64(1024)
    total_cap = init_total + tail_pad

    cl_data = np.zeros(total_cap, dtype=np.int32)
    cl_arr_start = np.empty(n_verts, dtype=np.int32)
    cl_arr_capacity = np.empty(n_verts, dtype=np.int32)
    comp_csize_arr = np.zeros(n_verts, dtype=np.int32)

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
            tmp = np.empty(row_len, dtype=np.int32)
            for k in range(row_len):
                tmp[k] = cl_indices[row_lo + k]
            tmp.sort()

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
def _check_cl_conflict_sorted(root_a, root_b, cl_data, cl_arr_start, comp_csize_arr):
    """
    Return True when two component roots have a cannot-link conflict.
    """
    size_a = comp_csize_arr[root_a]
    size_b = comp_csize_arr[root_b]
    if size_a == 0 or size_b == 0:
        return False

    if size_a <= size_b:
        small_root = root_a
        big_root = root_b
        small_size = size_a
    else:
        small_root = root_b
        big_root = root_a
        small_size = size_b

    start = cl_arr_start[small_root]
    target = np.int32(big_root)
    lo = np.int32(0)
    hi = small_size
    while lo < hi:
        mid = (lo + hi) >> np.int32(1)
        v = cl_data[start + mid]
        if v == target:
            return True
        if v < target:
            lo = mid + np.int32(1)
        else:
            hi = mid
    return False


@numba.njit(cache=NUMBA_CACHE)
def _sorted_array_replace(start, size, old_val, new_val, cl_data):
    """
    Replace ``old_val`` with ``new_val`` in a sorted array slot.
    """
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
        return size

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
        for k in range(int(old_pos), int(size) - 1):
            cl_data[start + np.int32(k)] = cl_data[start + np.int32(k + 1)]
        return size - np.int32(1)

    ins_pos = lo2
    if ins_pos > old_pos:
        ins_pos = ins_pos - np.int32(1)

    if ins_pos == old_pos:
        cl_data[start + old_pos] = new_val
    elif ins_pos < old_pos:
        k = old_pos
        while k > ins_pos:
            cl_data[start + k] = cl_data[start + np.int32(k - 1)]
            k -= np.int32(1)
        cl_data[start + ins_pos] = new_val
    else:
        k = old_pos
        while k < ins_pos:
            cl_data[start + k] = cl_data[start + np.int32(k + 1)]
            k += np.int32(1)
        cl_data[start + ins_pos] = new_val

    return size


@numba.njit(cache=NUMBA_CACHE)
def _merge_cl_arrays(
    new_root,
    old_root,
    cl_data,
    cl_arr_start,
    cl_arr_capacity,
    comp_csize_arr,
    next_free_arr,
    scratch,
):
    """
    Merge ``old_root`` cannot-link partners into ``new_root``.
    """
    size_new = comp_csize_arr[new_root]
    size_old = comp_csize_arr[old_root]

    if size_old == 0:
        comp_csize_arr[old_root] = np.int32(0)
        cl_arr_capacity[old_root] = np.int32(0)
        return

    new_root_start = cl_arr_start[new_root]
    old_root_start = cl_arr_start[old_root]

    for k in range(size_old):
        p = cl_data[old_root_start + np.int32(k)]
        if p == new_root or p == old_root:
            continue
        new_size_p = _sorted_array_replace(
            cl_arr_start[p], comp_csize_arr[p], old_root, new_root, cl_data
        )
        comp_csize_arr[p] = new_size_p

    i_s = np.int32(0)
    i_l = np.int32(0)
    out = np.int32(0)
    while i_s < size_old and i_l < size_new:
        v_s = cl_data[old_root_start + i_s]
        v_l = cl_data[new_root_start + i_l]
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

    while i_s < size_old:
        v_s = cl_data[old_root_start + i_s]
        i_s += np.int32(1)
        if v_s == new_root or v_s == old_root:
            continue
        scratch[out] = v_s
        out += np.int32(1)

    while i_l < size_new:
        v_l = cl_data[new_root_start + i_l]
        i_l += np.int32(1)
        if v_l == new_root or v_l == old_root:
            continue
        scratch[out] = v_l
        out += np.int32(1)

    merged_size = out
    new_root_cap = cl_arr_capacity[new_root]
    if merged_size <= new_root_cap:
        dst = cl_arr_start[new_root]
        for k in range(int(merged_size)):
            cl_data[dst + np.int32(k)] = scratch[k]
    else:
        new_cap = np.int32(2) * merged_size
        if new_cap < np.int32(8):
            new_cap = np.int32(8)
        tail = next_free_arr[0]
        if tail + np.int64(new_cap) > np.int64(len(cl_data)):
            raise RuntimeError(
                "fast_hdbscan: cl_data tail buffer exhausted during merge."
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
def _select_per_vertex_cl_sorted(
    distances,
    indices,
    indptr,
    point_components,
    cl_data,
    cl_arr_start,
    comp_csize_arr,
    predecessors,
    vert_best_dst,
    vert_best_wt,
):
    """
    Select the cheapest CL-safe outgoing edge for each vertex.
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

            root_from = predecessors[from_component]
            root_to = predecessors[to_component]

            if _check_cl_conflict_sorted(
                root_from, root_to, cl_data, cl_arr_start, comp_csize_arr
            ):
                continue

            vert_best_dst[v] = neighbor
            vert_best_wt[v] = distance
            break


@numba.njit(cache=NUMBA_CACHE)
def select_components_cl_sorted(
    distances,
    indices,
    indptr,
    point_components,
    cl_data,
    cl_arr_start,
    comp_csize_arr,
    predecessors,
):
    """
    Select the cheapest CL-safe outgoing edge for each component.
    """
    n = len(point_components)

    vert_best_dst = np.full(n, -1, dtype=np.int32)
    vert_best_wt = np.full(n, np.inf, dtype=np.float32)
    _select_per_vertex_cl_sorted(
        distances,
        indices,
        indptr,
        point_components,
        cl_data,
        cl_arr_start,
        comp_csize_arr,
        predecessors,
        vert_best_dst,
        vert_best_wt,
    )

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
def _bfs_heaviest_edge(
    comp_u,
    comp_v,
    tree_src,
    tree_dst,
    tree_wt,
    tree_alive,
    adj_head,
    adj_next,
    adj_edge_idx,
    bfs_queue,
    bfs_parent_edge,
    bfs_visited,
):
    """
    Return the heaviest alive edge on the path from ``comp_u`` to ``comp_v``.
    """
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

        e = adj_head[node]
        while e >= 0:
            ei = adj_edge_idx[e]
            if not tree_alive[ei]:
                e = adj_next[e]
                continue
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
        for i in range(q_back):
            bfs_visited[bfs_queue[i]] = False
        return np.int32(-1)

    heaviest_idx = np.int32(-1)
    heaviest_wt = np.float32(-1.0)
    cur = comp_v
    while cur != comp_u:
        ei = bfs_parent_edge[cur]
        if tree_wt[ei] > heaviest_wt or (
            tree_wt[ei] == heaviest_wt and (heaviest_idx < 0 or ei < heaviest_idx)
        ):
            heaviest_wt = tree_wt[ei]
            heaviest_idx = np.int32(ei)
        if tree_src[ei] == cur:
            cur = tree_dst[ei]
        else:
            cur = tree_src[ei]

    for i in range(q_back):
        bfs_visited[bfs_queue[i]] = False

    return heaviest_idx


@numba.njit(cache=NUMBA_CACHE)
def validate_and_prune_merges(
    candidate_src,
    candidate_dst,
    candidate_wt,
    n_candidates,
    cl_indices,
    cl_indptr,
    point_components,
    n_verts,
    _adj_head,
    _adj_next,
    _adj_edge_idx,
    _bfs_queue,
    _bfs_parent_edge,
    _bfs_visited,
    _temp_parent,
):
    """
    Validate tentative Boruvka merges and prune transitive CL violations.
    """
    if n_candidates == 0:
        return candidate_src[:0], candidate_dst[:0], candidate_wt[:0], 0

    tree_src = np.empty(n_candidates, dtype=np.int32)
    tree_dst = np.empty(n_candidates, dtype=np.int32)
    tree_wt = np.empty(n_candidates, dtype=np.float32)
    tree_alive = np.ones(n_candidates, dtype=np.bool_)

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

    involved_comps = involved_comps[:n_involved]
    involved_comps.sort()
    n_unique = 0
    for i in range(n_involved):
        if i == 0 or involved_comps[i] != involved_comps[i - 1]:
            involved_comps[n_unique] = involved_comps[i]
            n_unique += 1
    involved_comps = involved_comps[:n_unique]

    is_involved = _bfs_visited
    for i in range(n_unique):
        is_involved[involved_comps[i]] = True

    scan_verts = np.empty(n_verts, dtype=np.int32)
    n_scan = 0
    for u in range(n_verts):
        if is_involved[point_components[u]]:
            scan_verts[n_scan] = u
            n_scan += 1

    for i in range(n_unique):
        is_involved[involved_comps[i]] = False

    pair_stride = np.int64(n_verts) + np.int64(1)
    dedup_seen = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.int8,
    )

    max_iters = n_candidates
    for _iteration in range(max_iters):
        dedup_seen.clear()

        for i in range(n_unique):
            c = involved_comps[i]
            _temp_parent[c] = c
            _adj_head[c] = -1

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

        edges_to_remove = np.full(n_candidates, False, dtype=np.bool_)
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

                if comp_u < comp_v:
                    pair_key = np.int64(comp_u) * pair_stride + np.int64(comp_v)
                else:
                    pair_key = np.int64(comp_v) * pair_stride + np.int64(comp_u)
                if pair_key in dedup_seen:
                    continue
                dedup_seen[pair_key] = np.int8(1)

                heaviest = _bfs_heaviest_edge(
                    comp_u,
                    comp_v,
                    tree_src,
                    tree_dst,
                    tree_wt,
                    tree_alive,
                    _adj_head,
                    _adj_next,
                    _adj_edge_idx,
                    _bfs_queue,
                    _bfs_parent_edge,
                    _bfs_visited,
                )
                if heaviest >= 0:
                    edges_to_remove[heaviest] = True
                    found_violation = True

        if not found_violation:
            break

        for i in range(n_candidates):
            if edges_to_remove[i]:
                tree_alive[i] = False

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
def boruvka_mst_cl(graph, cl_indices, cl_indptr, band_fraction=np.inf, overwrite=False):
    """
    Boruvka MST with cannot-link constraints for precomputed core graphs.
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

    cl_data, cl_arr_start, cl_arr_capacity, comp_csize_arr, next_free_arr = (
        _init_cl_arrays(cl_indices, cl_indptr, n)
    )
    scratch_size = np.int64(len(cl_indices)) + np.int64(2)
    if scratch_size < np.int64(2 * n):
        cl_scratch = np.empty(scratch_size, dtype=np.int32)
    else:
        cl_scratch = np.empty(np.int64(2 * n), dtype=np.int32)

    edges_list = [np.empty((0, 3), dtype=np.float64) for _ in range(0)]
    use_banding = band_fraction < 1e30

    max_adj = 2 * n
    _adj_head = np.full(n, -1, dtype=np.int32)
    _adj_next = np.empty(max_adj, dtype=np.int32)
    _adj_edge_idx = np.empty(max_adj, dtype=np.int32)
    _bfs_queue = np.empty(n, dtype=np.int32)
    _bfs_parent_edge = np.full(n, -1, dtype=np.int32)
    _bfs_visited = np.zeros(n, dtype=np.bool_)
    _temp_parent = np.arange(n, dtype=np.int32)

    compressed_roots = np.arange(n, dtype=np.int32)

    while n_components > 1:
        for i in range(n):
            root = disjoint_set.parent[i]
            while disjoint_set.parent[root] != root:
                root = disjoint_set.parent[root]
            compressed_roots[i] = root

        cand_src, cand_dst, cand_wt, n_cand = select_components_cl_sorted(
            distances,
            indices,
            indptr,
            point_components,
            cl_data,
            cl_arr_start,
            comp_csize_arr,
            compressed_roots,
        )
        if n_cand == 0:
            break

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

                _merge_cl_arrays(
                    new_root,
                    old_root,
                    cl_data,
                    cl_arr_start,
                    cl_arr_capacity,
                    comp_csize_arr,
                    next_free_arr,
                    cl_scratch,
                )

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


