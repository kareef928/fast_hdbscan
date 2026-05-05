"""
Per-component parallel Phase 1 for constrained Boruvka.

Key idea: parallelize over *active components* (one thread per component) instead
of per-vertex. For each component:

  1. Walk its CL linked list ONCE to build a deduped set of forbidden target
     component-roots (the roots its constraints currently resolve to).
  2. Scan every vertex in the component to find the cheapest outgoing edge
     whose target root is NOT in that forbidden set.

This replaces the per-edge linked-list walk in `_select_per_vertex_cl` with a
short linear scan over a small per-component cache. It also enables an early
break per CSR row when `d >= component_best_so_far` (rows are sorted ascending).

Correctness rests on the same invariant the baseline relies on: the cl_indices
CSR is symmetric, so walking only the source component's list is sufficient
(every constraint pair appears in both endpoints' lists).
"""

from __future__ import annotations

import numpy as np
import numba

from .core_graph import (
    update_point_components,
    update_graph_components,
    _init_cl_pool,
    _merge_cl_lists,
    validate_and_prune_merges,
)
from .disjoint_set import ds_rank_create, ds_find


NUMBA_CACHE = False  # avoid stale cache during development


@numba.njit(cache=NUMBA_CACHE)
def _build_comp_grouping(compressed_roots, comp_csize, n,
                         active_roots_out, comp_v_indptr_out,
                         comp_v_list_out, forbidden_off_out,
                         root_to_active, cursor):
    """Group vertices by their (compressed) root. All outputs caller-allocated.

    Returns n_active (number of distinct live components this round).
    """
    n_active = 0
    for i in range(n):
        r = compressed_roots[i]
        if root_to_active[r] < 0:
            root_to_active[r] = n_active
            active_roots_out[n_active] = r
            n_active += 1

    for i in range(n_active + 1):
        comp_v_indptr_out[i] = 0
    for i in range(n):
        ai = root_to_active[compressed_roots[i]]
        comp_v_indptr_out[ai + 1] += 1
    for i in range(n_active):
        comp_v_indptr_out[i + 1] += comp_v_indptr_out[i]

    for i in range(n_active):
        cursor[i] = comp_v_indptr_out[i]
    for i in range(n):
        ai = root_to_active[compressed_roots[i]]
        comp_v_list_out[cursor[ai]] = i
        cursor[ai] += 1

    forbidden_off_out[0] = 0
    for i in range(n_active):
        forbidden_off_out[i + 1] = (
            forbidden_off_out[i] + comp_csize[active_roots_out[i]]
        )

    # reset scratch for next round
    for i in range(n_active):
        root_to_active[active_roots_out[i]] = -1

    return n_active


@numba.njit(parallel=True, cache=NUMBA_CACHE)
def _select_per_component_cl_opt(
        distances, indices, indptr,
        point_components, predecessors,
        comp_head, pool_vertex, pool_next,
        active_roots, n_active,
        comp_v_indptr, comp_v_list,
        forbidden_buf, forbidden_off,
        out_src, out_dst, out_wt):
    """Phase 1, per-component parallel.

    For each active root r in parallel:
      - Build deduped forbidden-target-roots list (one CL walk per component).
      - Scan all vertices in the component, picking the lightest legal edge.
    Writes one candidate per active component into out_src/out_dst/out_wt.
    """
    for ri in numba.prange(n_active):
        r = active_roots[ri]
        forb_base = forbidden_off[ri]

        # 1) build forbidden-target-roots set for this component
        n_forb = 0
        cur = comp_head[r]
        while cur >= 0:
            v_cl = pool_vertex[cur]
            t = predecessors[v_cl]
            while predecessors[t] != t:
                t = predecessors[t]
            if t != r:
                already = False
                for j in range(n_forb):
                    if forbidden_buf[forb_base + j] == t:
                        already = True
                        break
                if not already:
                    forbidden_buf[forb_base + n_forb] = t
                    n_forb += 1
            cur = pool_next[cur]

        # 2) scan every vertex in the component for the lightest legal edge
        best_w = np.float32(np.inf)
        best_src = np.int32(-1)
        best_dst = np.int32(-1)
        v_start = comp_v_indptr[ri]
        v_end = comp_v_indptr[ri + 1]
        for vi in range(v_start, v_end):
            v = comp_v_list[vi]
            i_start = indptr[v]
            i_end = indptr[v + 1]
            for idx in range(i_start, i_end):
                nbr = indices[idx]
                if nbr == -1:
                    break
                d = distances[idx]
                # CSR rows sorted ascending: once d >= best_w, no point continuing this row
                if d >= best_w:
                    break
                rt = predecessors[point_components[nbr]]
                forbidden = False
                for j in range(n_forb):
                    if forbidden_buf[forb_base + j] == rt:
                        forbidden = True
                        break
                if forbidden:
                    continue
                # First-valid in this row that beats best_w => row's best
                best_w = d
                best_src = np.int32(v)
                best_dst = np.int32(nbr)
                break

        out_src[ri] = best_src
        out_dst[ri] = best_dst
        out_wt[ri] = best_w


@numba.njit(cache=NUMBA_CACHE)
def _pack_candidates(out_src_full, out_dst_full, out_wt_full, n_active):
    """Drop -1 entries (components whose every outgoing edge was forbidden)."""
    n_keep = 0
    for i in range(n_active):
        if out_src_full[i] >= 0:
            n_keep += 1
    src = np.empty(n_keep, dtype=np.int32)
    dst = np.empty(n_keep, dtype=np.int32)
    wt = np.empty(n_keep, dtype=np.float32)
    j = 0
    for i in range(n_active):
        if out_src_full[i] >= 0:
            src[j] = out_src_full[i]
            dst[j] = out_dst_full[i]
            wt[j] = out_wt_full[i]
            j += 1
    return src, dst, wt, n_keep


def select_components_cl_opt(distances, indices, indptr, point_components,
                              comp_head, comp_csize, pool_vertex, pool_next,
                              predecessors,
                              cl_pool_size,
                              # caller-owned per-round scratch (length n / n+1):
                              _active_roots, _comp_v_indptr, _comp_v_list,
                              _forbidden_off, _forbidden_buf,
                              _root_to_active, _cursor,
                              _out_src_full, _out_dst_full, _out_wt_full):
    """Phase 1 wrapper. Same return signature as core_graph.select_components_cl."""
    n = len(point_components)
    n_active = _build_comp_grouping(
        predecessors, comp_csize, n,
        _active_roots, _comp_v_indptr, _comp_v_list,
        _forbidden_off, _root_to_active, _cursor,
    )
    _select_per_component_cl_opt(
        distances, indices, indptr,
        point_components, predecessors,
        comp_head, pool_vertex, pool_next,
        _active_roots, n_active,
        _comp_v_indptr, _comp_v_list,
        _forbidden_buf, _forbidden_off,
        _out_src_full, _out_dst_full, _out_wt_full,
    )
    return _pack_candidates(_out_src_full, _out_dst_full, _out_wt_full, n_active)


def boruvka_mst_cl_opt(graph, cl_indices, cl_indptr, band_fraction=np.inf,
                        overwrite=False):
    """Drop-in replacement for boruvka_mst_cl using per-component parallel Phase 1."""
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

    pool_vertex, pool_next, comp_head, comp_tail, comp_csize = _init_cl_pool(
        cl_indices, cl_indptr, n
    )
    cl_pool_size = max(len(cl_indices), 1)

    edges_list = []
    use_banding = band_fraction < 1e30
    n_rounds = np.int32(0)

    # Reused scratch for validate_and_prune_merges
    max_adj = 2 * n
    _adj_head = np.full(n, -1, dtype=np.int32)
    _adj_next = np.empty(max_adj, dtype=np.int32)
    _adj_edge_idx = np.empty(max_adj, dtype=np.int32)
    _bfs_queue = np.empty(n, dtype=np.int32)
    _bfs_parent_edge = np.full(n, -1, dtype=np.int32)
    _bfs_visited = np.zeros(n, dtype=np.bool_)
    _temp_parent = np.arange(n, dtype=np.int32)

    compressed_roots = np.arange(n, dtype=np.int32)

    # Reused scratch for optimized Phase 1
    _active_roots = np.empty(n, dtype=np.int32)
    _comp_v_indptr = np.empty(n + 1, dtype=np.int32)
    _comp_v_list = np.empty(n, dtype=np.int32)
    _forbidden_off = np.empty(n + 1, dtype=np.int32)
    _forbidden_buf = np.empty(cl_pool_size, dtype=np.int32)
    _root_to_active = np.full(n, -1, dtype=np.int32)
    _cursor = np.empty(n, dtype=np.int32)
    _out_src_full = np.empty(n, dtype=np.int32)
    _out_dst_full = np.empty(n, dtype=np.int32)
    _out_wt_full = np.empty(n, dtype=np.float32)

    while n_components > 1:
        for i in range(n):
            root = disjoint_set.parent[i]
            while disjoint_set.parent[root] != root:
                root = disjoint_set.parent[root]
            compressed_roots[i] = root

        cand_src, cand_dst, cand_wt, n_cand = select_components_cl_opt(
            distances, indices, indptr, point_components,
            comp_head, comp_csize, pool_vertex, pool_next,
            compressed_roots, cl_pool_size,
            _active_roots, _comp_v_indptr, _comp_v_list,
            _forbidden_off, _forbidden_buf,
            _root_to_active, _cursor,
            _out_src_full, _out_dst_full, _out_wt_full,
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
            cand_src, cand_dst, cand_wt, n_cand,
            cl_indices, cl_indptr, point_components, n,
            _adj_head, _adj_next, _adj_edge_idx,
            _bfs_queue, _bfs_parent_edge, _bfs_visited, _temp_parent,
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
                    new_root, old_root = root_src, root_dst
                elif disjoint_set.rank[root_src] < disjoint_set.rank[root_dst]:
                    new_root, old_root = root_dst, root_src
                else:
                    new_root, old_root = root_src, root_dst
                    disjoint_set.rank[new_root] += 1
                disjoint_set.parent[old_root] = new_root
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
    num_edges = sum(e.shape[0] for e in edges_list)
    result = np.empty((num_edges, 3), dtype=np.float64)
    for edges in edges_list:
        result[counter:counter + edges.shape[0]] = edges
        counter += edges.shape[0]
    return n_components, point_components, result, n_rounds
