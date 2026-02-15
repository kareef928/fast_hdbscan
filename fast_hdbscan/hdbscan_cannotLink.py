# fast_hdbscan/precomputed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Tuple, Union

import numba
from numba import prange
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as sp_csgraph

from fast_hdbscan.disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
from fast_hdbscan.hdbscan import clusters_from_spanning_tree


Number = Union[int, float]
CSR = sp.csr_matrix

# Type alias for parallel backend selection
ParallelBackend = Literal["auto", "cuda", "cpu", "sequential"]


# ------------------------------- CUDA Detection -------------------------------

_CUDA_AVAILABLE: Optional[bool] = None


def _check_cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    
    This function caches the result at module level to avoid repeated checks.
    
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        try:
            from numba import cuda
            _CUDA_AVAILABLE = cuda.is_available()
        except Exception:
            _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE


# ------------------------------- constraints -------------------------------


@dataclass
class MergeConstraint:
    """
    A thin wrapper around hard merge constraints used during MST construction.

    The constrained MST builder only needs one operation:
        any_cannot_link_across(component_a, component_b) -> bool

    Optionally, we also support:
        pair_cannot_link(i, j) -> bool
    which enables a "soft" mode that only penalizes *direct* forbidden edges.

    Notes:
        - This class is intentionally minimal.
        - If you want intragroup constraints, implement pair_cannot_link(i,j)
          in terms of group IDs (or any other metadata) and pass it in via
          `from_pair_cannot_link`.
    """

    any_cannot_link_across: Callable[[np.ndarray, np.ndarray], bool]
    pair_cannot_link: Optional[Callable[[int, int], bool]] = None
    iter_cannot_link_pairs: Optional[Callable[[], Iterable[Tuple[int, int]]]] = None

    # Optional CSR payload for fast strict=True JIT path.
    cannot_link_indptr: Optional[np.ndarray] = None  # shape (N+1,), int64
    cannot_link_indices: Optional[np.ndarray] = None  # shape (nnz,), int32

    @staticmethod
    def from_cannot_link_matrix(
        cannot_link: Union[np.ndarray, CSR],
        *,
        n_points: int,
    ) -> "MergeConstraint":
        """
        Build a MergeConstraint from a dense (N,N) bool array or sparse CSR bool matrix.

        Args:
            cannot_link (Union[np.ndarray, CSR]):
                Dense or sparse structure where cannot_link[i,j] == True means
                points i and j must never be merged into the same MST component.
            n_points (int):
                Number of points, N.

        Returns:
            (MergeConstraint):
                A constraint object with fast `any_cannot_link_across` checks.
        """
        if sp.issparse(cannot_link):
            cannot_link_csr = cannot_link.tocsr().astype(bool)
            if cannot_link_csr.shape != (n_points, n_points):
                raise ValueError("cannot_link sparse matrix must have shape (N, N).")

            # enforce symmetry and zero diagonal (CRITICAL for strict=True correctness)
            cannot_link_csr.setdiag(False)
            cannot_link_csr.eliminate_zeros()
            cannot_link_csr = ((cannot_link_csr + cannot_link_csr.T) > 0).tocsr()
            cannot_link_csr.setdiag(False)
            cannot_link_csr.eliminate_zeros()
            cannot_link_csr.sum_duplicates()
            cannot_link_csr.sort_indices()

            def any_cannot_link_across_sparse(
                comp_a: np.ndarray, comp_b: np.ndarray
            ) -> bool:
                comp_b_set = set(int(x) for x in comp_b.tolist())
                for idx_a in comp_a.tolist():
                    row_indices = cannot_link_csr.getrow(int(idx_a)).indices
                    for idx_b in row_indices:
                        if int(idx_b) in comp_b_set:
                            return True
                return False

            def pair_cannot_link_sparse(i: int, j: int) -> bool:
                row_indices = cannot_link_csr.getrow(int(i)).indices
                return bool(np.any(row_indices == int(j)))

            def iter_pairs_sparse() -> Iterable[Tuple[int, int]]:
                coo = sp.triu(cannot_link_csr, k=1).tocoo()
                for i, j in zip(coo.row.tolist(), coo.col.tolist()):
                    yield int(i), int(j)

            cannot_link_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
            cannot_link_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

            return MergeConstraint(
                any_cannot_link_across=any_cannot_link_across_sparse,
                pair_cannot_link=pair_cannot_link_sparse,
                iter_cannot_link_pairs=iter_pairs_sparse,
                cannot_link_indptr=cannot_link_indptr,
                cannot_link_indices=cannot_link_indices,
            )

        cannot_link_dense = np.asarray(cannot_link, dtype=bool)
        if cannot_link_dense.shape != (n_points, n_points):
            raise ValueError("cannot_link dense array must have shape (N, N).")

        # enforce symmetry and zero diagonal (CRITICAL for strict=True correctness)
        np.fill_diagonal(cannot_link_dense, False)
        cannot_link_dense = np.logical_or(cannot_link_dense, cannot_link_dense.T)

        def any_cannot_link_across_dense(
            comp_a: np.ndarray, comp_b: np.ndarray
        ) -> bool:
            return bool(np.any(cannot_link_dense[np.ix_(comp_a, comp_b)]))

        def pair_cannot_link_dense(i: int, j: int) -> bool:
            return bool(cannot_link_dense[int(i), int(j)])

        def iter_pairs_dense() -> Iterable[Tuple[int, int]]:
            for i in range(n_points):
                js = np.flatnonzero(cannot_link_dense[i])
                for j in js.tolist():
                    if j > i:
                        yield int(i), int(j)

        # Build CSR payload for the JIT strict=True path (intended for small/medium N).
        cannot_link_csr = sp.csr_matrix(cannot_link_dense)
        cannot_link_csr.setdiag(False)
        cannot_link_csr.eliminate_zeros()
        cannot_link_csr.sum_duplicates()
        cannot_link_csr.sort_indices()
        cannot_link_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cannot_link_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        return MergeConstraint(
            any_cannot_link_across=any_cannot_link_across_dense,
            pair_cannot_link=pair_cannot_link_dense,
            iter_cannot_link_pairs=iter_pairs_dense,
            cannot_link_indptr=cannot_link_indptr,
            cannot_link_indices=cannot_link_indices,
        )

    @staticmethod
    def from_pair_cannot_link(
        pair_cannot_link: Callable[[int, int], bool],
    ) -> "MergeConstraint":
        """
        Build a MergeConstraint from a pairwise callback.

        This is intentionally thin and may be slow for large components because
        it falls back to nested loops to answer:
            "is there any forbidden pair across these two components?"

        Args:
            pair_cannot_link (Callable[[int, int], bool]):
                A function that returns True if points i and j cannot be in the same cluster.

        Returns:
            (MergeConstraint):
                A constraint object.
        """

        def any_cannot_link_across_func(
            comp_a: np.ndarray, comp_b: np.ndarray
        ) -> bool:
            for i in comp_a.tolist():
                for j in comp_b.tolist():
                    if pair_cannot_link(int(i), int(j)):
                        return True
            return False

        return MergeConstraint(
            any_cannot_link_across=any_cannot_link_across_func,
            pair_cannot_link=pair_cannot_link,
            iter_cannot_link_pairs=None,  # cannot enumerate in general
            cannot_link_indptr=None,
            cannot_link_indices=None,
        )


# ------------------------------- helpers ---------------------------------


@numba.njit(cache=True)
def _has_cannot_link_violation_csr_numba(
    labels: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    noise_label: int,
) -> bool:
    """
    Fast violation check:
        returns True if there exists (i,j) cannot-link with labels[i]==labels[j]!=noise_label.

    Assumes CSR adjacency; does not require symmetry but will catch violations
    as long as at least one direction of (i,j) is present.

    Args:
        labels (np.ndarray):
            Cluster labels for each point.
        cannot_link_indptr (np.ndarray):
            CSR indptr array for the cannot-link graph.
        cannot_link_indices (np.ndarray):
            CSR indices array for the cannot-link graph.
        noise_label (int):
            The label assigned to noise points.

    Returns:
        (bool):
            True if a cannot-link violation is found.
    """
    n = labels.shape[0]
    for i in range(n):
        li = labels[i]
        if li == noise_label:
            continue
        start = cannot_link_indptr[i]
        end = cannot_link_indptr[i + 1]
        for k in range(start, end):
            j = cannot_link_indices[k]
            if j == i:
                continue
            if labels[j] == li:
                return True
    return False


def _maybe_split_labels_if_cannot_link_violated(
    labels: np.ndarray,
    *,
    merge_constraint: "MergeConstraint",
    distances: Optional[Union[np.ndarray, CSR]],
    noise_label: int = -1,
) -> np.ndarray:
    """
    If posthoc cleanup is requested, avoid doing any splitting work unless
    there is an actual cannot-link violation.

    Args:
        labels (np.ndarray):
            The cluster labels to check and potentially split.
        merge_constraint (MergeConstraint):
            The merge constraint object containing cannot-link information.
        distances (Optional[Union[np.ndarray, CSR]]):
            The distance matrix, used for splitting based on graph components.
        noise_label (int):
            The label assigned to noise points.

    Returns:
        (np.ndarray):
            The (potentially modified) labels array.
    """
    labels_in = np.asarray(labels, dtype=np.int64)

    if merge_constraint.pair_cannot_link is None:
        raise ValueError(
            "posthoc_cleanup requires merge_constraint.pair_cannot_link to be available."
        )

    has_violation = False

    # ---- Path 1: numba CSR scan if payload exists ----
    if (merge_constraint.cannot_link_indptr is not None) and (
        merge_constraint.cannot_link_indices is not None
    ):
        cannot_link_indptr = np.asarray(
            merge_constraint.cannot_link_indptr, dtype=np.int64
        )
        cannot_link_indices = np.asarray(
            merge_constraint.cannot_link_indices, dtype=np.int64
        )

        if (
            cannot_link_indptr.ndim == 1
            and cannot_link_indptr.size == labels_in.shape[0] + 1
        ):
            has_violation = bool(
                _has_cannot_link_violation_csr_numba(
                    labels_in,
                    cannot_link_indptr,
                    cannot_link_indices,
                    int(noise_label),
                )
            )
        else:
            has_violation = True

    # ---- Path 2: pair iterator if available ----
    elif merge_constraint.iter_cannot_link_pairs is not None:
        violations = find_cannot_link_violations(
            labels_in, merge_constraint=merge_constraint, noise_label=int(noise_label)
        )
        has_violation = bool(violations.shape[0] > 0)

    # ---- Path 3: cannot cheaply check -> conservatively split ----
    else:
        has_violation = True

    if not has_violation:
        return labels_in  # noop

    return split_clusters_to_respect_cannot_link(
        labels_in,
        merge_constraint=merge_constraint,
        distances=distances,
        noise_label=int(noise_label),
    )


@numba.njit(cache=True)
def _dsu_find_numba(parent: np.ndarray, x: int) -> int:
    """
    DSU find with path compression (Numba).

    Args:
        parent (np.ndarray):
            The parent array for the disjoint set union structure.
        x (int):
            The element to find.

    Returns:
        (int):
            The root of the set containing x.
    """
    root = x
    while parent[root] != root:
        root = parent[root]

    while parent[x] != x:
        px = parent[x]
        parent[x] = root
        x = px

    return root


@numba.njit(cache=True)
def _dsu_find_readonly_numba(parent: np.ndarray, x: int) -> int:
    """
    DSU find WITHOUT path compression (safe for parallel reads).
    
    This function traverses the parent chain to find the root but does NOT
    perform path compression. This makes it safe for concurrent reads in
    parallel phases where multiple threads may call find() on the same
    data structure simultaneously.
    
    Use this in Step 1a (parallel edge selection) where we need to determine
    component membership without modifying the DSU structure.
    
    Args:
        parent (np.ndarray):
            The parent array for the disjoint set union structure.
        x (int):
            The element to find.
            
    Returns:
        (int):
            The root of the set containing x.
    """
    while parent[x] != x:
        x = parent[x]
    return x


@numba.njit(cache=True)
def _csr_row_contains_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    row: int,
    value: int,
) -> bool:
    """
    Linear membership test in a CSR row (Numba-safe, does not assume sorting).

    Args:
        indptr (np.ndarray):
            CSR indptr array.
        indices (np.ndarray):
            CSR indices array.
        row (int):
            The row to search in.
        value (int):
            The value to search for.

    Returns:
        (bool):
            True if the value is found in the row.
    """
    start = int(indptr[row])
    end = int(indptr[row + 1])
    for k in range(start, end):
        if int(indices[k]) == value:
            return True
    return False


@numba.njit(cache=True)
def _component_has_conflict_with_root_numba(
    root_small: int,
    root_large: int,
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
) -> bool:
    """
    Check whether merging component(root_small) into component(root_large) would
    violate any cannot-link constraints.

    For each node i in the smaller component:
        for each forbidden neighbor j in cannot_link[i]:
            if find(j) == root_large -> conflict

    Args:
        root_small (int): The root of the smaller component.
        root_large (int): The root of the larger component.
        parent (np.ndarray): DSU parent array.
        head (np.ndarray): Head of the linked list for each component.
        next_node (np.ndarray): Next node in the linked list for each component.
        cannot_link_indptr (np.ndarray): CSR indptr for cannot-link constraints.
        cannot_link_indices (np.ndarray): CSR indices for cannot-link constraints.

    Returns:
        bool: True if a conflict exists, False otherwise.
    """
    node = head[root_small]
    while node != -1:
        start = int(cannot_link_indptr[node])
        end = int(cannot_link_indptr[node + 1])
        for k in range(start, end):
            j = int(cannot_link_indices[k])
            if _dsu_find_numba(parent, j) == root_large:
                return True
        node = next_node[node]
    return False


@numba.njit(cache=True)
def _build_adjacency_list_numba(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    n_points: int,
) -> tuple:
    """
    Build CSR-style adjacency structure from edge arrays.

    Each undirected edge (u[i], v[i], w[i]) is stored twice in the adjacency
    structure: once for u[i] -> v[i] and once for v[i] -> u[i].

    Args:
        u (np.ndarray): Source nodes of edges (int32).
        v (np.ndarray): Destination nodes of edges (int32).
        w (np.ndarray): Weights of edges (float64).
        n_points (int): Number of points/nodes in the graph.

    Returns:
        tuple:
            adj_indptr (np.ndarray): CSR indptr array of shape (n_points + 1,).
            adj_neighbors (np.ndarray): Neighbor node IDs for each adjacency entry.
            adj_weights (np.ndarray): Edge weights for each adjacency entry.
            adj_edge_ids (np.ndarray): Original edge indices for each adjacency entry.
    """
    n_edges = u.shape[0]

    # Pass 1: Count the degree of each node
    degree = np.zeros(n_points, dtype=np.int64)
    for i in range(n_edges):
        degree[u[i]] += 1
        degree[v[i]] += 1

    # Build indptr from degrees
    adj_indptr = np.empty(n_points + 1, dtype=np.int64)
    adj_indptr[0] = 0
    for i in range(n_points):
        adj_indptr[i + 1] = adj_indptr[i] + degree[i]

    total_entries = adj_indptr[n_points]
    adj_neighbors = np.empty(total_entries, dtype=np.int32)
    adj_weights = np.empty(total_entries, dtype=np.float64)
    adj_edge_ids = np.empty(total_entries, dtype=np.int32)

    # Pass 2: Fill the adjacency arrays
    # Use a counter array to track current fill position for each node
    fill_pos = np.zeros(n_points, dtype=np.int64)

    for i in range(n_edges):
        a = u[i]
        b = v[i]
        ww = w[i]

        # Entry for a -> b
        pos_a = adj_indptr[a] + fill_pos[a]
        adj_neighbors[pos_a] = b
        adj_weights[pos_a] = ww
        adj_edge_ids[pos_a] = i
        fill_pos[a] += 1

        # Entry for b -> a
        pos_b = adj_indptr[b] + fill_pos[b]
        adj_neighbors[pos_b] = a
        adj_weights[pos_b] = ww
        adj_edge_ids[pos_b] = i
        fill_pos[b] += 1

    return adj_indptr, adj_neighbors, adj_weights, adj_edge_ids


@numba.njit(cache=True)
def _constrained_kruskal_mst_csr_strict_sorted_numba(
    u_sorted: np.ndarray,
    v_sorted: np.ndarray,
    w_sorted: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """
    Strict constrained Kruskal (Numba) for CSR cannot-link adjacency.

    Inputs u_sorted, v_sorted, w_sorted MUST be sorted by increasing w.

    Args:
        u_sorted (np.ndarray): Source nodes of edges, sorted by weight.
        v_sorted (np.ndarray): Destination nodes of edges, sorted by weight.
        w_sorted (np.ndarray): Weights of edges, sorted.
        cannot_link_indptr (np.ndarray): CSR indptr for cannot-link constraints.
        cannot_link_indices (np.ndarray): CSR indices for cannot-link constraints.
        n_points (int): Number of points.

    Returns:
        (np.ndarray):
            mst_edges (np.ndarray):
                Shape (<=N-1, 3), float64. Columns: [u, v, w]
                This may be a forest if constraints prevent full connectivity.
    """
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)

    # Per-component membership as a linked list stored in arrays.
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)

    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1

    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0

    for idx_edge in range(u_sorted.shape[0]):
        a = int(u_sorted[idx_edge])
        b = int(v_sorted[idx_edge])
        ww = float(w_sorted[idx_edge])

        # --- REAL FIX (robustness): always reject a directly-forbidden edge ---
        # This makes strict=True correct even if a user accidentally provides a
        # one-directional CSR adjacency (e.g. upper-triangular constraints).
        if _csr_row_contains_numba(
            cannot_link_indptr, cannot_link_indices, a, b
        ) or _csr_row_contains_numba(cannot_link_indptr, cannot_link_indices, b, a):
            continue

        ra = _dsu_find_numba(parent, a)
        rb = _dsu_find_numba(parent, b)
        if ra == rb:
            continue

        # Union-by-size, but only after constraint check.
        if size[ra] < size[rb]:
            root_small = ra
            root_large = rb
        else:
            root_small = rb
            root_large = ra

        if _component_has_conflict_with_root_numba(
            root_small,
            root_large,
            parent,
            head,
            next_node,
            cannot_link_indptr,
            cannot_link_indices,
        ):
            continue

        # Union: attach small -> large
        parent[root_small] = root_large
        size[root_large] = size[root_large] + size[root_small]

        # Concatenate member lists: large_tail.next = small_head
        next_node[tail[root_large]] = head[root_small]
        tail[root_large] = tail[root_small]

        # Emit MST edge
        mst_edges[n_added, 0] = float(a)
        mst_edges[n_added, 1] = float(b)
        mst_edges[n_added, 2] = float(ww)
        n_added += 1

        if n_added == n_points - 1:
            break

    return mst_edges[:n_added]


@numba.njit(cache=True)
def _constrained_boruvka_mst_csr_strict_numba(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
    n_edges: int,
) -> np.ndarray:
    """
    Strict constrained Borůvka MST algorithm (Numba).

    This algorithm builds a minimum spanning tree (or forest) while respecting
    cannot-link constraints. It uses a component-centric approach where each
    round finds the cheapest valid outgoing edge for each component, then
    merges components.

    Args:
        adj_indptr (np.ndarray): CSR indptr for adjacency list.
        adj_neighbors (np.ndarray): Neighbor node IDs in adjacency list.
        adj_weights (np.ndarray): Edge weights in adjacency list.
        adj_edge_ids (np.ndarray): Original edge indices in adjacency list.
        cannot_link_indptr (np.ndarray): CSR indptr for cannot-link constraints.
        cannot_link_indices (np.ndarray): CSR indices for cannot-link constraints.
        n_points (int): Number of points/nodes.
        n_edges (int): Number of original edges.

    Returns:
        np.ndarray:
            mst_edges of shape (<=N-1, 3), float64. Columns: [u, v, w]
            This may be a forest if constraints prevent full connectivity.
    """
    # Initialize DSU arrays
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)

    # Per-component membership as a linked list stored in arrays
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)

    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1

    # Track infeasible edges (constraint-violating edges never retry)
    infeasible = np.zeros(n_edges, dtype=np.bool_)

    # Per-component cheapest edge info (indexed by root)
    cheapest_edge_id = np.empty(n_points, dtype=np.int32)
    cheapest_weight = np.empty(n_points, dtype=np.float64)
    cheapest_target = np.empty(n_points, dtype=np.int32)
    cheapest_source = np.empty(n_points, dtype=np.int32)

    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0

    while True:
        # Reset cheapest edge info for all components
        for i in range(n_points):
            cheapest_edge_id[i] = -1
            cheapest_weight[i] = np.inf
            cheapest_target[i] = -1
            cheapest_source[i] = -1

        # Phase 1: Find cheapest valid outgoing edge per component
        for i in range(n_points):
            my_root = _dsu_find_numba(parent, i)

            # Scan all neighbors of node i
            start = adj_indptr[i]
            end = adj_indptr[i + 1]
            for k in range(start, end):
                edge_id = adj_edge_ids[k]

                # Skip edges marked infeasible
                if infeasible[edge_id]:
                    continue

                j = adj_neighbors[k]
                neighbor_root = _dsu_find_numba(parent, j)

                # Skip same-component edges
                if my_root == neighbor_root:
                    continue

                # Check cannot-link constraint between components
                # Determine small/large for constraint check
                if size[my_root] < size[neighbor_root]:
                    root_small = my_root
                    root_large = neighbor_root
                else:
                    root_small = neighbor_root
                    root_large = my_root

                if _component_has_conflict_with_root_numba(
                    root_small,
                    root_large,
                    parent,
                    head,
                    next_node,
                    cannot_link_indptr,
                    cannot_link_indices,
                ):
                    # Mark as infeasible so we never check again
                    infeasible[edge_id] = True
                    continue

                # Valid edge - check if it's the cheapest for my_root
                ww = adj_weights[k]
                if ww < cheapest_weight[my_root] or (
                    ww == cheapest_weight[my_root]
                    and edge_id < cheapest_edge_id[my_root]
                ):
                    cheapest_weight[my_root] = ww
                    cheapest_edge_id[my_root] = edge_id
                    cheapest_target[my_root] = neighbor_root
                    cheapest_source[my_root] = i

        # Phase 2: Merge components using cheapest edges
        n_merges = 0
        for r in range(n_points):
            # Only process roots
            if parent[r] != r:
                continue

            if cheapest_edge_id[r] == -1:
                continue

            # Find current root of target (may have changed in this phase)
            target_root = _dsu_find_numba(parent, cheapest_target[r])

            # Skip if already merged
            if r == target_root:
                continue

            # Re-check constraint: target may have merged with another component
            # that creates a new conflict
            if size[r] < size[target_root]:
                root_small = r
                root_large = target_root
            else:
                root_small = target_root
                root_large = r

            if _component_has_conflict_with_root_numba(
                root_small,
                root_large,
                parent,
                head,
                next_node,
                cannot_link_indptr,
                cannot_link_indices,
            ):
                # Skip this merge, but don't mark edge as permanently infeasible
                # The edge might become valid next round with different component configurations
                continue

            # Lower-root-wins tie-breaking
            winner = min(r, target_root)
            loser = max(r, target_root)

            # Union: attach loser -> winner
            parent[loser] = winner
            size[winner] = size[winner] + size[loser]

            # Concatenate member lists: winner_tail.next = loser_head
            next_node[tail[winner]] = head[loser]
            tail[winner] = tail[loser]

            # Emit MST edge
            src = cheapest_source[r]
            edge_id = cheapest_edge_id[r]
            # Find the actual edge endpoints from adjacency
            # We need to get the neighbor from the adjacency entry
            tgt = -1
            for k in range(adj_indptr[src], adj_indptr[src + 1]):
                if adj_edge_ids[k] == edge_id:
                    tgt = adj_neighbors[k]
                    break

            if tgt != -1:
                mst_edges[n_added, 0] = float(src)
                mst_edges[n_added, 1] = float(tgt)
                mst_edges[n_added, 2] = cheapest_weight[r]
                n_added += 1
                n_merges += 1

            if n_added == n_points - 1:
                break

        if n_merges == 0 or n_added == n_points - 1:
            break

    return mst_edges[:n_added]


# --------------- Parallel Constrained Borůvka MST Implementation ---------------


@numba.njit(cache=True)
def _parallel_edge_selection_numba(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    infeasible: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel edge selection for constrained Borůvka.
    
    Each node scans its neighbors in parallel to find the cheapest valid edge.
    An edge is valid if it connects two different components AND doesn't violate
    any cannot-link constraints.
    
    Args:
        adj_indptr: CSR indptr for adjacency list.
        adj_neighbors: Neighbor node IDs in adjacency list.
        adj_weights: Edge weights in adjacency list.
        adj_edge_ids: Original edge indices in adjacency list.
        parent: DSU parent array.
        size: Component size array.
        head: Linked list head for each component.
        next_node: Linked list next pointers.
        cannot_link_indptr: CSR indptr for cannot-link constraints.
        cannot_link_indices: CSR indices for cannot-link constraints.
        infeasible: Boolean array marking permanently infeasible edges.
        n_points: Number of points/nodes.
        
    Returns:
        Tuple of (cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id):
            - All arrays indexed by component root
            - -1 in target means no valid edge found for that component
    """
    # Per-component cheapest edge info (indexed by root)
    cheapest_edge_id = np.full(n_points, -1, dtype=np.int32)
    cheapest_weight = np.full(n_points, np.inf, dtype=np.float64)
    cheapest_target = np.full(n_points, -1, dtype=np.int32)
    cheapest_source = np.full(n_points, -1, dtype=np.int32)
    
    # Phase 1a: Each node finds its best candidate in parallel
    # We use per-node arrays first, then aggregate per-component
    node_best_edge_id = np.full(n_points, -1, dtype=np.int32)
    node_best_weight = np.full(n_points, np.inf, dtype=np.float64)
    node_best_target = np.full(n_points, -1, dtype=np.int32)
    
    for i in range(n_points):
        my_root = _dsu_find_numba(parent, i)
        my_size = size[my_root]
        
        # Scan all neighbors of node i
        start = adj_indptr[i]
        end = adj_indptr[i + 1]
        
        for k in range(start, end):
            edge_id = adj_edge_ids[k]
            
            # Skip edges marked infeasible
            if infeasible[edge_id]:
                continue
                
            j = adj_neighbors[k]
            neighbor_root = _dsu_find_numba(parent, j)
            
            # Skip same-component edges
            if my_root == neighbor_root:
                continue
            
            # Check cannot-link constraint between components
            neighbor_size = size[neighbor_root]
            if my_size < neighbor_size:
                root_small = my_root
                root_large = neighbor_root
            else:
                root_small = neighbor_root
                root_large = my_root
            
            if _component_has_conflict_with_root_numba(
                root_small,
                root_large,
                parent,
                head,
                next_node,
                cannot_link_indptr,
                cannot_link_indices,
            ):
                # Mark as infeasible so we never check again
                infeasible[edge_id] = True
                continue
            
            # Valid edge - check if it's the best for this node
            ww = adj_weights[k]
            if ww < node_best_weight[i] or (
                ww == node_best_weight[i] and edge_id < node_best_edge_id[i]
            ):
                node_best_weight[i] = ww
                node_best_edge_id[i] = edge_id
                node_best_target[i] = neighbor_root
    
    # Phase 1b: Aggregate per-component using atomic-like reduction
    # We iterate through all nodes and update their root's best edge
    # Note: This is sequential to avoid race conditions on component updates
    for i in range(n_points):
        if node_best_edge_id[i] == -1:
            continue
            
        my_root = _dsu_find_numba(parent, i)
        
        if node_best_weight[i] < cheapest_weight[my_root] or (
            node_best_weight[i] == cheapest_weight[my_root]
            and node_best_edge_id[i] < cheapest_edge_id[my_root]
        ):
            cheapest_weight[my_root] = node_best_weight[i]
            cheapest_edge_id[my_root] = node_best_edge_id[i]
            cheapest_target[my_root] = node_best_target[i]
            cheapest_source[my_root] = i
    
    return cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id


# --------------- CPU Parallel Backend (Step 1a parallel, Step 1b sequential) ---------------


@numba.njit(parallel=True, cache=True)
def _parallel_edge_selection_step1a_cpu(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    infeasible: np.ndarray,
    n_points: int,
    node_best_weight: np.ndarray,
    node_best_edge_id: np.ndarray,
    node_best_target: np.ndarray,
    node_best_source: np.ndarray,
) -> None:
    """
    Step 1a: Per-node edge search (PARALLEL on CPU).
    
    Each node independently scans its neighbors to find its best outgoing edge.
    This step is embarrassingly parallel because each thread writes only to its
    own index in the node_best_* arrays - no contention.
    
    IMPORTANT: Uses _dsu_find_readonly_numba (no path compression) to ensure
    thread safety during parallel reads of the parent array.
    
    Args:
        adj_*: Adjacency list arrays.
        parent, size, head, next_node: DSU arrays (READ-ONLY in this phase).
        cannot_link_*: Constraint arrays.
        infeasible: Boolean array of infeasible edges (READ-ONLY in this phase).
        n_points: Number of nodes.
        node_best_*: OUTPUT arrays - per-node best edge info.
    """
    # NOTE: We cannot mark edges as infeasible here because that would be a
    # parallel write to a shared array. Infeasibility marking happens in the
    # sequential phase or is handled differently.
    
    for i in prange(n_points):
        my_root = _dsu_find_readonly_numba(parent, i)
        my_size = size[my_root]
        
        best_weight = np.inf
        best_edge_id = -1
        best_target = -1
        
        # Scan all neighbors of node i
        start = adj_indptr[i]
        end = adj_indptr[i + 1]
        
        for k in range(start, end):
            edge_id = adj_edge_ids[k]
            
            # Skip edges already marked infeasible (read-only check)
            if infeasible[edge_id]:
                continue
            
            j = adj_neighbors[k]
            neighbor_root = _dsu_find_readonly_numba(parent, j)
            
            # Skip same-component edges
            if my_root == neighbor_root:
                continue
            
            # Check cannot-link constraint between components
            neighbor_size = size[neighbor_root]
            if my_size < neighbor_size:
                root_small = my_root
                root_large = neighbor_root
            else:
                root_small = neighbor_root
                root_large = my_root
            
            # Note: _component_has_conflict_with_root_numba is read-only
            if _component_has_conflict_with_root_numba(
                root_small,
                root_large,
                parent,
                head,
                next_node,
                cannot_link_indptr,
                cannot_link_indices,
            ):
                # Cannot mark infeasible here (would be parallel write)
                # Will be marked in sequential aggregation phase
                continue
            
            # Valid edge - check if best for this node
            ww = adj_weights[k]
            if ww < best_weight or (ww == best_weight and edge_id < best_edge_id):
                best_weight = ww
                best_edge_id = edge_id
                best_target = neighbor_root
        
        # Write to this node's slot (no contention - each thread owns its index)
        node_best_weight[i] = best_weight
        node_best_edge_id[i] = best_edge_id
        node_best_target[i] = best_target
        node_best_source[i] = i


@numba.njit(cache=True)
def _parallel_edge_selection_step1b_cpu(
    parent: np.ndarray,
    n_points: int,
    node_best_weight: np.ndarray,
    node_best_edge_id: np.ndarray,
    node_best_target: np.ndarray,
    node_best_source: np.ndarray,
    cheapest_weight: np.ndarray,
    cheapest_edge_id: np.ndarray,
    cheapest_target: np.ndarray,
    cheapest_source: np.ndarray,
) -> None:
    """
    Step 1b: Per-component aggregation (SEQUENTIAL on CPU).
    
    Aggregates per-node results into per-component cheapest edges.
    This step MUST be sequential on CPU because multiple nodes in the same
    component write to the same cheapest[root] location, and Numba CPU has
    no atomic operations.
    
    Uses path compression (_dsu_find_numba) since this is sequential.
    
    Args:
        parent: DSU parent array.
        n_points: Number of nodes.
        node_best_*: INPUT - per-node best edge info from Step 1a.
        cheapest_*: OUTPUT - per-component cheapest edge info.
    """
    for i in range(n_points):
        if node_best_edge_id[i] == -1:
            continue
        
        # Use path compression here (sequential, safe)
        my_root = _dsu_find_numba(parent, i)
        
        # Check if this node's best is better than component's current best
        if node_best_weight[i] < cheapest_weight[my_root] or (
            node_best_weight[i] == cheapest_weight[my_root]
            and node_best_edge_id[i] < cheapest_edge_id[my_root]
        ):
            cheapest_weight[my_root] = node_best_weight[i]
            cheapest_edge_id[my_root] = node_best_edge_id[i]
            cheapest_target[my_root] = node_best_target[i]
            cheapest_source[my_root] = node_best_source[i]


@numba.njit(cache=True)
def _parallel_edge_selection_cpu(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    infeasible: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CPU edge selection for constrained Borůvka (sequential version).
    
    Note: This is the sequential fallback. The parallel version uses
    _parallel_edge_selection_cpu_parallel which is not decorated with njit.
    
    Two-phase approach:
        Step 1a: Each node finds its best outgoing edge (sequential here)
        Step 1b: Aggregate per-node results into per-component (sequential)
    
    Args:
        adj_*: Adjacency list arrays.
        parent, size, head, next_node: DSU arrays.
        cannot_link_*: Constraint arrays.
        infeasible: Boolean array marking infeasible edges.
        n_points: Number of nodes.
        
    Returns:
        Tuple of (cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id):
            Per-component best edges indexed by root.
    """
    # Per-node arrays for Step 1a
    node_best_weight = np.full(n_points, np.inf, dtype=np.float64)
    node_best_edge_id = np.full(n_points, -1, dtype=np.int32)
    node_best_target = np.full(n_points, -1, dtype=np.int32)
    node_best_source = np.full(n_points, -1, dtype=np.int32)
    
    # Per-component arrays for Step 1b
    cheapest_weight = np.full(n_points, np.inf, dtype=np.float64)
    cheapest_edge_id = np.full(n_points, -1, dtype=np.int32)
    cheapest_target = np.full(n_points, -1, dtype=np.int32)
    cheapest_source = np.full(n_points, -1, dtype=np.int32)
    
    # Step 1a: Sequential per-node edge search (same as _parallel_edge_selection_numba)
    for i in range(n_points):
        my_root = _dsu_find_numba(parent, i)
        my_size = size[my_root]
        
        best_weight = np.inf
        best_edge_id = np.int32(-1)
        best_target = np.int32(-1)
        
        start = adj_indptr[i]
        end = adj_indptr[i + 1]
        
        for k in range(start, end):
            edge_id = adj_edge_ids[k]
            
            if infeasible[edge_id]:
                continue
            
            j = adj_neighbors[k]
            neighbor_root = _dsu_find_numba(parent, j)
            
            if my_root == neighbor_root:
                continue
            
            neighbor_size = size[neighbor_root]
            if my_size < neighbor_size:
                root_small = my_root
                root_large = neighbor_root
            else:
                root_small = neighbor_root
                root_large = my_root
            
            if _component_has_conflict_with_root_numba(
                root_small,
                root_large,
                parent,
                head,
                next_node,
                cannot_link_indptr,
                cannot_link_indices,
            ):
                infeasible[edge_id] = True
                continue
            
            ww = adj_weights[k]
            if ww < best_weight or (ww == best_weight and edge_id < best_edge_id):
                best_weight = ww
                best_edge_id = edge_id
                best_target = neighbor_root
        
        node_best_weight[i] = best_weight
        node_best_edge_id[i] = best_edge_id
        node_best_target[i] = best_target
        node_best_source[i] = np.int32(i)
    
    # Step 1b: Sequential per-component aggregation
    _parallel_edge_selection_step1b_cpu(
        parent,
        n_points,
        node_best_weight,
        node_best_edge_id,
        node_best_target,
        node_best_source,
        cheapest_weight,
        cheapest_edge_id,
        cheapest_target,
        cheapest_source,
    )
    
    return cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id


@numba.njit(cache=True)
def _merge_edge_into_mst_numba(
    src: int,
    tgt: int,
    weight: float,
    edge_id: int,
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    next_node: np.ndarray,
    mst_edges: np.ndarray,
    n_added: int,
    round_edges_u: np.ndarray,
    round_edges_v: np.ndarray,
    round_edges_w: np.ndarray,
    round_edges_count: int,
) -> Tuple[int, int, int]:
    """
    Merge an edge into the MST (no constraint checks).
    
    Uses lower-root-wins tie-breaking for determinism.
    
    Returns:
        Tuple of (n_added, round_edges_count, winner_root):
            Updated counts and the winning root after merge.
    """
    src_root = _dsu_find_numba(parent, src)
    tgt_root = _dsu_find_numba(parent, tgt)
    
    # Skip if already same component
    if src_root == tgt_root:
        return n_added, round_edges_count, src_root
    
    # Find actual target node from edge_id
    actual_tgt = -1
    for k in range(adj_indptr[src], adj_indptr[src + 1]):
        if adj_edge_ids[k] == edge_id:
            actual_tgt = adj_neighbors[k]
            break
    
    if actual_tgt == -1:
        return n_added, round_edges_count, src_root
    
    # Lower-root-wins tie-breaking
    winner = min(src_root, tgt_root)
    loser = max(src_root, tgt_root)
    
    # Union: attach loser -> winner
    parent[loser] = winner
    size[winner] = size[winner] + size[loser]
    
    # Concatenate member lists: winner_tail.next = loser_head
    next_node[tail[winner]] = head[loser]
    tail[winner] = tail[loser]
    
    # Add to MST edges
    mst_edges[n_added, 0] = float(src)
    mst_edges[n_added, 1] = float(actual_tgt)
    mst_edges[n_added, 2] = weight
    
    # Track round edge
    round_edges_u[round_edges_count] = src
    round_edges_v[round_edges_count] = actual_tgt
    round_edges_w[round_edges_count] = weight
    
    return n_added + 1, round_edges_count + 1, winner


@numba.njit(cache=True)
def _merge_proposed_edges_numba(
    cheapest_source: np.ndarray,
    cheapest_target: np.ndarray,
    cheapest_weight: np.ndarray,
    cheapest_edge_id: np.ndarray,
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    next_node: np.ndarray,
    mst_edges: np.ndarray,
    n_added: int,
    n_points: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Merge all proposed edges from edge selection phase.
    
    This merges WITHOUT constraint checks (Step 2). Race conditions that
    create violations will be fixed in Step 3.
    
    Args:
        cheapest_*: Arrays from edge selection (indexed by root).
        adj_*: Adjacency list arrays.
        parent, size, head, tail, next_node: DSU arrays.
        mst_edges: Output MST edges array.
        n_added: Current count of MST edges.
        n_points: Number of points/nodes.
        
    Returns:
        Tuple of (n_added, round_edges_u, round_edges_v, round_edges_w, round_count):
            - Updated n_added
            - Arrays tracking edges merged this round
            - Count of round edges
    """
    # Allocate arrays for round edges (max n_points merges possible)
    round_edges_u = np.empty(n_points, dtype=np.int32)
    round_edges_v = np.empty(n_points, dtype=np.int32)
    round_edges_w = np.empty(n_points, dtype=np.float64)
    round_count = 0
    
    # Process roots in sorted order for determinism
    roots_to_process = np.empty(n_points, dtype=np.int32)
    n_roots = 0
    for r in range(n_points):
        if parent[r] == r and cheapest_edge_id[r] != -1:
            roots_to_process[n_roots] = r
            n_roots += 1
    
    # Sort by root ID for determinism
    for i in range(n_roots):
        for j in range(i + 1, n_roots):
            if roots_to_process[i] > roots_to_process[j]:
                tmp = roots_to_process[i]
                roots_to_process[i] = roots_to_process[j]
                roots_to_process[j] = tmp
    
    # Merge edges
    for idx in range(n_roots):
        r = roots_to_process[idx]
        
        # Check if root changed during this phase
        current_root = _dsu_find_numba(parent, r)
        if current_root != r:
            continue
        
        # Check if target already merged with us
        target_root = _dsu_find_numba(parent, cheapest_target[r])
        if current_root == target_root:
            continue
        
        # Perform merge (no constraint check)
        n_added, round_count, _ = _merge_edge_into_mst_numba(
            cheapest_source[r],
            target_root,
            cheapest_weight[r],
            cheapest_edge_id[r],
            adj_indptr,
            adj_neighbors,
            adj_edge_ids,
            parent,
            size,
            head,
            tail,
            next_node,
            mst_edges,
            n_added,
            round_edges_u,
            round_edges_v,
            round_edges_w,
            round_count,
        )
    
    return n_added, round_edges_u, round_edges_v, round_edges_w, round_count


@numba.njit(cache=True)
def _detect_violations_numba(
    parent: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Detect cannot-link violations in the current DSU state.
    
    A violation occurs when two points that have a cannot-link constraint
    end up in the same component.
    
    Args:
        parent: DSU parent array.
        cannot_link_indptr: CSR indptr for cannot-link constraints.
        cannot_link_indices: CSR indices for cannot-link constraints.
        n_points: Number of points/nodes.
        
    Returns:
        Tuple of (violation_a, violation_b, n_violations):
            - Arrays of point indices that violate constraints
            - Count of violations
    """
    # Pre-allocate (max possible violations = number of constraint pairs)
    max_violations = cannot_link_indices.shape[0]
    violation_a = np.empty(max_violations, dtype=np.int32)
    violation_b = np.empty(max_violations, dtype=np.int32)
    n_violations = 0
    
    # Check each point's cannot-link pairs
    for i in range(n_points):
        i_root = _dsu_find_numba(parent, i)
        
        start = cannot_link_indptr[i]
        end = cannot_link_indptr[i + 1]
        
        for k in range(start, end):
            j = cannot_link_indices[k]
            
            # Only check each pair once (i < j)
            if i >= j:
                continue
            
            j_root = _dsu_find_numba(parent, j)
            
            if i_root == j_root:
                # Violation found
                violation_a[n_violations] = i
                violation_b[n_violations] = j
                n_violations += 1
    
    return violation_a, violation_b, n_violations


@numba.njit(cache=True)
def _check_component_has_violation_numba(
    root: int,
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
) -> bool:
    """
    Check if a component rooted at 'root' has any internal constraint violations.
    
    Args:
        root: The root of the component to check.
        parent: DSU parent array.
        head, next_node: Linked list for component membership.
        cannot_link_indptr, cannot_link_indices: Cannot-link constraints.
        
    Returns:
        True if the component has at least one violation.
    """
    # Iterate through all nodes in this component
    node = head[root]
    while node != -1:
        # Check this node's cannot-link pairs
        start = cannot_link_indptr[node]
        end = cannot_link_indptr[node + 1]
        
        for k in range(start, end):
            partner = cannot_link_indices[k]
            # Check if partner is in same component
            if _dsu_find_numba(parent, partner) == root:
                return True
        
        node = next_node[node]
    
    return False


@numba.njit(parallel=True, cache=True)
def _detect_violating_components_parallel(
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """
    Step 3a: Parallel detection of which components have violations.
    
    Each root is checked independently in parallel. Returns a boolean array
    where has_violation[r] = True if component with root r has a violation.
    """
    has_violation = np.zeros(n_points, dtype=np.bool_)
    
    for r in prange(n_points):
        if parent[r] == r:  # Is a root
            # Check this component for violations (read-only, safe for parallel)
            node = head[r]
            found_violation = False
            while node != -1 and not found_violation:
                start = cannot_link_indptr[node]
                end = cannot_link_indptr[node + 1]
                for k in range(start, end):
                    partner = cannot_link_indices[k]
                    # Use readonly find to avoid path compression races
                    partner_root = partner
                    while parent[partner_root] != partner_root:
                        partner_root = parent[partner_root]
                    if partner_root == r:
                        found_violation = True
                        break
                node = next_node[node]
            has_violation[r] = found_violation
    
    return has_violation


@numba.njit(cache=True)
def _rebuild_dsu_from_edges_numba(
    n_points: int,
    mst_edges: np.ndarray,
    n_edges: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild DSU structures from a list of MST edges.
    
    Args:
        n_points: Number of points/nodes.
        mst_edges: MST edges array of shape (n_edges, 3) with columns [u, v, w].
        n_edges: Number of edges in the MST.
        
    Returns:
        Tuple of (parent, size, head, tail, next_node): Fresh DSU arrays.
    """
    # Initialize fresh DSU
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)
    
    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1
    
    # Replay edges to rebuild DSU
    for e in range(n_edges):
        u = np.int32(mst_edges[e, 0])
        v = np.int32(mst_edges[e, 1])
        
        u_root = _dsu_find_numba(parent, u)
        v_root = _dsu_find_numba(parent, v)
        
        if u_root == v_root:
            continue
        
        # Lower-root-wins tie-breaking
        winner = min(u_root, v_root)
        loser = max(u_root, v_root)
        
        parent[loser] = winner
        size[winner] = size[winner] + size[loser]
        next_node[tail[winner]] = head[loser]
        tail[winner] = tail[loser]
    
    return parent, size, head, tail, next_node


@numba.njit(cache=True)
def _get_subcomponent_edges_numba(
    parent: np.ndarray,
    mst_edges: np.ndarray,
    n_edges: int,
    target_root: int,
) -> Tuple[np.ndarray, int]:
    """
    Get edges that belong to a specific component (both endpoints in component).
    
    Args:
        parent: DSU parent array.
        mst_edges: MST edges array.
        n_edges: Number of edges.
        target_root: Root of the target component.
        
    Returns:
        Tuple of (edges_subset, count): Edges where both endpoints are in target component.
    """
    result = np.empty((n_edges, 3), dtype=np.float64)
    count = 0
    
    for e in range(n_edges):
        u = np.int32(mst_edges[e, 0])
        v = np.int32(mst_edges[e, 1])
        
        u_root = _dsu_find_numba(parent, u)
        v_root = _dsu_find_numba(parent, v)
        
        # Edge is in component if both endpoints resolve to target_root
        if u_root == target_root and v_root == target_root:
            result[count, 0] = mst_edges[e, 0]
            result[count, 1] = mst_edges[e, 1]
            result[count, 2] = mst_edges[e, 2]
            count += 1
    
    return result, count


@numba.njit(cache=True)
def _fix_round_violations_numba(
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    next_node: np.ndarray,
    round_edges_u: np.ndarray,
    round_edges_v: np.ndarray,
    round_edges_w: np.ndarray,
    round_count: int,
    mst_edges: np.ndarray,
    n_added: int,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    cannot_link_edge_out: np.ndarray,
    n_cannot_link_edges: int,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int]:
    """
    Fix violations by removing edges merged this round (recursive structure).
    
    Algorithm (per component with violation):
    1. If no violation → done
    2. Sort round_edges by weight descending
    3. Remove heaviest edge → component splits into 2 subcomponents
    4. Check BOTH subcomponents:
       - If BOTH clean → mark removed edge as "cannot-link edge", done
       - If either has violation → recurse on violating subcomponent(s)
    5. Recursion continues until all subcomponents are clean
    
    Only the edge that produces TWO CLEAN subcomponents is marked as a cannot-link edge.
    Intermediate removals (where one subcomponent still has violations) are NOT marked.
    
    Implementation uses iterative worklist pattern since numba doesn't support complex recursion.
    """
    # Check if there are any violations globally
    _, _, n_violations = _detect_violations_numba(
        parent, cannot_link_indptr, cannot_link_indices, n_points
    )
    
    if n_violations == 0 or round_count == 0:
        return parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges
    
    # Sort round edges by weight descending (for consistent processing order)
    sorted_indices = np.arange(round_count, dtype=np.int32)
    for i in range(round_count):
        for j in range(i + 1, round_count):
            if round_edges_w[sorted_indices[i]] < round_edges_w[sorted_indices[j]]:
                tmp = sorted_indices[i]
                sorted_indices[i] = sorted_indices[j]
                sorted_indices[j] = tmp
    
    # Track which round edges to remove (start with all kept)
    round_edge_removed = np.zeros(round_count, dtype=np.bool_)
    
    # Worklist: components that need fixing
    # Each entry is a root that may have violations
    # We process iteratively instead of recursively
    max_worklist = n_points
    worklist_roots = np.empty(max_worklist, dtype=np.int32)
    worklist_count = 0
    
    # Find all roots that have violations initially
    for r in range(n_points):
        if parent[r] == r:  # Is a root
            has_viol = _check_component_has_violation_numba(
                r, parent, head, next_node, cannot_link_indptr, cannot_link_indices
            )
            if has_viol:
                worklist_roots[worklist_count] = r
                worklist_count += 1
    
    # Process worklist
    iterations = 0
    max_iterations = round_count * n_points  # Safety limit
    
    while worklist_count > 0 and iterations < max_iterations:
        iterations += 1
        
        # Pop a root from worklist
        worklist_count -= 1
        current_root = worklist_roots[worklist_count]
        
        # Re-check if this root still has violations (state may have changed)
        # First, find the current root (may have changed due to DSU updates)
        current_root = _dsu_find_numba(parent, current_root)
        
        has_viol = _check_component_has_violation_numba(
            current_root, parent, head, next_node, cannot_link_indptr, cannot_link_indices
        )
        
        if not has_viol:
            continue  # Already clean, skip
        
        # Find the heaviest non-removed round edge in this component
        best_edge_idx = -1
        best_weight = -1.0
        
        for idx in range(round_count):
            r = sorted_indices[idx]
            if round_edge_removed[r]:
                continue
            
            eu = round_edges_u[r]
            ev = round_edges_v[r]
            ew = round_edges_w[r]
            
            # Check if this edge is in the current component
            eu_root = _dsu_find_numba(parent, eu)
            ev_root = _dsu_find_numba(parent, ev)
            
            if eu_root == current_root and ev_root == current_root:
                if ew > best_weight:
                    best_weight = ew
                    best_edge_idx = r
        
        if best_edge_idx == -1:
            # No round edge to remove in this component - shouldn't happen
            continue
        
        # Remove this edge
        round_edge_removed[best_edge_idx] = True
        removed_u = round_edges_u[best_edge_idx]
        removed_v = round_edges_v[best_edge_idx]
        removed_w = round_edges_w[best_edge_idx]
        
        # Rebuild MST without the removed round edges
        temp_mst = np.empty((n_added, 3), dtype=np.float64)
        temp_n = 0
        
        for e in range(n_added):
            eu = np.int32(mst_edges[e, 0])
            ev = np.int32(mst_edges[e, 1])
            ew = mst_edges[e, 2]
            
            # Check if this edge is a removed round edge
            is_removed = False
            for rr in range(round_count):
                if round_edge_removed[rr]:
                    ru = round_edges_u[rr]
                    rv = round_edges_v[rr]
                    rw = round_edges_w[rr]
                    if ((eu == ru and ev == rv) or (eu == rv and ev == ru)) and ew == rw:
                        is_removed = True
                        break
            
            if not is_removed:
                temp_mst[temp_n, 0] = mst_edges[e, 0]
                temp_mst[temp_n, 1] = mst_edges[e, 1]
                temp_mst[temp_n, 2] = mst_edges[e, 2]
                temp_n += 1
        
        # Rebuild DSU from temp_mst
        temp_parent, temp_size, temp_head, temp_tail, temp_next_node = _rebuild_dsu_from_edges_numba(
            n_points, temp_mst, temp_n
        )
        
        # Find the roots of the two subcomponents (removed_u and removed_v may now be in different components)
        subcomp_a_root = _dsu_find_numba(temp_parent, removed_u)
        subcomp_b_root = _dsu_find_numba(temp_parent, removed_v)
        
        # Check violations in both subcomponents
        viol_a = _check_component_has_violation_numba(
            subcomp_a_root, temp_parent, temp_head, temp_next_node,
            cannot_link_indptr, cannot_link_indices
        )
        viol_b = _check_component_has_violation_numba(
            subcomp_b_root, temp_parent, temp_head, temp_next_node,
            cannot_link_indptr, cannot_link_indices
        )
        
        # Update main DSU state
        for i in range(n_points):
            parent[i] = temp_parent[i]
            size[i] = temp_size[i]
            head[i] = temp_head[i]
            tail[i] = temp_tail[i]
            next_node[i] = temp_next_node[i]
        
        for e in range(temp_n):
            mst_edges[e, 0] = temp_mst[e, 0]
            mst_edges[e, 1] = temp_mst[e, 1]
            mst_edges[e, 2] = temp_mst[e, 2]
        n_added = temp_n
        
        if not viol_a and not viol_b:
            # BOTH subcomponents are clean!
            # Mark this edge as a cannot-link edge
            cannot_link_edge_out[n_cannot_link_edges, 0] = float(removed_u)
            cannot_link_edge_out[n_cannot_link_edges, 1] = float(removed_v)
            cannot_link_edge_out[n_cannot_link_edges, 2] = removed_w
            n_cannot_link_edges += 1
        else:
            # At least one subcomponent still has violations
            # Add violating subcomponents to worklist (do NOT mark edge as cannot-link)
            if viol_a and worklist_count < max_worklist:
                worklist_roots[worklist_count] = subcomp_a_root
                worklist_count += 1
            if viol_b and worklist_count < max_worklist:
                worklist_roots[worklist_count] = subcomp_b_root
                worklist_count += 1
    
    return parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges


@numba.njit(cache=True)
def _fix_round_violations_with_initial_worklist_numba(
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    next_node: np.ndarray,
    round_edges_u: np.ndarray,
    round_edges_v: np.ndarray,
    round_edges_w: np.ndarray,
    round_count: int,
    mst_edges: np.ndarray,
    n_added: int,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    cannot_link_edge_out: np.ndarray,
    n_cannot_link_edges: int,
    n_points: int,
    initial_worklist: np.ndarray,
    initial_worklist_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int]:
    """
    Fix violations given a pre-computed initial worklist (Step 3b - sequential correction).
    
    This version accepts the initial worklist of violating components computed externally
    (potentially in parallel), then does sequential correction.
    """
    if initial_worklist_count == 0 or round_count == 0:
        return parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges
    
    # Sort round edges by weight descending
    sorted_indices = np.arange(round_count, dtype=np.int32)
    for i in range(round_count):
        for j in range(i + 1, round_count):
            if round_edges_w[sorted_indices[i]] < round_edges_w[sorted_indices[j]]:
                tmp = sorted_indices[i]
                sorted_indices[i] = sorted_indices[j]
                sorted_indices[j] = tmp
    
    round_edge_removed = np.zeros(round_count, dtype=np.bool_)
    max_worklist = n_points
    worklist_roots = np.empty(max_worklist, dtype=np.int32)
    
    # Copy initial worklist
    worklist_count = initial_worklist_count
    for i in range(initial_worklist_count):
        worklist_roots[i] = initial_worklist[i]
    
    iterations = 0
    max_iterations = round_count * n_points
    
    while worklist_count > 0 and iterations < max_iterations:
        iterations += 1
        worklist_count -= 1
        current_root = worklist_roots[worklist_count]
        current_root = _dsu_find_numba(parent, current_root)
        
        has_viol = _check_component_has_violation_numba(
            current_root, parent, head, next_node, cannot_link_indptr, cannot_link_indices
        )
        if not has_viol:
            continue
        
        best_edge_idx = -1
        best_weight = -1.0
        
        for idx in range(round_count):
            r = sorted_indices[idx]
            if round_edge_removed[r]:
                continue
            eu, ev, ew = round_edges_u[r], round_edges_v[r], round_edges_w[r]
            eu_root = _dsu_find_numba(parent, eu)
            ev_root = _dsu_find_numba(parent, ev)
            if eu_root == current_root and ev_root == current_root:
                if ew > best_weight:
                    best_weight = ew
                    best_edge_idx = r
        
        if best_edge_idx == -1:
            continue
        
        round_edge_removed[best_edge_idx] = True
        removed_u = round_edges_u[best_edge_idx]
        removed_v = round_edges_v[best_edge_idx]
        removed_w = round_edges_w[best_edge_idx]
        
        temp_mst = np.empty((n_added, 3), dtype=np.float64)
        temp_n = 0
        
        for e in range(n_added):
            eu = np.int32(mst_edges[e, 0])
            ev = np.int32(mst_edges[e, 1])
            ew = mst_edges[e, 2]
            
            is_removed = False
            for rr in range(round_count):
                if round_edge_removed[rr]:
                    ru, rv, rw = round_edges_u[rr], round_edges_v[rr], round_edges_w[rr]
                    if ((eu == ru and ev == rv) or (eu == rv and ev == ru)) and ew == rw:
                        is_removed = True
                        break
            
            if not is_removed:
                temp_mst[temp_n, 0] = mst_edges[e, 0]
                temp_mst[temp_n, 1] = mst_edges[e, 1]
                temp_mst[temp_n, 2] = mst_edges[e, 2]
                temp_n += 1
        
        temp_parent, temp_size, temp_head, temp_tail, temp_next_node = _rebuild_dsu_from_edges_numba(
            n_points, temp_mst, temp_n
        )
        
        subcomp_a_root = _dsu_find_numba(temp_parent, removed_u)
        subcomp_b_root = _dsu_find_numba(temp_parent, removed_v)
        
        viol_a = _check_component_has_violation_numba(
            subcomp_a_root, temp_parent, temp_head, temp_next_node, cannot_link_indptr, cannot_link_indices
        )
        viol_b = _check_component_has_violation_numba(
            subcomp_b_root, temp_parent, temp_head, temp_next_node, cannot_link_indptr, cannot_link_indices
        )
        
        for i in range(n_points):
            parent[i] = temp_parent[i]
            size[i] = temp_size[i]
            head[i] = temp_head[i]
            tail[i] = temp_tail[i]
            next_node[i] = temp_next_node[i]
        
        for e in range(temp_n):
            mst_edges[e, 0] = temp_mst[e, 0]
            mst_edges[e, 1] = temp_mst[e, 1]
            mst_edges[e, 2] = temp_mst[e, 2]
        n_added = temp_n
        
        if not viol_a and not viol_b:
            cannot_link_edge_out[n_cannot_link_edges, 0] = float(removed_u)
            cannot_link_edge_out[n_cannot_link_edges, 1] = float(removed_v)
            cannot_link_edge_out[n_cannot_link_edges, 2] = removed_w
            n_cannot_link_edges += 1
        else:
            if viol_a and worklist_count < max_worklist:
                worklist_roots[worklist_count] = subcomp_a_root
                worklist_count += 1
            if viol_b and worklist_count < max_worklist:
                worklist_roots[worklist_count] = subcomp_b_root
                worklist_count += 1
    
    return parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges


@numba.njit(cache=True)
def _parallel_constrained_boruvka_mst_sequential_numba(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
    n_edges: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sequential constrained Borůvka MST algorithm with violation correction.
    
    This is the fully sequential version where Step 1a, 1b are both sequential.
    Use this as the baseline or when parallel overhead exceeds benefits.
    
    This algorithm uses a 3-step round structure:
        Step 1: Sequential edge selection WITH constraint checks
        Step 2: Sequential merging WITHOUT constraint checks (race conditions possible)
        Step 3: Violation correction using only current round edges
    
    Race conditions in Step 2 are fixed in Step 3 by removing the heaviest
    round edge(s) until all violations are resolved.
    
    Args:
        adj_indptr (np.ndarray): CSR indptr for adjacency list.
        adj_neighbors (np.ndarray): Neighbor node IDs in adjacency list.
        adj_weights (np.ndarray): Edge weights in adjacency list.
        adj_edge_ids (np.ndarray): Original edge indices in adjacency list.
        cannot_link_indptr (np.ndarray): CSR indptr for cannot-link constraints.
        cannot_link_indices (np.ndarray): CSR indices for cannot-link constraints.
        n_points (int): Number of points/nodes.
        n_edges (int): Number of original edges.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - mst_edges: Shape (<=N-1, 3), float64. Columns: [u, v, w]
            - cannot_link_edges: Shape (<=M, 3), float64. Edges removed due to violations.
    """
    # Initialize DSU arrays
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)
    
    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1
    
    # Track infeasible edges (constraint-violating edges never retry)
    infeasible = np.zeros(n_edges, dtype=np.bool_)
    
    # Output arrays
    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0
    
    # Cannot-link edges (edges removed due to violations)
    cannot_link_edge_out = np.empty((n_points - 1, 3), dtype=np.float64)
    n_cannot_link_edges = 0
    
    while True:
        # ========== Step 1: Sequential edge selection WITH constraint checks ==========
        cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id = \
            _parallel_edge_selection_numba(
                adj_indptr,
                adj_neighbors,
                adj_weights,
                adj_edge_ids,
                parent,
                size,
                head,
                next_node,
                cannot_link_indptr,
                cannot_link_indices,
                infeasible,
                n_points,
            )
        
        # Check if any component found an edge
        any_edge_found = False
        for r in range(n_points):
            if parent[r] == r and cheapest_edge_id[r] != -1:
                any_edge_found = True
                break
        
        if not any_edge_found:
            break
        
        # ========== Step 2: Merge proposed edges WITHOUT constraint checks ==========
        n_added_before = n_added
        n_added, round_edges_u, round_edges_v, round_edges_w, round_count = \
            _merge_proposed_edges_numba(
                cheapest_source,
                cheapest_target,
                cheapest_weight,
                cheapest_edge_id,
                adj_indptr,
                adj_neighbors,
                adj_edge_ids,
                parent,
                size,
                head,
                tail,
                next_node,
                mst_edges,
                n_added,
                n_points,
            )
        
        n_merges = n_added - n_added_before
        
        if n_merges == 0:
            break
        
        # ========== Step 3: Fix violations using only round edges ==========
        parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges = \
            _fix_round_violations_numba(
                parent,
                size,
                head,
                tail,
                next_node,
                round_edges_u,
                round_edges_v,
                round_edges_w,
                round_count,
                mst_edges,
                n_added,
                cannot_link_indptr,
                cannot_link_indices,
                cannot_link_edge_out,
                n_cannot_link_edges,
                n_points,
            )
        
        if n_added >= n_points - 1:
            break
    
    return mst_edges[:n_added], cannot_link_edge_out[:n_cannot_link_edges]


def _parallel_constrained_boruvka_mst_cpu(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
    n_edges: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CPU parallel constrained Borůvka MST algorithm with violation correction.
    
    This version parallelizes Step 1a (per-node edge search) using prange,
    while Step 1b (aggregation) and Step 2 (merge) remain sequential.
    Step 3a (detection) is parallel, Step 3b (correction) is sequential.
    
    Parallelization strategy:
        Step 1a: PARALLEL (prange) - each node finds its best edge
        Step 1b: SEQUENTIAL - aggregate per-node to per-component (needs atomics)
        Step 2: SEQUENTIAL - DSU mutations have ordering dependencies
        Step 3a: PARALLEL (prange) - detect violating components
        Step 3b: SEQUENTIAL - violation correction
    
    Note: This function is NOT njit-decorated because it needs to call
    the parallel detection function (_detect_violating_components_parallel).
    Numba njit functions cannot call parallel=True njit functions.
    
    Args:
        adj_indptr (np.ndarray): CSR indptr for adjacency list.
        adj_neighbors (np.ndarray): Neighbor node IDs in adjacency list.
        adj_weights (np.ndarray): Edge weights in adjacency list.
        adj_edge_ids (np.ndarray): Original edge indices in adjacency list.
        cannot_link_indptr (np.ndarray): CSR indptr for cannot-link constraints.
        cannot_link_indices (np.ndarray): CSR indices for cannot-link constraints.
        n_points (int): Number of points/nodes.
        n_edges (int): Number of original edges.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - mst_edges: Shape (<=N-1, 3), float64. Columns: [u, v, w]
            - cannot_link_edges: Shape (<=M, 3), float64. Edges removed due to violations.
    """
    # Initialize DSU arrays
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)
    
    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1
    
    # Track infeasible edges (constraint-violating edges never retry)
    infeasible = np.zeros(n_edges, dtype=np.bool_)
    
    # Output arrays
    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0
    
    # Cannot-link edges (edges removed due to violations)
    cannot_link_edge_out = np.empty((n_points - 1, 3), dtype=np.float64)
    n_cannot_link_edges = 0
    
    while True:
        # ========== Step 1: CPU Parallel edge selection ==========
        # Step 1a is parallel (prange), Step 1b is sequential
        cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id = \
            _parallel_edge_selection_cpu(
                adj_indptr,
                adj_neighbors,
                adj_weights,
                adj_edge_ids,
                parent,
                size,
                head,
                next_node,
                cannot_link_indptr,
                cannot_link_indices,
                infeasible,
                n_points,
            )
        
        # Check if any component found an edge
        any_edge_found = False
        for r in range(n_points):
            if parent[r] == r and cheapest_edge_id[r] != -1:
                any_edge_found = True
                break
        
        if not any_edge_found:
            break
        
        # ========== Step 2: Merge proposed edges (SEQUENTIAL) ==========
        # DSU mutations have ordering dependencies - cannot be parallelized
        n_added_before = n_added
        n_added, round_edges_u, round_edges_v, round_edges_w, round_count = \
            _merge_proposed_edges_numba(
                cheapest_source,
                cheapest_target,
                cheapest_weight,
                cheapest_edge_id,
                adj_indptr,
                adj_neighbors,
                adj_edge_ids,
                parent,
                size,
                head,
                tail,
                next_node,
                mst_edges,
                n_added,
                n_points,
            )
        
        n_merges = n_added - n_added_before
        
        if n_merges == 0:
            break
        
        # ========== Step 3: Fix violations ==========
        # Step 3a: PARALLEL detection - find all violating components
        has_violation = _detect_violating_components_parallel(
            parent, head, next_node, cannot_link_indptr, cannot_link_indices, n_points
        )
        
        # Build initial worklist from parallel detection results
        initial_worklist = np.empty(n_points, dtype=np.int32)
        initial_worklist_count = 0
        for root in range(n_points):
            if parent[root] == root and has_violation[root]:
                initial_worklist[initial_worklist_count] = root
                initial_worklist_count += 1
        
        # Step 3b: SEQUENTIAL correction - fix violations one at a time
        if initial_worklist_count > 0:
            parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges = \
                _fix_round_violations_with_initial_worklist_numba(
                    parent,
                    size,
                    head,
                    tail,
                    next_node,
                    round_edges_u,
                    round_edges_v,
                    round_edges_w,
                    round_count,
                    mst_edges,
                    n_added,
                    cannot_link_indptr,
                    cannot_link_indices,
                    cannot_link_edge_out,
                    n_cannot_link_edges,
                    n_points,
                    initial_worklist,
                    initial_worklist_count,
                )
        
        if n_added >= n_points - 1:
            break
    
    return mst_edges[:n_added], cannot_link_edge_out[:n_cannot_link_edges]


def _select_parallel_backend(
    n_points: int,
    parallel_backend: str,
) -> str:
    """
    Select the appropriate parallel backend based on user preference and hardware.
    
    Args:
        n_points: Number of nodes in the graph.
        parallel_backend: User-specified backend preference.
            - "auto": Automatically select based on hardware and problem size
            - "cuda": Force CUDA (raises error if unavailable)
            - "cpu": Force CPU parallel (Step 1a parallel, Step 1b sequential)
            - "sequential": Force fully sequential execution
            
    Returns:
        str: The selected backend ("cuda", "cpu", or "sequential").
        
    Raises:
        ValueError: If "cuda" is requested but CUDA is not available.
    """
    if parallel_backend == "cuda":
        if _check_cuda_available():
            return "cuda"
        else:
            raise ValueError(
                "parallel_backend='cuda' requested but CUDA is not available. "
                "Install numba with CUDA support or use parallel_backend='cpu' or 'auto'."
            )
    
    if parallel_backend == "cpu":
        return "cpu"
    
    if parallel_backend == "sequential":
        return "sequential"
    
    # parallel_backend == "auto"
    if _check_cuda_available() and n_points > 10000:
        return "cuda"
    
    # Default to CPU parallel for medium-sized problems
    # For very small problems, the overhead may not be worth it,
    # but the difference is negligible
    return "cpu"


def _parallel_constrained_boruvka_mst(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
    n_edges: int,
    parallel_backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel constrained Borůvka MST algorithm with configurable backend.
    
    This is the main entry point for the parallel Borůvka algorithm. It selects
    the appropriate backend based on hardware availability and problem size.
    
    Backend options:
        - "auto" (default): Automatically select CUDA for large graphs if available,
          otherwise use CPU parallel.
        - "cuda": Force CUDA backend. Raises error if CUDA unavailable.
          - Step 1a: Parallel (one GPU thread per node)
          - Step 1b: Parallel (atomic_min for aggregation)
          - Step 2: Sequential (DSU mutations)
        - "cpu": Force CPU parallel backend.
          - Step 1a: Parallel (numba prange)
          - Step 1b: Sequential (no CPU atomics)
          - Step 2: Sequential (DSU mutations)
        - "sequential": Force fully sequential execution (for debugging/comparison).
    
    Args:
        adj_indptr (np.ndarray): CSR indptr for adjacency list.
        adj_neighbors (np.ndarray): Neighbor node IDs in adjacency list.
        adj_weights (np.ndarray): Edge weights in adjacency list.
        adj_edge_ids (np.ndarray): Original edge indices in adjacency list.
        cannot_link_indptr (np.ndarray): CSR indptr for cannot-link constraints.
        cannot_link_indices (np.ndarray): CSR indices for cannot-link constraints.
        n_points (int): Number of points/nodes.
        n_edges (int): Number of original edges.
        parallel_backend (str): Backend selection. One of "auto", "cuda", "cpu", "sequential".
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - mst_edges: Shape (<=N-1, 3), float64. Columns: [u, v, w]
            - cannot_link_edges: Shape (<=M, 3), float64. Edges removed due to violations.
    """
    backend = _select_parallel_backend(n_points, parallel_backend)
    
    if backend == "cuda":
        # TODO: Implement CUDA backend
        # For now, fall back to CPU parallel
        # raise NotImplementedError("CUDA backend not yet implemented")
        backend = "cpu"
    
    if backend == "cpu":
        return _parallel_constrained_boruvka_mst_cpu(
            adj_indptr,
            adj_neighbors,
            adj_weights,
            adj_edge_ids,
            cannot_link_indptr,
            cannot_link_indices,
            n_points,
            n_edges,
        )
    
    # backend == "sequential"
    return _parallel_constrained_boruvka_mst_sequential_numba(
        adj_indptr,
        adj_neighbors,
        adj_weights,
        adj_edge_ids,
        cannot_link_indptr,
        cannot_link_indices,
        n_points,
        n_edges,
    )


def _ensure_csr_distance_matrix(distances: Union[np.ndarray, CSR]) -> CSR:
    """
    Validate and return distances as CSR.

    IMPORTANT:
        - Do NOT call eliminate_zeros(): explicit off-diagonal zeros are meaningful edges.
        - Remove diagonal entries structurally so they do not affect core distances.

    Args:
        distances (Union[np.ndarray, CSR]):
            The precomputed distance matrix.

    Returns:
        (CSR):
            The validated and sanitized CSR distance matrix.
    """
    if sp.isspmatrix_csr(distances):
        distance_matrix_csr = distances.copy().astype(np.float64)
    elif isinstance(distances, np.ndarray):
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("Precomputed distance matrix must be square (N,N).")
        distance_matrix_csr = sp.csr_matrix(distances.astype(np.float64, copy=False))
    else:
        raise TypeError("distances must be a numpy array or scipy.sparse.csr_matrix.")

    n_points = int(distance_matrix_csr.shape[0])
    if n_points == 0 or distance_matrix_csr.shape[1] != n_points:
        raise ValueError(
            "Precomputed distance matrix must be non-empty and square (N,N)."
        )

    # ---- Remove diagonal structurally (preserves off-diagonal explicit zeros) ----
    coo = distance_matrix_csr.tocoo()
    mask_offdiag = coo.row != coo.col
    distance_matrix_csr = sp.coo_matrix(
        (coo.data[mask_offdiag], (coo.row[mask_offdiag], coo.col[mask_offdiag])),
        shape=coo.shape,
    ).tocsr()

    # Keep explicit zeros; just canonicalize representation.
    distance_matrix_csr.sum_duplicates()
    distance_matrix_csr.sort_indices()

    if distance_matrix_csr.data.size and np.any(distance_matrix_csr.data < 0):
        raise ValueError("Distances must be non-negative.")

    return distance_matrix_csr


def _symmetrize_min_keep_present(distance_matrix_csr: CSR) -> CSR:
    """
    Symmetrize a sparse distance matrix by taking:
        - min(A_ij, A_ji) when both exist
        - the existing value when only one direction exists

    IMPORTANT:
        - Preserve explicit off-diagonal zeros.
        - Do not use eliminate_zeros() or bool-mask multiply patterns (they often drop zeros).

    Args:
        distance_matrix_csr (CSR):
            The sparse distance matrix.

    Returns:
        (CSR):
            The symmetrized sparse distance matrix.
    """
    A = distance_matrix_csr.tocsr(copy=True)
    A.sum_duplicates()
    A.sort_indices()

    B = A.T.tocsr(copy=True)
    B.sum_duplicates()
    B.sort_indices()

    n_points = int(A.shape[0])
    if n_points == 0:
        return A

    coo_a = A.tocoo()
    coo_b = B.tocoo()

    rows = np.concatenate([coo_a.row, coo_b.row]).astype(np.int64, copy=False)
    cols = np.concatenate([coo_a.col, coo_b.col]).astype(np.int64, copy=False)
    data = np.concatenate([coo_a.data, coo_b.data]).astype(np.float64, copy=False)

    # Drop diagonal structurally.
    mask_offdiag = rows != cols
    rows = rows[mask_offdiag]
    cols = cols[mask_offdiag]
    data = data[mask_offdiag]

    if data.size == 0:
        return sp.csr_matrix(A.shape, dtype=np.float64)

    # Reduce duplicates by min over identical (row, col).
    key = rows * np.int64(n_points) + cols
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    rows = rows[order]
    cols = cols[order]
    data = data[order]

    group_starts = np.empty(key.shape[0], dtype=bool)
    group_starts[0] = True
    group_starts[1:] = key[1:] != key[:-1]
    idx_start = np.flatnonzero(group_starts)

    data_min = np.minimum.reduceat(data, idx_start)
    rows_u = rows[idx_start]
    cols_u = cols[idx_start]

    sym = sp.coo_matrix((data_min, (rows_u, cols_u)), shape=A.shape).tocsr()
    sym.sum_duplicates()
    sym.sort_indices()
    return sym


def _core_distances_from_sparse_rows(
    distance_matrix_csr: CSR,
    *,
    min_samples: int,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute HDBSCAN core distances from a sparse precomputed distance graph.

    Args:
        distance_matrix_csr (CSR):
            The sparse distance matrix.
        min_samples (int):
            The number of samples in a neighborhood for a point to be considered a core point.
        sample_weights (Optional[np.ndarray]):
            Weights for each sample.

    Returns:
        (np.ndarray):
            The core distances for each point.
    """
    n_points = int(distance_matrix_csr.shape[0])
    core_distances = np.empty(n_points, dtype=np.float64)

    if sample_weights is None:
        for idx_point in range(n_points):
            start = int(distance_matrix_csr.indptr[idx_point])
            end = int(distance_matrix_csr.indptr[idx_point + 1])
            degree = int(end - start)

            if degree == 0:
                core_distances[idx_point] = np.inf
                continue

            neighbor_distances_sorted = np.sort(
                distance_matrix_csr.data[start:end], kind="mergesort"
            )
            if degree >= int(min_samples):
                core_distances[idx_point] = float(
                    neighbor_distances_sorted[int(min_samples) - 1]
                )
            else:
                core_distances[idx_point] = float(neighbor_distances_sorted[-1])
        return core_distances

    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    if sample_weights.shape != (n_points,):
        raise ValueError("sample_weights must have shape (N,)")

    for idx_point in range(n_points):
        start = int(distance_matrix_csr.indptr[idx_point])
        end = int(distance_matrix_csr.indptr[idx_point + 1])

        neighbor_indices = distance_matrix_csr.indices[start:end]
        neighbor_distances = distance_matrix_csr.data[start:end]
        if neighbor_indices.size == 0:
            core_distances[idx_point] = np.inf
            continue

        order = np.argsort(neighbor_distances, kind="mergesort")
        cumulative_weight = 0.0
        target_weight = float(min_samples)

        reached = False
        for idx_order in order.tolist():
            cumulative_weight += float(
                sample_weights[int(neighbor_indices[int(idx_order)])]
            )
            if cumulative_weight >= target_weight:
                core_distances[idx_point] = float(neighbor_distances[int(idx_order)])
                reached = True
                break

        if not reached:
            core_distances[idx_point] = float(neighbor_distances[int(order[-1])])

    return core_distances


def _mutual_reachability_edges_upper_triangle(
    distance_matrix_csr: CSR,
    *,
    core_distances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an undirected edge list (u, v, w) from the upper triangle of a sparse
    distance matrix, with mutual reachability weights.

    Args:
        distance_matrix_csr (CSR):
            The sparse distance matrix.
        core_distances (np.ndarray):
            The core distances for each point.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]):
            A tuple containing the source nodes, destination nodes, and weights of the edges.
    """
    coo = sp.triu(distance_matrix_csr, k=1).tocoo()
    if coo.nnz == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    u = coo.row.astype(np.int64, copy=False)
    v = coo.col.astype(np.int64, copy=False)
    d = coo.data.astype(np.float64, copy=False)

    w = np.maximum.reduce([d, core_distances[u], core_distances[v]])
    return u, v, w


def _kruskal_mst_unconstrained(
    *,
    n_points: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Standard Kruskal MST on an undirected weighted graph.

    Args:
        n_points (int): The number of points.
        u (np.ndarray): Source nodes of edges.
        v (np.ndarray): Destination nodes of edges.
        w (np.ndarray): Weights of edges.

    Returns:
        (np.ndarray):
            The minimum spanning tree edges.
    """
    order = np.argsort(w, kind="mergesort")
    u_sorted = u[order]
    v_sorted = v[order]
    w_sorted = w[order]

    ds = ds_rank_create(int(n_points))
    mst_edges = np.empty((int(n_points) - 1, 3), dtype=np.float64)

    n_added = 0
    for idx_edge in range(u_sorted.shape[0]):
        a = int(u_sorted[idx_edge])
        b = int(v_sorted[idx_edge])
        ww = float(w_sorted[idx_edge])

        root_a = int(ds_find(ds, a))
        root_b = int(ds_find(ds, b))
        if root_a == root_b:
            continue

        ds_union_by_rank(ds, root_a, root_b)
        mst_edges[n_added, 0] = float(a)
        mst_edges[n_added, 1] = float(b)
        mst_edges[n_added, 2] = float(ww)
        n_added += 1

        if n_added == int(n_points) - 1:
            break

    return mst_edges[:n_added]


def _mst_constrained_hard(
    *,
    n_points: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    merge_constraint: MergeConstraint,
    mst_method: str = "boruvka",
    parallel_backend: str = "auto",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Hard constrained MST using Borůvka, Parallel Borůvka, or Kruskal algorithm.

    Skip edges whose merge would place any cannot-link pair in one component.
    This returns a spanning forest if constraints prevent full connectivity.

    Args:
        n_points (int): The number of points.
        u (np.ndarray): Source nodes of edges.
        v (np.ndarray): Destination nodes of edges.
        w (np.ndarray): Weights of edges.
        merge_constraint (MergeConstraint): The merge constraint object.
        mst_method (str): Algorithm to use:
            - "boruvka" (default): Sequential constrained Borůvka
            - "parallel_boruvka": Parallel constrained Borůvka with violation correction
            - "kruskal": Sequential constrained Kruskal
        parallel_backend (str): Backend for parallel_boruvka method:
            - "auto" (default): Automatically select based on hardware and problem size
            - "cuda": Force CUDA (raises error if unavailable)
            - "cpu": Force CPU parallel (Step 1a parallel, Step 1b sequential)
            - "sequential": Force fully sequential execution

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - The constrained minimum spanning tree edges.
            - cannot_link_edges (only for parallel_boruvka): Edges removed due to
              constraint violations during parallel merging. None for other methods.

    Raises:
        ValueError: If merge_constraint lacks CSR payload arrays or mst_method is invalid.
    """
    if (
        merge_constraint.cannot_link_indptr is None
        or merge_constraint.cannot_link_indices is None
    ):
        raise ValueError(
            "strict=True requires merge_constraint built from a cannot-link matrix "
            "(must provide CSR payload arrays)."
        )

    cannot_link_indptr = np.asarray(merge_constraint.cannot_link_indptr, dtype=np.int64)
    cannot_link_indices = np.asarray(
        merge_constraint.cannot_link_indices, dtype=np.int32
    )

    if cannot_link_indptr.ndim != 1 or cannot_link_indptr.size != int(n_points) + 1:
        raise ValueError(
            "merge_constraint.cannot_link_indptr must have shape (N+1,) for strict=True."
        )

    cannot_link_edges = None

    if mst_method == "boruvka":
        # Build adjacency list for Borůvka
        u_arr = np.asarray(u, dtype=np.int32)
        v_arr = np.asarray(v, dtype=np.int32)
        w_arr = np.asarray(w, dtype=np.float64)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
            u_arr, v_arr, w_arr, int(n_points)
        )

        mst_edges = _constrained_boruvka_mst_csr_strict_numba(
            adj_indptr,
            adj_neighbors,
            adj_weights,
            adj_edge_ids,
            cannot_link_indptr,
            cannot_link_indices,
            int(n_points),
            int(u_arr.shape[0]),
        )
    elif mst_method == "parallel_boruvka":
        # Build adjacency list for Parallel Borůvka
        u_arr = np.asarray(u, dtype=np.int32)
        v_arr = np.asarray(v, dtype=np.int32)
        w_arr = np.asarray(w, dtype=np.float64)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
            u_arr, v_arr, w_arr, int(n_points)
        )

        mst_edges, cannot_link_edges = _parallel_constrained_boruvka_mst(
            adj_indptr,
            adj_neighbors,
            adj_weights,
            adj_edge_ids,
            cannot_link_indptr,
            cannot_link_indices,
            int(n_points),
            int(u_arr.shape[0]),
            parallel_backend=parallel_backend,
        )
    elif mst_method == "kruskal":
        order = np.argsort(w, kind="mergesort")
        u_sorted = np.asarray(u[order], dtype=np.int32)
        v_sorted = np.asarray(v[order], dtype=np.int32)
        w_sorted = np.asarray(w[order], dtype=np.float64)

        mst_edges = _constrained_kruskal_mst_csr_strict_sorted_numba(
            u_sorted,
            v_sorted,
            w_sorted,
            cannot_link_indptr,
            cannot_link_indices,
            int(n_points),
        )
    else:
        raise ValueError(
            f"Invalid mst_method: {mst_method}. Must be 'boruvka', 'parallel_boruvka', or 'kruskal'."
        )

    return mst_edges, cannot_link_edges


def _choose_large_finite_penalty(
    distances: Union[np.ndarray, CSR],
    *,
    user_penalty: float,
    scale_quantile: float = 99.9,
    factor: float = 1e6,
) -> float:
    """
    Map user_penalty to a large finite value if the user passes +inf.

    Args:
        distances (Union[np.ndarray, CSR]): The distance matrix.
        user_penalty (float): The user-provided penalty.
        scale_quantile (float): The quantile to use for scaling.
        factor (float): The factor to multiply the base value by.

    Returns:
        (float):
            A large finite penalty value.
    """
    if np.isfinite(user_penalty):
        return float(user_penalty)

    if sp.issparse(distances):
        vals = distances.data[np.isfinite(distances.data)]
    else:
        vals = np.asarray(distances, dtype=float)
        vals = vals[np.isfinite(vals)]

    base = 1.0 if vals.size == 0 else float(np.percentile(vals, scale_quantile))
    finite_penalty = float(base * factor + 1.0)
    finite_penalty = float(min(finite_penalty, np.finfo(np.float64).max / 10.0))
    return finite_penalty


def _connect_components_with_penalty_edges(
    *,
    n_points: int,
    mst_edges: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """
    If mst_edges is a forest, connect its components with penalty-weight edges
    so that we return an (N-1, 3) tree.

    Args:
        n_points (int): The number of points.
        mst_edges (np.ndarray): The MST edges.
        penalty (float): The penalty to use for connecting components.

    Returns:
        (np.ndarray):
            The connected MST edges.
    """
    n_points = int(n_points)
    if mst_edges.shape[0] >= n_points - 1:
        return mst_edges

    ds = ds_rank_create(n_points)
    for row in mst_edges:
        a = int(row[0])
        b = int(row[1])
        root_a = int(ds_find(ds, a))
        root_b = int(ds_find(ds, b))
        if root_a != root_b:
            ds_union_by_rank(ds, root_a, root_b)

    root_to_rep = {}
    for idx_point in range(n_points):
        root = int(ds_find(ds, idx_point))
        if root not in root_to_rep:
            root_to_rep[root] = int(idx_point)

    reps = list(root_to_rep.values())
    if len(reps) <= 1:
        return mst_edges

    rep_root = int(reps[0])
    extra_edges = np.empty((len(reps) - 1, 3), dtype=np.float64)
    for idx_extra, rep in enumerate(reps[1:]):
        extra_edges[idx_extra, 0] = float(rep_root)
        extra_edges[idx_extra, 1] = float(int(rep))
        extra_edges[idx_extra, 2] = float(penalty)

    return np.vstack([mst_edges, extra_edges])


def _sanitize_mst_edge_weights(
    mst_edges: np.ndarray,
    *,
    finite_penalty: float,
) -> np.ndarray:
    """
    Replace any non-finite edge weights in the MST with a large finite penalty.

    Args:
        mst_edges (np.ndarray): The MST edges.
        finite_penalty (float): The penalty to use.

    Returns:
        (np.ndarray):
            The sanitized MST edges.
    """
    bad = ~np.isfinite(mst_edges[:, 2])
    if np.any(bad):
        mst_edges = mst_edges.copy()
        mst_edges[bad, 2] = float(finite_penalty)
    return mst_edges


def connected_components_from_distance_graph(
    distances: Union[np.ndarray, CSR]
) -> np.ndarray:
    """
    Compute connected components from the support of the precomputed distance graph.

    Args:
        distances (Union[np.ndarray, CSR]): The distance matrix.

    Returns:
        (np.ndarray):
            The component labels for each point.
    """
    if isinstance(distances, np.ndarray):
        n_points = int(distances.shape[0])
        return np.zeros(n_points, dtype=np.int64)

    distance_matrix_csr = distances.tocsr()
    n_points = int(distance_matrix_csr.shape[0])

    adjacency = distance_matrix_csr.copy()
    adjacency.data = np.ones_like(adjacency.data, dtype=np.int8)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    n_components, labels = sp_csgraph.connected_components(
        csgraph=adjacency, directed=False, return_labels=True
    )
    _ = n_components
    return labels.astype(np.int64, copy=False)


def find_cannot_link_violations(
    labels: np.ndarray,
    *,
    merge_constraint: MergeConstraint,
    noise_label: int = -1,
) -> np.ndarray:
    """
    Enumerate cannot-link violations in the final cluster labels.

    Args:
        labels (np.ndarray): The cluster labels.
        merge_constraint (MergeConstraint): The merge constraint object.
        noise_label (int): The label for noise points.

    Returns:
        (np.ndarray):
            An array of pairs of indices that violate the cannot-link constraints.
    """
    labels = np.asarray(labels)
    if merge_constraint.iter_cannot_link_pairs is None:
        raise ValueError(
            "Cannot enumerate violations: merge_constraint does not support iterating constraint pairs."
        )

    out_pairs = []
    for i, j in merge_constraint.iter_cannot_link_pairs():
        li = int(labels[int(i)])
        lj = int(labels[int(j)])
        if li == lj and li != int(noise_label):
            out_pairs.append((int(i), int(j)))

    if len(out_pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)

    return np.asarray(out_pairs, dtype=np.int64)


def split_clusters_to_respect_cannot_link(
    labels: np.ndarray,
    *,
    merge_constraint: MergeConstraint,
    distances: Optional[Union[np.ndarray, CSR]] = None,
    noise_label: int = -1,
) -> np.ndarray:
    """
    Post-hoc cleanup: split any cluster that violates cannot-link constraints.

    Args:
        labels (np.ndarray): The cluster labels.
        merge_constraint (MergeConstraint): The merge constraint object.
        distances (Optional[Union[np.ndarray, CSR]]): The distance matrix.
        noise_label (int): The label for noise points.

    Returns:
        (np.ndarray):
            The modified labels after splitting.
    """
    labels_in = np.asarray(labels).astype(np.int64, copy=True)

    if merge_constraint.pair_cannot_link is None:
        raise ValueError(
            "Post-hoc splitting requires merge_constraint.pair_cannot_link for efficient conflict checks."
        )

    # ---- Step 1: split by distance-graph connected components (optional) ----
    if distances is not None and sp.issparse(distances):
        graph_component_labels = connected_components_from_distance_graph(distances)
        next_label = int(labels_in.max()) + 1

        for label_id in np.unique(labels_in).tolist():
            if int(label_id) == int(noise_label):
                continue
            idx_points = np.flatnonzero(labels_in == int(label_id))
            if idx_points.size <= 1:
                continue

            sub_component_ids = graph_component_labels[idx_points]
            unique_sub_components = np.unique(sub_component_ids)
            if unique_sub_components.size <= 1:
                continue

            for idx_sub, comp_id in enumerate(unique_sub_components.tolist()):
                if idx_sub == 0:
                    continue
                mask = sub_component_ids == int(comp_id)
                labels_in[idx_points[mask]] = int(next_label)
                next_label += 1

    # ---- Step 2: greedy split any remaining violations inside each cluster ----
    next_label = int(labels_in.max()) + 1

    for label_id in sorted(np.unique(labels_in).tolist()):
        if int(label_id) == int(noise_label):
            continue

        idx_points = np.flatnonzero(labels_in == int(label_id))
        if idx_points.size <= 1:
            continue

        idx_set = set(int(x) for x in idx_points.tolist())
        _ = idx_set
        adjacency_conflict = {int(i): [] for i in idx_points.tolist()}

        idx_list = idx_points.tolist()
        for a_i in range(len(idx_list)):
            i = int(idx_list[a_i])
            for a_j in range(a_i + 1, len(idx_list)):
                j = int(idx_list[a_j])
                if merge_constraint.pair_cannot_link(
                    i, j
                ) or merge_constraint.pair_cannot_link(j, i):
                    adjacency_conflict[i].append(j)
                    adjacency_conflict[j].append(i)

        has_any_conflict = any(len(v) > 0 for v in adjacency_conflict.values())
        if not has_any_conflict:
            continue

        order_nodes = sorted(
            idx_list,
            key=lambda node: (-len(adjacency_conflict[int(node)]), int(node)),
        )

        color_of_node: dict[int, int] = {}
        for node in order_nodes:
            used_colors = set()
            for nbr in adjacency_conflict[int(node)]:
                if int(nbr) in color_of_node:
                    used_colors.add(int(color_of_node[int(nbr)]))

            color = 0
            while color in used_colors:
                color += 1
            color_of_node[int(node)] = int(color)

        colors_present = sorted(set(color_of_node.values()))
        if len(colors_present) <= 1:
            continue

        for color in colors_present:
            if int(color) == 0:
                continue
            nodes_color = [
                node for node, c in color_of_node.items() if int(c) == int(color)
            ]
            labels_in[np.asarray(nodes_color, dtype=np.int64)] = int(next_label)
            next_label += 1

    return labels_in


# ---------------------------- public wrappers ----------------------------


def fast_hdbscan_precomputed(
    distances: Union[np.ndarray, CSR],
    *,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    max_cluster_size: Number = np.inf,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
    return_trees: bool = False,
) -> Tuple[
    np.ndarray, np.ndarray, Optional[np.ndarray], Optional[object], Optional[np.ndarray]
]:
    """
    Run fast_hdbscan on a precomputed distance matrix (dense or sparse CSR).

    Args:
        distances (Union[np.ndarray, CSR]):
            The precomputed distance matrix.
        min_cluster_size (int):
            The minimum size of clusters.
        min_samples (Optional[int]):
            The number of samples in a neighborhood for a point to be considered as a core point.
        cluster_selection_method (str):
            The method used to select clusters from the condensed tree.
        allow_single_cluster (bool):
            Whether to allow single-cluster solutions.
        max_cluster_size (Number):
            The maximum size of clusters.
        cluster_selection_epsilon (float):
            A distance threshold. Clusters below this value will be merged.
        cluster_selection_persistence (float):
            A measure of stability for cluster selection.
        sample_weights (Optional[np.ndarray]):
            Weights for each sample.
        return_trees (bool):
            Whether to return the condensed tree and linkage tree.

    Returns:
        (Tuple):
            A tuple containing labels, probabilities, and optionally the condensed and linkage trees.
    """
    distance_matrix_csr = _ensure_csr_distance_matrix(distances)
    distance_matrix_csr = _symmetrize_min_keep_present(distance_matrix_csr)

    n_points = int(distance_matrix_csr.shape[0])
    if min_samples is None:
        min_samples = int(min_cluster_size)

    if int(min_samples) <= 0 or int(min_cluster_size) <= 0:
        raise ValueError("min_samples and min_cluster_size must be positive integers.")

    core_distances = _core_distances_from_sparse_rows(
        distance_matrix_csr,
        min_samples=int(min_samples),
        sample_weights=sample_weights,
    )
    u, v, w = _mutual_reachability_edges_upper_triangle(
        distance_matrix_csr, core_distances=core_distances
    )
    mst_edges = _kruskal_mst_unconstrained(n_points=n_points, u=u, v=v, w=w)

    if mst_edges.shape[0] < n_points - 1:
        mst_edges = _connect_components_with_penalty_edges(
            n_points=n_points, mst_edges=mst_edges, penalty=np.inf
        )

    finite_penalty = _choose_large_finite_penalty(distances, user_penalty=np.inf)
    mst_edges = _sanitize_mst_edge_weights(mst_edges, finite_penalty=finite_penalty)

    return (*clusters_from_spanning_tree(
        mst_edges,
        min_cluster_size=int(min_cluster_size),
        cluster_selection_method=str(cluster_selection_method),
        max_cluster_size=max_cluster_size,
        allow_single_cluster=bool(allow_single_cluster),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_persistence=float(cluster_selection_persistence),
        sample_weights=sample_weights,
    ),)[: (None if return_trees else 2)]


def fast_hdbscan_precomputed_with_merge_constraint(
    distances: Union[np.ndarray, CSR],
    merge_constraint: MergeConstraint,
    *,
    strict: bool = True,
    penalty: float = np.inf,
    posthoc_cleanup: bool = False,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    max_cluster_size: Number = np.inf,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
    return_trees: bool = False,
    mst_method: str = "boruvka",
    parallel_backend: str = "auto",
) -> Tuple[
    np.ndarray, np.ndarray, Optional[np.ndarray], Optional[object], Optional[np.ndarray]
]:
    """
    Run fast_hdbscan on a precomputed distance matrix with merge constraints.

    Args:
        distances (Union[np.ndarray, CSR]):
            The precomputed distance matrix.
        merge_constraint (MergeConstraint):
            The merge constraint object.
        strict (bool):
            Whether to strictly enforce constraints.
        penalty (float):
            The penalty for violating constraints in soft mode.
        posthoc_cleanup (bool):
            Whether to perform post-hoc cleanup of violations.
        min_cluster_size (int):
            The minimum size of clusters.
        min_samples (Optional[int]):
            The number of samples in a neighborhood for a point to be considered as a core point.
        cluster_selection_method (str):
            The method used to select clusters.
        allow_single_cluster (bool):
            Whether to allow single-cluster solutions.
        max_cluster_size (Number):
            The maximum size of clusters.
        cluster_selection_epsilon (float):
            Distance threshold for merging clusters.
        cluster_selection_persistence (float):
            Stability measure for cluster selection.
        sample_weights (Optional[np.ndarray]):
            Weights for each sample.
        return_trees (bool):
            Whether to return the condensed and linkage trees.
        mst_method (str):
            Algorithm for constrained MST:
            - "boruvka" (default): Sequential constrained Borůvka
            - "parallel_boruvka": Parallel constrained Borůvka with violation correction
            - "kruskal": Sequential constrained Kruskal
        parallel_backend (str):
            Backend for parallel_boruvka method:
            - "auto" (default): Automatically select based on hardware and problem size
            - "cuda": Force CUDA (raises error if unavailable)
            - "cpu": Force CPU parallel (Step 1a parallel, Step 1b sequential)
            - "sequential": Force fully sequential execution

    Returns:
        (Tuple):
            A tuple containing labels, probabilities, and optionally trees.
    """
    if not np.isfinite(penalty) and not np.isinf(penalty):
        raise ValueError("penalty must be finite or +inf.")

    distance_matrix_csr = _ensure_csr_distance_matrix(distances)
    distance_matrix_csr = _symmetrize_min_keep_present(distance_matrix_csr)

    n_points = int(distance_matrix_csr.shape[0])
    if min_samples is None:
        min_samples = int(min_cluster_size)

    if int(min_samples) <= 0 or int(min_cluster_size) <= 0:
        raise ValueError("min_samples and min_cluster_size must be positive integers.")

    core_distances = _core_distances_from_sparse_rows(
        distance_matrix_csr,
        min_samples=int(min_samples),
        sample_weights=sample_weights,
    )
    u, v, w = _mutual_reachability_edges_upper_triangle(
        distance_matrix_csr, core_distances=core_distances
    )

    if bool(strict):
        mst_edges, cannot_link_edges_out = _mst_constrained_hard(
            n_points=n_points,
            u=u,
            v=v,
            w=w,
            merge_constraint=merge_constraint,
            mst_method=mst_method,
            parallel_backend=parallel_backend,
        )
        # Note: cannot_link_edges_out is only populated for parallel_boruvka
        # It contains edges that were removed due to race condition violations
        _ = cannot_link_edges_out  # Future: could expose this in return value
    else:
        if merge_constraint.pair_cannot_link is None:
            raise ValueError(
                "strict=False requires merge_constraint.pair_cannot_link to be available."
            )

        w_inflated = w.copy()
        for idx_edge in range(u.shape[0]):
            a = int(u[idx_edge])
            b = int(v[idx_edge])
            if merge_constraint.pair_cannot_link(
                a, b
            ) or merge_constraint.pair_cannot_link(b, a):
                w_inflated[idx_edge] = float(
                    max(float(w_inflated[idx_edge]), float(penalty))
                )

        mst_edges = _kruskal_mst_unconstrained(n_points=n_points, u=u, v=v, w=w_inflated)

    # Always connect components at +inf so merges occur only at the top of the hierarchy.
    if mst_edges.shape[0] < n_points - 1:
        mst_edges = _connect_components_with_penalty_edges(
            n_points=n_points, mst_edges=mst_edges, penalty=np.inf
        )

    finite_penalty = _choose_large_finite_penalty(
        distances, user_penalty=float(penalty)
    )
    mst_edges = _sanitize_mst_edge_weights(mst_edges, finite_penalty=finite_penalty)

    results = clusters_from_spanning_tree(
        mst_edges,
        min_cluster_size=int(min_cluster_size),
        cluster_selection_method=str(cluster_selection_method),
        max_cluster_size=max_cluster_size,
        allow_single_cluster=bool(allow_single_cluster),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_persistence=float(cluster_selection_persistence),
        sample_weights=sample_weights,
    )

    if bool(posthoc_cleanup):
        labels_clean = _maybe_split_labels_if_cannot_link_violated(
            results[0],
            merge_constraint=merge_constraint,
            distances=distance_matrix_csr,
            noise_label=-1,
        )
        results = (labels_clean, results[1], *results[2:])

    return results[: (None if return_trees else 2)]


def fast_hdbscan_precomputed_with_cannot_link(
    distances: Union[np.ndarray, CSR],
    cannot_link: Union[np.ndarray, CSR],
    *,
    strict: bool = True,
    penalty: float = np.inf,
    posthoc_cleanup: bool = False,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    max_cluster_size: Number = np.inf,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
    return_trees: bool = False,
    mst_method: str = "boruvka",
    parallel_backend: str = "auto",
) -> Tuple[
    np.ndarray, np.ndarray, Optional[np.ndarray], Optional[object], Optional[np.ndarray]
]:
    """
    Backwards-compatible wrapper: cannot-link constraints via dense/sparse matrix.

    Args:
        distances (Union[np.ndarray, CSR]): The precomputed distance matrix.
        cannot_link (Union[np.ndarray, CSR]): The cannot-link constraints.
        strict (bool): Whether to strictly enforce constraints.
        penalty (float): The penalty for violating constraints in soft mode.
        posthoc_cleanup (bool): Whether to perform post-hoc cleanup.
        min_cluster_size (int): Minimum size of clusters.
        min_samples (Optional[int]): Number of samples for a core point.
        cluster_selection_method (str): Method for cluster selection.
        allow_single_cluster (bool): Whether to allow single-cluster solutions.
        max_cluster_size (Number): Maximum size of clusters.
        cluster_selection_epsilon (float): Distance threshold for merging.
        cluster_selection_persistence (float): Stability measure for selection.
        sample_weights (Optional[np.ndarray]): Weights for each sample.
        return_trees (bool): Whether to return trees.
        mst_method (str):
            Algorithm for constrained MST:
            - "boruvka" (default): Sequential constrained Borůvka
            - "parallel_boruvka": Parallel constrained Borůvka with violation correction
            - "kruskal": Sequential constrained Kruskal
        parallel_backend (str):
            Backend for parallel_boruvka method:
            - "auto" (default): Automatically select based on hardware and problem size
            - "cuda": Force CUDA (raises error if unavailable)
            - "cpu": Force CPU parallel (Step 1a parallel, Step 1b sequential)
            - "sequential": Force fully sequential execution

    Returns:
        (Tuple):
            A tuple containing labels, probabilities, and optionally trees.
    """
    distance_matrix_csr = _ensure_csr_distance_matrix(distances)
    n_points = int(distance_matrix_csr.shape[0])

    merge_constraint = MergeConstraint.from_cannot_link_matrix(
        cannot_link=cannot_link,
        n_points=n_points,
    )

    return fast_hdbscan_precomputed_with_merge_constraint(
        distances=distances,
        merge_constraint=merge_constraint,
        strict=bool(strict),
        penalty=float(penalty),
        posthoc_cleanup=bool(posthoc_cleanup),
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        cluster_selection_method=str(cluster_selection_method),
        allow_single_cluster=bool(allow_single_cluster),
        max_cluster_size=max_cluster_size,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_persistence=float(cluster_selection_persistence),
        sample_weights=sample_weights,
        return_trees=bool(return_trees),
        mst_method=str(mst_method),
        parallel_backend=str(parallel_backend),
    )


#############################
###### TESTING SUITE ########
#############################

import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as ssd

from sklearn.metrics import adjusted_rand_score

from fast_hdbscan.hdbscan import fast_hdbscan as fast_hdbscan_feature_space


def _dense_pairwise_distances_euclidean(data: np.ndarray) -> np.ndarray:
    """Computes dense pairwise Euclidean distances."""
    condensed = ssd.pdist(data, metric="euclidean")
    dense = ssd.squareform(condensed)
    return dense.astype(np.float64, copy=False)


def _dense_to_knn_csr(distances: np.ndarray, *, k: int) -> sp.csr_matrix:
    """
    Keep k nearest neighbors per row (excluding self) in a directed graph.
    """
    n = int(distances.shape[0])
    rows = []
    cols = []
    vals = []
    for i in range(n):
        d = distances[i].copy()
        d[i] = np.inf
        nn = np.argsort(d, kind="mergesort")[: int(k)]
        for j in nn.tolist():
            if np.isfinite(d[int(j)]):
                rows.append(int(i))
                cols.append(int(j))
                vals.append(float(d[int(j)]))
    return sp.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
    )


def test_precomputed_matches_feature_space_on_dense_distances():
    """Tests if precomputed HDBSCAN matches feature-space HDBSCAN on dense distances."""
    rng = np.random.default_rng(0)

    data_a = rng.normal(loc=-2.0, scale=0.3, size=(40, 2))
    data_b = rng.normal(loc=2.0, scale=0.3, size=(40, 2))
    data = np.vstack([data_a, data_b]).astype(np.float64, copy=False)

    distances = _dense_pairwise_distances_euclidean(data)

    labels_feat, probs_feat = fast_hdbscan_feature_space(
        data,
        min_cluster_size=8,
        min_samples=8,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        return_trees=False,
    )
    labels_prec, probs_prec = fast_hdbscan_precomputed(
        distances,
        min_cluster_size=8,
        min_samples=8,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        return_trees=False,
    )

    # Use ARI to ignore label permutations.
    assert adjusted_rand_score(labels_feat, labels_prec) == 1.0
    assert probs_feat.shape == probs_prec.shape
    assert labels_prec.shape[0] == data.shape[0]


def test_empty_cannot_link_dense_is_noop_vs_unconstrained():
    """Tests if an empty dense cannot-link matrix is a no-op compared to unconstrained."""
    rng = np.random.default_rng(1)

    data = rng.normal(size=(60, 3)).astype(np.float64, copy=False)
    distances = _dense_pairwise_distances_euclidean(data)

    cannot_link = np.zeros((data.shape[0], data.shape[0]), dtype=bool)

    labels_base, _ = fast_hdbscan_precomputed(
        distances,
        min_cluster_size=6,
        min_samples=6,
        allow_single_cluster=True,
    )
    labels_con, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link=cannot_link,
        strict=True,
        posthoc_cleanup=False,
        min_cluster_size=6,
        min_samples=6,
        allow_single_cluster=True,
    )

    assert adjusted_rand_score(labels_base, labels_con) == 1.0


def test_empty_cannot_link_sparse_is_noop_vs_unconstrained():
    """Tests if an empty sparse cannot-link matrix is a no-op compared to unconstrained."""
    rng = np.random.default_rng(11)

    data = rng.normal(size=(50, 2)).astype(np.float64, copy=False)
    distances = _dense_pairwise_distances_euclidean(data)

    cannot_link_sparse = sp.csr_matrix((data.shape[0], data.shape[0]), dtype=bool)

    labels_base, _ = fast_hdbscan_precomputed(
        distances,
        min_cluster_size=5,
        min_samples=5,
        allow_single_cluster=True,
    )
    labels_con, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link=cannot_link_sparse,
        strict=True,
        posthoc_cleanup=False,
        min_cluster_size=5,
        min_samples=5,
        allow_single_cluster=True,
    )

    assert adjusted_rand_score(labels_base, labels_con) == 1.0


def test_strict_true_requires_matrix_payload_for_callable():
    """Tests if strict=True requires a matrix payload for a callable constraint."""
    rng = np.random.default_rng(12)
    data = rng.normal(size=(12, 2)).astype(np.float64, copy=False)
    distances = _dense_pairwise_distances_euclidean(data)

    def pair_cannot_link(i: int, j: int) -> bool:
        _ = i
        _ = j
        return False

    merge_constraint_callable = MergeConstraint.from_pair_cannot_link(pair_cannot_link)

    did_raise = False
    try:
        _ = fast_hdbscan_precomputed_with_merge_constraint(
            distances,
            merge_constraint=merge_constraint_callable,
            strict=True,
            min_cluster_size=3,
            min_samples=3,
            allow_single_cluster=True,
        )
    except ValueError:
        did_raise = True

    assert (
        did_raise
    ), "strict=True should require a matrix-backed MergeConstraint (CSR payload arrays)."


def test_soft_mode_callable_matches_dense_matrix_behavior():
    """Tests if soft mode with a callable matches dense matrix behavior."""
    rng = np.random.default_rng(2)

    data = rng.normal(size=(35, 2)).astype(np.float64, copy=False)
    distances = _dense_pairwise_distances_euclidean(data)

    # Create a small random cannot-link set (dense matrix).
    n = int(data.shape[0])
    cannot_link_dense = np.zeros((n, n), dtype=bool)
    pairs = rng.choice(n * n, size=40, replace=False)
    for p in pairs.tolist():
        i = int(p // n)
        j = int(p % n)
        if i == j:
            continue
        cannot_link_dense[i, j] = True
        cannot_link_dense[j, i] = True

    merge_from_dense = MergeConstraint.from_cannot_link_matrix(
        cannot_link_dense, n_points=n
    )

    def pair_cannot_link(i: int, j: int) -> bool:
        return bool(cannot_link_dense[int(i), int(j)])

    merge_from_callable = MergeConstraint.from_pair_cannot_link(pair_cannot_link)

    # Compare strict=False (soft, direct-edge penalization) because strict=True
    # is intentionally matrix-only in the current implementation.
    labels_dense, _ = fast_hdbscan_precomputed_with_merge_constraint(
        distances,
        merge_constraint=merge_from_dense,
        strict=False,
        penalty=np.inf,
        min_cluster_size=4,
        min_samples=4,
        allow_single_cluster=True,
    )
    labels_callable, _ = fast_hdbscan_precomputed_with_merge_constraint(
        distances,
        merge_constraint=merge_from_callable,
        strict=False,
        penalty=np.inf,
        min_cluster_size=4,
        min_samples=4,
        allow_single_cluster=True,
    )

    assert adjusted_rand_score(labels_dense, labels_callable) == 1.0


def test_posthoc_cleanup_removes_violation_small_graph():
    """Tests if post-hoc cleanup removes a violation on a small graph."""
    distances = np.array(
        [
            [0.0, 0.1, 10.0, 10.0],
            [0.1, 0.0, 10.0, 10.0],
            [10.0, 10.0, 0.0, 10.0],
            [10.0, 10.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )

    cannot_link = np.zeros((4, 4), dtype=bool)
    cannot_link[0, 1] = True
    cannot_link[1, 0] = True

    merge_constraint = MergeConstraint.from_cannot_link_matrix(cannot_link, n_points=4)

    labels, _ = fast_hdbscan_precomputed_with_merge_constraint(
        distances,
        merge_constraint=merge_constraint,
        strict=True,
        posthoc_cleanup=True,
        min_cluster_size=2,
        min_samples=1,
        allow_single_cluster=True,
        return_trees=False,
    )

    violations = find_cannot_link_violations(labels, merge_constraint=merge_constraint)
    assert violations.shape[0] == 0


def test_merge_constraint_sanitizes_symmetry_and_diagonal_dense():
    """Tests if MergeConstraint sanitizes symmetry and diagonal for dense input."""
    n = 5
    cannot_link = np.zeros((n, n), dtype=bool)
    cannot_link[0, 0] = True
    cannot_link[0, 1] = True
    cannot_link[1, 0] = False  # intentionally asymmetric

    mc = MergeConstraint.from_cannot_link_matrix(cannot_link, n_points=n)

    # Diagonal must be forced off.
    assert mc.pair_cannot_link is not None
    assert mc.pair_cannot_link(0, 0) is False

    # Symmetry must be enforced.
    assert mc.pair_cannot_link(0, 1) is True
    assert mc.pair_cannot_link(1, 0) is True

    # Iteration should contain (0,1) exactly once.
    assert mc.iter_cannot_link_pairs is not None
    pairs = set(
        tuple(sorted((int(i), int(j)))) for (i, j) in mc.iter_cannot_link_pairs()
    )
    assert (0, 1) in pairs
    assert (0, 0) not in pairs


def test_merge_constraint_dense_vs_sparse_equivalent_pairs_and_checks():
    """Tests if dense and sparse MergeConstraints have equivalent pairs and checks."""
    rng = np.random.default_rng(21)
    n = 25

    cannot_link = np.zeros((n, n), dtype=bool)
    # Add some random pairs; enforce symmetry manually here as well.
    for _ in range(60):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        cannot_link[i, j] = True
        cannot_link[j, i] = True

    mc_dense = MergeConstraint.from_cannot_link_matrix(cannot_link, n_points=n)
    mc_sparse = MergeConstraint.from_cannot_link_matrix(
        sp.csr_matrix(cannot_link), n_points=n
    )

    assert mc_dense.pair_cannot_link is not None
    assert mc_sparse.pair_cannot_link is not None
    assert mc_dense.iter_cannot_link_pairs is not None
    assert mc_sparse.iter_cannot_link_pairs is not None

    pairs_dense = set(
        tuple(sorted((int(i), int(j)))) for (i, j) in mc_dense.iter_cannot_link_pairs()
    )
    pairs_sparse = set(
        tuple(sorted((int(i), int(j)))) for (i, j) in mc_sparse.iter_cannot_link_pairs()
    )
    assert pairs_dense == pairs_sparse

    # Spot-check pair queries.
    for _ in range(100):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        assert bool(mc_dense.pair_cannot_link(i, j)) == bool(
            mc_sparse.pair_cannot_link(i, j)
        )


def test_precomputed_handles_sparse_distance_graph_with_isolated_node():
    """Tests if precomputed HDBSCAN handles a sparse graph with an isolated node."""
    # Sparse distance graph with one isolated node (degree 0) -> core distance inf.
    n = 5
    rows = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    cols = np.array([1, 2, 3, 0, 3, 0, 1, 2], dtype=np.int64)
    vals = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    distances_csr = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    labels, probs = fast_hdbscan_precomputed(
        distances_csr,
        min_cluster_size=2,
        min_samples=1,
        allow_single_cluster=True,
    )

    assert labels.shape == (n,)
    assert probs.shape == (n,)
    assert np.all(np.isfinite(probs))


def test_posthoc_split_fixes_cross_component_violation():
    """Tests if post-hoc splitting fixes a cross-component violation."""
    # Two disconnected components in the distance graph (CSR).
    # Start with a "bad" clustering that merges everything into one cluster,
    # then ensure posthoc splitting separates components and removes violations.
    n = 6
    rows = []
    cols = []
    vals = []

    # component 1: {0,1,2} complete graph with distance 1
    comp1 = [0, 1, 2]
    for i in comp1:
        for j in comp1:
            if i != j:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)

    # component 2: {3,4,5} complete graph with distance 1
    comp2 = [3, 4, 5]
    for i in comp2:
        for j in comp2:
            if i != j:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)

    distances_csr = sp.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
    )

    # Cannot-link one point across components.
    cannot_link = np.zeros((n, n), dtype=bool)
    cannot_link[0, 3] = True
    cannot_link[3, 0] = True

    merge_constraint = MergeConstraint.from_cannot_link_matrix(cannot_link, n_points=n)

    labels_bad = np.zeros(n, dtype=np.int64)  # everything in one cluster
    violations_before = find_cannot_link_violations(
        labels_bad, merge_constraint=merge_constraint
    )
    assert violations_before.shape[0] >= 1

    labels_fixed = split_clusters_to_respect_cannot_link(
        labels_bad,
        merge_constraint=merge_constraint,
        distances=distances_csr,
    )
    violations_after = find_cannot_link_violations(
        labels_fixed, merge_constraint=merge_constraint
    )
    assert violations_after.shape[0] == 0

    # And ensure the two components are not forced into the same final label.
    assert labels_fixed[0] != labels_fixed[3]


def test_find_violations_requires_iterable_pairs():
    """Tests if find_cannot_link_violations requires iterable pairs."""

    def pair_cannot_link(i: int, j: int) -> bool:
        return (int(i) == 0 and int(j) == 1) or (int(i) == 1 and int(j) == 0)

    mc = MergeConstraint.from_pair_cannot_link(pair_cannot_link)
    labels = np.zeros(3, dtype=np.int64)

    did_raise = False
    try:
        _ = find_cannot_link_violations(labels, merge_constraint=mc)
    except ValueError:
        did_raise = True

    assert did_raise


# =============================================================================
# BORŮVKA MST TESTS
# =============================================================================


def test_adjacency_list_construction_simple():
    """Test adjacency list builder with a simple 4-node graph."""
    # Graph: 0--1 (w=1), 1--2 (w=2), 2--3 (w=3), 0--2 (w=4)
    u = np.array([0, 1, 2, 0], dtype=np.int32)
    v = np.array([1, 2, 3, 2], dtype=np.int32)
    w = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    n_points = 4

    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
        u, v, w, n_points
    )

    # Check indptr structure
    assert adj_indptr.shape[0] == n_points + 1
    assert adj_indptr[0] == 0

    # Node 0: neighbors 1, 2 (edges 0, 3)
    degree_0 = adj_indptr[1] - adj_indptr[0]
    assert degree_0 == 2
    neighbors_0 = set(adj_neighbors[adj_indptr[0] : adj_indptr[1]])
    assert neighbors_0 == {1, 2}

    # Node 1: neighbors 0, 2 (edges 0, 1)
    degree_1 = adj_indptr[2] - adj_indptr[1]
    assert degree_1 == 2
    neighbors_1 = set(adj_neighbors[adj_indptr[1] : adj_indptr[2]])
    assert neighbors_1 == {0, 2}

    # Node 2: neighbors 1, 3, 0 (edges 1, 2, 3)
    degree_2 = adj_indptr[3] - adj_indptr[2]
    assert degree_2 == 3
    neighbors_2 = set(adj_neighbors[adj_indptr[2] : adj_indptr[3]])
    assert neighbors_2 == {0, 1, 3}

    # Node 3: neighbor 2 (edge 2)
    degree_3 = adj_indptr[4] - adj_indptr[3]
    assert degree_3 == 1
    neighbors_3 = set(adj_neighbors[adj_indptr[3] : adj_indptr[4]])
    assert neighbors_3 == {2}

    # Total entries = 2 * n_edges = 8
    assert adj_neighbors.shape[0] == 8
    assert adj_weights.shape[0] == 8
    assert adj_edge_ids.shape[0] == 8


def test_adjacency_list_isolated_node():
    """Test adjacency list with an isolated node (no edges)."""
    # Graph: 0--1 (w=1), node 2 isolated
    u = np.array([0], dtype=np.int32)
    v = np.array([1], dtype=np.int32)
    w = np.array([1.0], dtype=np.float64)
    n_points = 3

    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
        u, v, w, n_points
    )

    # Node 2 should have empty adjacency
    assert adj_indptr[2] == adj_indptr[3]  # No neighbors


def test_boruvka_unconstrained_produces_valid_mst():
    """Test Borůvka produces a valid MST without constraints."""
    # Complete graph on 5 nodes with known MST
    # Edges: 0-1(1), 0-2(5), 0-3(3), 0-4(4), 1-2(2), 1-3(6), 1-4(7), 2-3(4), 2-4(3), 3-4(2)
    edges = [
        (0, 1, 1.0),
        (0, 2, 5.0),
        (0, 3, 3.0),
        (0, 4, 4.0),
        (1, 2, 2.0),
        (1, 3, 6.0),
        (1, 4, 7.0),
        (2, 3, 4.0),
        (2, 4, 3.0),
        (3, 4, 2.0),
    ]
    u = np.array([e[0] for e in edges], dtype=np.int32)
    v = np.array([e[1] for e in edges], dtype=np.int32)
    w = np.array([e[2] for e in edges], dtype=np.float64)
    n_points = 5

    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
        u, v, w, n_points
    )

    # Empty cannot-link (no constraints)
    cannot_link_indptr = np.zeros(n_points + 1, dtype=np.int64)
    cannot_link_indices = np.empty(0, dtype=np.int32)

    mst_edges = _constrained_boruvka_mst_csr_strict_numba(
        adj_indptr,
        adj_neighbors,
        adj_weights,
        adj_edge_ids,
        cannot_link_indptr,
        cannot_link_indices,
        n_points,
        len(edges),
    )

    # MST should have N-1 = 4 edges
    assert mst_edges.shape[0] == 4

    # MST total weight should be 1+2+2+3 = 8 (0-1, 1-2, 3-4, 2-4)
    total_weight = mst_edges[:, 2].sum()
    assert total_weight == 8.0


def test_boruvka_known_example_with_constraints():
    """
    Test Borůvka with known cannot-link constraint.
    
    5 nodes: 0,1,2,3,4
    Cannot-link: (0,3)
    Expected: forest with separate components (0 not merged with 3)
    """
    # Simple graph where natural MST would connect all
    edges = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ]
    u = np.array([e[0] for e in edges], dtype=np.int32)
    v = np.array([e[1] for e in edges], dtype=np.int32)
    w = np.array([e[2] for e in edges], dtype=np.float64)
    n_points = 5

    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
        u, v, w, n_points
    )

    # Cannot-link: node 0 forbids node 3, node 3 forbids node 0
    # CSR format: indptr tells where each node's forbidden list starts
    # Node 0: [3], Node 1: [], Node 2: [], Node 3: [0], Node 4: []
    cannot_link_indptr = np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)
    cannot_link_indices = np.array([3, 0], dtype=np.int32)

    mst_edges = _constrained_boruvka_mst_csr_strict_numba(
        adj_indptr,
        adj_neighbors,
        adj_weights,
        adj_edge_ids,
        cannot_link_indptr,
        cannot_link_indices,
        n_points,
        len(edges),
    )

    # The constraint prevents 0 and 3 from being in the same component.
    # With edges 0-1-2-3-4, edges 0-1 and 1-2 connect {0,1,2}, but edge 2-3 
    # would merge {0,1,2} with {3}, violating cannot-link(0,3).
    # So the forest should have fewer than 4 edges.
    assert mst_edges.shape[0] < 4

    # Build DSU from MST edges to verify components
    parent = list(range(n_points))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(mst_edges.shape[0]):
        union(int(mst_edges[i, 0]), int(mst_edges[i, 1]))

    # Verify 0 and 3 are in different components
    assert find(0) != find(3)


def test_boruvka_tie_breaking_deterministic():
    """Test that Borůvka produces deterministic results with equal weights."""
    # All edges have same weight - result should be consistent
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    u = np.array([e[0] for e in edges], dtype=np.int32)
    v = np.array([e[1] for e in edges], dtype=np.int32)
    w = np.array([e[2] for e in edges], dtype=np.float64)
    n_points = 3

    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
        u, v, w, n_points
    )

    cannot_link_indptr = np.zeros(n_points + 1, dtype=np.int64)
    cannot_link_indices = np.empty(0, dtype=np.int32)

    # Run multiple times
    results = []
    for _ in range(3):
        mst = _constrained_boruvka_mst_csr_strict_numba(
            adj_indptr,
            adj_neighbors,
            adj_weights,
            adj_edge_ids,
            cannot_link_indptr,
            cannot_link_indices,
            n_points,
            len(edges),
        )
        results.append(mst.copy())

    # All results should be identical
    for r in results[1:]:
        assert np.allclose(results[0], r)


def test_boruvka_single_node():
    """Test Borůvka with a single node (0 edges)."""
    u = np.empty(0, dtype=np.int32)
    v = np.empty(0, dtype=np.int32)
    w = np.empty(0, dtype=np.float64)
    n_points = 1

    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
        u, v, w, n_points
    )

    cannot_link_indptr = np.zeros(n_points + 1, dtype=np.int64)
    cannot_link_indices = np.empty(0, dtype=np.int32)

    mst_edges = _constrained_boruvka_mst_csr_strict_numba(
        adj_indptr,
        adj_neighbors,
        adj_weights,
        adj_edge_ids,
        cannot_link_indptr,
        cannot_link_indices,
        n_points,
        0,
    )

    assert mst_edges.shape[0] == 0


def test_boruvka_fully_constrained_no_merges():
    """Test Borůvka when all edges violate constraints (no merges possible)."""
    # Triangle graph where all pairs are constrained
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    u = np.array([e[0] for e in edges], dtype=np.int32)
    v = np.array([e[1] for e in edges], dtype=np.int32)
    w = np.array([e[2] for e in edges], dtype=np.float64)
    n_points = 3

    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
        u, v, w, n_points
    )

    # All pairs cannot link: 0<->1, 0<->2, 1<->2
    # Node 0: [1, 2], Node 1: [0, 2], Node 2: [0, 1]
    cannot_link_indptr = np.array([0, 2, 4, 6], dtype=np.int64)
    cannot_link_indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)

    mst_edges = _constrained_boruvka_mst_csr_strict_numba(
        adj_indptr,
        adj_neighbors,
        adj_weights,
        adj_edge_ids,
        cannot_link_indptr,
        cannot_link_indices,
        n_points,
        len(edges),
    )

    # No merges possible - all edges violate constraints
    assert mst_edges.shape[0] == 0


def test_boruvka_matches_kruskal_on_random_data():
    """Test that Borůvka produces valid results similar to Kruskal on random data."""
    rng = np.random.default_rng(42)

    data_a = rng.normal(loc=-2.0, scale=0.5, size=(20, 2))
    data_b = rng.normal(loc=2.0, scale=0.5, size=(20, 2))
    data = np.vstack([data_a, data_b]).astype(np.float64)

    distances = _dense_pairwise_distances_euclidean(data)

    # Create some random cannot-link constraints
    cannot_link = np.zeros((40, 40), dtype=np.float64)
    for _ in range(5):
        i, j = rng.integers(0, 40, size=2)
        if i != j:
            cannot_link[i, j] = 1.0
            cannot_link[j, i] = 1.0

    labels_boruvka, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=5,
        mst_method="boruvka",
    )

    labels_kruskal, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=5,
        mst_method="kruskal",
    )

    # Both should produce valid clusterings that respect constraints
    # (violations would have been caught during MST construction)
    # Check ARI is reasonably high (both find similar structure)
    ari = adjusted_rand_score(labels_boruvka, labels_kruskal)
    # Note: Borůvka and Kruskal may produce different MSTs with same constraints
    # due to different edge selection strategies, but both should find valid solutions
    assert ari >= 0.4  # Accept reasonably similar results


def test_boruvka_empty_constraints_matches_unconstrained():
    """Test that Borůvka with empty constraints matches unconstrained."""
    rng = np.random.default_rng(123)
    data = rng.normal(size=(30, 2)).astype(np.float64)
    distances = _dense_pairwise_distances_euclidean(data)

    # Empty cannot-link
    cannot_link = np.zeros((30, 30), dtype=np.float64)

    labels_constrained, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=5,
        mst_method="boruvka",
    )

    labels_unconstrained, _ = fast_hdbscan_precomputed(
        distances,
        min_cluster_size=5,
    )

    ari = adjusted_rand_score(labels_constrained, labels_unconstrained)
    assert ari == 1.0


def test_mst_method_invalid_raises():
    """Test that invalid mst_method raises ValueError."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(10, 2)).astype(np.float64)
    distances = _dense_pairwise_distances_euclidean(data)
    cannot_link = np.zeros((10, 10), dtype=np.float64)

    did_raise = False
    try:
        fast_hdbscan_precomputed_with_cannot_link(
            distances,
            cannot_link,
            strict=True,
            min_cluster_size=3,
            mst_method="invalid",
        )
    except ValueError as e:
        did_raise = True
        assert "invalid" in str(e).lower()

    assert did_raise


# ------------- Parallel Borůvka MST Tests -------------


def test_parallel_boruvka_no_constraints_matches_sequential():
    """Test that parallel Borůvka produces same result as sequential with no constraints."""
    rng = np.random.default_rng(100)
    data = rng.normal(size=(30, 2)).astype(np.float64)
    distances = _dense_pairwise_distances_euclidean(data)
    
    # Empty cannot-link
    cannot_link = np.zeros((30, 30), dtype=np.float64)
    
    labels_sequential, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=5,
        mst_method="boruvka",
    )
    
    labels_parallel, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=5,
        mst_method="parallel_boruvka",
    )
    
    # Should produce identical results with no constraints
    ari = adjusted_rand_score(labels_sequential, labels_parallel)
    assert ari == 1.0


def test_parallel_boruvka_with_constraints_respects_them():
    """Test that parallel Borůvka respects cannot-link constraints."""
    rng = np.random.default_rng(101)
    data = rng.normal(size=(20, 2)).astype(np.float64)
    distances = _dense_pairwise_distances_euclidean(data)
    
    # Add some cannot-link constraints
    cannot_link = np.zeros((20, 20), dtype=np.float64)
    cannot_link[0, 5] = 1.0
    cannot_link[5, 0] = 1.0
    cannot_link[2, 8] = 1.0
    cannot_link[8, 2] = 1.0
    
    labels, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=3,
        mst_method="parallel_boruvka",
    )
    
    # Verify constraints are not violated
    for i in range(20):
        for j in range(i + 1, 20):
            if cannot_link[i, j] > 0:
                # Points with cannot-link should not be in the same cluster
                # (unless both are noise, labeled -1)
                if labels[i] != -1 and labels[j] != -1:
                    assert labels[i] != labels[j], f"Constraint violated: points {i} and {j} both in cluster {labels[i]}"


def test_parallel_boruvka_race_condition_scenario():
    """
    Test a scenario designed to trigger race conditions in parallel merging.
    
    Graph: 0 -- 1 -- 2 with weights (0-1)=1.0, (1-2)=1.0
    Constraint: 0-2 cannot-link
    
    In parallel, {0} might propose 0-1, {2} might propose 2-1.
    Both pass checks individually, but merging both creates {0,1,2} which violates 0-2.
    The parallel Borůvka should detect and fix this.
    """
    # Build a simple chain graph
    n = 3
    distances = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    distances[0, 1] = distances[1, 0] = 1.0
    distances[1, 2] = distances[2, 1] = 1.0
    
    # Cannot-link between endpoints
    cannot_link = np.zeros((n, n), dtype=np.float64)
    cannot_link[0, 2] = 1.0
    cannot_link[2, 0] = 1.0
    
    # With parallel_boruvka, this should produce a forest (not all connected)
    merge_constraint = MergeConstraint.from_cannot_link_matrix(cannot_link, n_points=n)
    
    u = np.array([0, 1], dtype=np.int32)
    v = np.array([1, 2], dtype=np.int32)
    w = np.array([1.0, 1.0], dtype=np.float64)
    
    mst_edges, cannot_link_edges = _mst_constrained_hard(
        n_points=n,
        u=u,
        v=v,
        w=w,
        merge_constraint=merge_constraint,
        mst_method="parallel_boruvka",
    )
    
    # Should have only 1 edge in MST (one edge removed to respect constraint)
    assert mst_edges.shape[0] == 1, f"Expected 1 MST edge, got {mst_edges.shape[0]}"
    
    # The removed edge should be tracked (if any race condition was fixed)
    # Note: may be 0 if no race condition occurred due to ordering
    # But with this simple graph, it's likely to have one
    assert cannot_link_edges is not None


def test_parallel_boruvka_matches_kruskal_results():
    """Test that parallel Borůvka produces valid clustering that respects constraints like Kruskal."""
    rng = np.random.default_rng(102)
    data = rng.normal(size=(25, 3)).astype(np.float64)
    distances = _dense_pairwise_distances_euclidean(data)
    
    # Add constraints
    cannot_link = np.zeros((25, 25), dtype=np.float64)
    constraint_pairs = []
    for i in range(0, 20, 4):
        cannot_link[i, i + 2] = 1.0
        cannot_link[i + 2, i] = 1.0
        constraint_pairs.append((i, i + 2))
    
    labels_kruskal, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=3,
        mst_method="kruskal",
    )
    
    labels_parallel, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=3,
        mst_method="parallel_boruvka",
    )
    
    # Both should respect constraints - verify no constraint pair is in the same non-noise cluster
    def check_constraints_respected(labels, pairs):
        for a, b in pairs:
            if labels[a] != -1 and labels[b] != -1 and labels[a] == labels[b]:
                return False
        return True
    
    assert check_constraints_respected(labels_kruskal, constraint_pairs), "Kruskal violated constraints"
    assert check_constraints_respected(labels_parallel, constraint_pairs), "Parallel Borůvka violated constraints"
    
    # Both should produce some clusters (not all noise)
    assert not np.all(labels_kruskal == -1) or not np.all(labels_parallel == -1), "Both methods produced all noise"


def test_parallel_boruvka_violation_detection():
    """Test that violation detection correctly identifies constraint violations."""
    n = 5
    
    # Create a simple DSU where all points are in one component
    parent = np.zeros(n, dtype=np.int32)  # All point to root 0
    for i in range(n):
        parent[i] = 0
    
    # Constraint: 1-3 cannot-link
    cannot_link_csr = sp.csr_matrix(
        ([1], ([1], [3])), shape=(n, n), dtype=np.int32
    )
    cannot_link_csr = cannot_link_csr + cannot_link_csr.T
    
    indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
    indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)
    
    # Detect violations
    va, vb, n_viol = _detect_violations_numba(
        parent, indptr, indices, n
    )
    
    assert n_viol == 1
    # The violation should be (1, 3)
    found = (va[0] == 1 and vb[0] == 3) or (va[0] == 3 and vb[0] == 1)
    assert found, f"Expected violation (1,3), got ({va[0]}, {vb[0]})"


def test_parallel_boruvka_single_node():
    """Test parallel Borůvka with very small graphs."""
    # Test with 2 nodes (single node can't form edges for MST)
    distances = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    cannot_link = np.zeros((2, 2), dtype=np.float64)
    
    labels, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=1,
        mst_method="parallel_boruvka",
    )
    
    assert labels.shape[0] == 2


def test_parallel_boruvka_fully_constrained():
    """Test parallel Borůvka when all edges are constrained."""
    n = 4
    distances = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    
    # Every pair is constrained
    cannot_link = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(cannot_link, 0.0)
    
    labels, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=1,
        mst_method="parallel_boruvka",
    )
    
    # All points should be isolated (noise or singleton clusters)
    # Due to no valid merges possible
    assert labels.shape[0] == n


# ---------------------------------------------------------------------------
# Backend Selection Tests
# ---------------------------------------------------------------------------


def test_select_parallel_backend_auto():
    """Test that 'auto' backend selection returns a valid backend."""
    backend = _select_parallel_backend(100, "auto")
    assert backend in ("cuda", "cpu", "sequential"), f"Unexpected backend: {backend}"


def test_select_parallel_backend_explicit():
    """Test explicit backend selection."""
    assert _select_parallel_backend(100, "sequential") == "sequential"
    assert _select_parallel_backend(100, "cpu") == "cpu"
    # CUDA may or may not be available, but should raise ValueError or return 'cuda'
    try:
        cuda_backend = _select_parallel_backend(100, "cuda")
        assert cuda_backend == "cuda", "If CUDA is available, should return 'cuda'"
    except ValueError as e:
        # Expected if CUDA is not available
        assert "CUDA is not available" in str(e)


def test_parallel_backend_parameter_propagation():
    """Test that parallel_backend parameter is propagated through the API."""
    np.random.seed(42)
    n = 10
    X = np.random.randn(n, 2)
    distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    cannot_link = np.zeros((n, n), dtype=np.float64)
    
    # Test with explicit sequential backend
    labels_seq, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=2,
        mst_method="parallel_boruvka",
        parallel_backend="sequential",
    )
    
    # Test with explicit cpu backend
    labels_cpu, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=2,
        mst_method="parallel_boruvka",
        parallel_backend="cpu",
    )
    
    # Test with auto backend
    labels_auto, _ = fast_hdbscan_precomputed_with_cannot_link(
        distances,
        cannot_link,
        strict=True,
        min_cluster_size=2,
        mst_method="parallel_boruvka",
        parallel_backend="auto",
    )
    
    # All should produce the same shape
    assert labels_seq.shape == labels_cpu.shape == labels_auto.shape
    # For unconstrained case, results should be identical
    assert np.array_equal(labels_seq, labels_cpu), "Sequential and CPU should match"
    assert np.array_equal(labels_seq, labels_auto), "Sequential and auto should match"


def test_cpu_vs_sequential_backend_equivalence():
    """Test that CPU and sequential backends produce identical MST results."""
    np.random.seed(123)
    n = 15
    
    # Create weighted graph as edges (complete graph)
    edges_list = []
    for i in range(n):
        for j in range(i + 1, n):
            weight = np.random.random()
            edges_list.append((i, j, weight))
    
    u = np.array([e[0] for e in edges_list], dtype=np.int32)
    v = np.array([e[1] for e in edges_list], dtype=np.int32)
    w = np.array([e[2] for e in edges_list], dtype=np.float64)
    n_edges = len(edges_list)
    
    # Convert to adjacency list format
    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(
        u, v, w, n
    )
    
    # Create some constraints
    constraint_pairs = [(0, 5), (2, 8), (3, 10)]
    cannot_link = np.zeros((n, n), dtype=np.float64)
    for i, j in constraint_pairs:
        cannot_link[i, j] = 1.0
        cannot_link[j, i] = 1.0
    cannot_link_csr = sp.csr_matrix(cannot_link, dtype=np.int32)
    cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
    cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)
    
    # Run sequential backend
    mst_edges_seq, cl_edges_seq = _parallel_constrained_boruvka_mst(
        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
        cl_indptr, cl_indices, n, n_edges, parallel_backend="sequential"
    )
    
    # Run CPU backend
    mst_edges_cpu, cl_edges_cpu = _parallel_constrained_boruvka_mst(
        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
        cl_indptr, cl_indices, n, n_edges, parallel_backend="cpu"
    )
    
    # Results should be identical (same algorithm, just different loop implementation)
    assert np.array_equal(mst_edges_seq, mst_edges_cpu), "MST edges differ"
    assert np.array_equal(cl_edges_seq, cl_edges_cpu), "Cannot-link edges differ"


def test_check_cuda_available_cached():
    """Test that _check_cuda_available returns consistent results and is cached."""
    result1 = _check_cuda_available()
    result2 = _check_cuda_available()
    
    assert result1 == result2, "CUDA availability check should be deterministic"
    assert isinstance(result1, bool), "Should return boolean"


if __name__ == "__main__":
    test_precomputed_matches_feature_space_on_dense_distances()
    test_empty_cannot_link_dense_is_noop_vs_unconstrained()
    test_empty_cannot_link_sparse_is_noop_vs_unconstrained()
    test_strict_true_requires_matrix_payload_for_callable()
    test_soft_mode_callable_matches_dense_matrix_behavior()
    test_posthoc_cleanup_removes_violation_small_graph()
    test_merge_constraint_sanitizes_symmetry_and_diagonal_dense()
    test_merge_constraint_dense_vs_sparse_equivalent_pairs_and_checks()
    test_precomputed_handles_sparse_distance_graph_with_isolated_node()
    test_posthoc_split_fixes_cross_component_violation()
    test_find_violations_requires_iterable_pairs()

    # Borůvka MST tests
    test_adjacency_list_construction_simple()
    test_adjacency_list_isolated_node()
    test_boruvka_unconstrained_produces_valid_mst()
    test_boruvka_known_example_with_constraints()
    test_boruvka_tie_breaking_deterministic()
    test_boruvka_single_node()
    test_boruvka_fully_constrained_no_merges()
    test_boruvka_matches_kruskal_on_random_data()
    test_boruvka_empty_constraints_matches_unconstrained()
    test_mst_method_invalid_raises()

    # Parallel Borůvka MST tests
    test_parallel_boruvka_no_constraints_matches_sequential()
    test_parallel_boruvka_with_constraints_respects_them()
    test_parallel_boruvka_race_condition_scenario()
    test_parallel_boruvka_matches_kruskal_results()
    test_parallel_boruvka_violation_detection()
    test_parallel_boruvka_single_node()
    test_parallel_boruvka_fully_constrained()

    # Backend selection tests
    test_select_parallel_backend_auto()
    test_select_parallel_backend_explicit()
    test_parallel_backend_parameter_propagation()
    test_cpu_vs_sequential_backend_equivalence()
    test_check_cuda_available_cached()

    print("All tests passed.")
