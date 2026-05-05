import numba
import numpy as np

from collections import namedtuple
from .variables import NUMBA_CACHE

RankDisjointSet = namedtuple("DisjointSet", ["parent", "rank"])
SizeDisjointSet = namedtuple("DisjointSet", ["parent", "size"])


@numba.njit(cache=NUMBA_CACHE)
def ds_rank_create(n_elements):
    return RankDisjointSet(np.arange(n_elements, dtype=np.int32), np.zeros(n_elements, dtype=np.int32))


@numba.njit(cache=NUMBA_CACHE)
def ds_size_create(n_elements):
    return SizeDisjointSet(np.arange(n_elements, dtype=np.int32), np.ones(n_elements, dtype=np.int32))


@numba.njit(cache=NUMBA_CACHE)
def ds_find(disjoint_set, x):
    while disjoint_set.parent[x] != x:
        x, disjoint_set.parent[x] = disjoint_set.parent[x], disjoint_set.parent[disjoint_set.parent[x]]

    return x


@numba.njit(cache=NUMBA_CACHE)
def ds_union_by_rank(disjoint_set, x, y):
    x = ds_find(disjoint_set, x)
    y = ds_find(disjoint_set, y)

    if x == y:
        return

    if disjoint_set.rank[x] < disjoint_set.rank[y]:
        x, y = y, x

    disjoint_set.parent[y] = x
    if disjoint_set.rank[x] == disjoint_set.rank[y]:
        disjoint_set.rank[x] += 1


@numba.njit(cache=NUMBA_CACHE)
def ds_find_readonly(parent, x):
    """Find root of x WITHOUT path compression (safe for parallel reads).

    Unlike ds_find, this operates on a raw parent array (not a DisjointSet
    namedtuple) and never writes to the array, making it safe for use inside
    numba.prange loops where multiple threads read concurrently.
    """
    while parent[x] != x:
        x = parent[x]
    return x


@numba.njit(cache=NUMBA_CACHE)
def ds_union_by_size(disjoint_set, x, y):
    x = ds_find(disjoint_set, x)
    y = ds_find(disjoint_set, y)

    if x == y:
        return

    if disjoint_set.size[x] < disjoint_set.size[y]:
        x, y = y, x

    disjoint_set.parent[y] = x
    disjoint_set.size[x] += disjoint_set.size[y]