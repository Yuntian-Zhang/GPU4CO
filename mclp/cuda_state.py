from __future__ import annotations
import numpy as np
from numba import cuda

@cuda.jit(device=True)
def _bit(fac, site, demand):
    return int((fac[demand >> 5, site] >> (demand & 31)) & np.uint32(1))


@cuda.jit(device=True)
def _next_rng(x):
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


@cuda.jit
def _init_counts(fac, selected, counts):
    t = cuda.grid(1)
    m = counts.shape[1]
    if t >= counts.shape[0] * m:
        return
    b, j = t // m, t % m
    c = 0
    for s in range(selected.shape[1]):
        c += _bit(fac, selected[b, s], j)
    counts[b, j] = c


@cuda.jit
def _update_swap_counts(fac, counts, changed, drops, adds):
    t = cuda.grid(1)
    m = counts.shape[1]
    if t >= counts.shape[0] * m:
        return
    b, j = t // m, t % m
    if changed[b]:
        counts[b, j] += _bit(fac, adds[b], j) - _bit(fac, drops[b], j)


@cuda.jit
def _destroy(selected, chosen, random_state, kick_pending,
             removed, tabu, k):
    b = cuda.grid(1)
    if b >= selected.shape[0] or kick_pending[b] == 0:
        return
    for t in range(k):
        prev = removed[b, t]
        if prev >= 0:
            tabu[b, prev] = 0
    x = random_state[b]
    for t in range(k):
        x = _next_rng(x)
        # Choose uniformly among the p-t still-populated slots.
        rank = int(x % np.uint32(selected.shape[1] - t))
        slot = -1
        for s in range(selected.shape[1]):
            if selected[b, s] >= 0:
                if rank == 0:
                    slot = s
                    break
                rank -= 1
        old = selected[b, slot]
        selected[b, slot] = -1
        chosen[b, old] = 0
        tabu[b, old] = 1
        removed[b, t] = old
    random_state[b] = x


@cuda.jit
def _remove_counts(fac, counts, kick_pending, removed, k):
    t = cuda.grid(1)
    m = counts.shape[1]
    if t >= counts.shape[0] * m:
        return
    b, j = t // m, t % m
    if kick_pending[b] != 0:
        c = counts[b, j]
        for s in range(k):
            c -= _bit(fac, removed[b, s], j)
        counts[b, j] = c


@cuda.jit
def _insert(selected, chosen, kick_pending, repair_adds):
    b = cuda.grid(1)
    if b >= selected.shape[0] or kick_pending[b] == 0:
        return
    site = repair_adds[b]
    for s in range(selected.shape[1]):
        if selected[b, s] == -1:
            selected[b, s] = site
            chosen[b, site] = 1
            break


@cuda.jit
def _add_counts(fac, counts, kick_pending, repair_adds):
    t = cuda.grid(1)
    m = counts.shape[1]
    if t >= counts.shape[0] * m:
        return
    b, j = t // m, t % m
    if kick_pending[b] != 0:
        counts[b, j] += _bit(fac, repair_adds[b], j)

