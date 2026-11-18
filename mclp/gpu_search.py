from __future__ import annotations

import os
import zipfile
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from numba import cuda

from . import cuda_state as base

THREADS = 8 if os.environ.get('NUMBA_ENABLE_CUDASIM') == '1' else 128
TILE = 2 * THREADS
NEG = -2147480000


@dataclass
class WeightedInstance:
    coords: np.ndarray  # shape (n,2), latitude, longitude
    weights: np.ndarray  # int32 demand weights
    adj: csr_matrix  # shape (n,n), facility x demand, int32 data
    words_t: np.ndarray  # (ceil(n/32), n), uint32, facility-bitsets

    @property
    def n(self):
        return int(len(self.weights))

    def value(self, sites):
        sites = np.asarray(sites, dtype=np.int32)
        if sites.size == 0:
            return 0
        # CSR only visits covered demand positions; duplicate demand positions
        # are removed by unique(), which is independent from the GPU kernels.
        indices = self.adj[sites, :].indices
        return int(self.weights[np.unique(indices)].sum(dtype=np.int64))

    def greedy(self, p: int):
        """Deterministic weighted greedy. No best-known solutions are used."""
        uncovered = self.weights.copy()
        chosen = []
        for _ in range(p):
            gains = np.asarray(self.adj @ uncovered).reshape(-1)
            if chosen:
                gains[np.asarray(chosen, np.int32)] = -1
            site = int(np.argmax(gains))
            chosen.append(site)
            a, b = int(self.adj.indptr[site]), int(self.adj.indptr[site + 1])
            uncovered[self.adj.indices[a:b]] = 0
        return tuple(sorted(chosen))


def load_repo_file(repo_zip: Path, suffix: str) -> str:
    with zipfile.ZipFile(repo_zip) as z:
        matches = [n for n in z.namelist() if n.endswith('/' + suffix)]
        if len(matches) != 1:
            raise ValueError(f'Expected one {suffix} inside {repo_zip}; got {matches}')
        return z.read(matches[0]).decode('utf-8-sig')


def build_adjacency(coords: np.ndarray, weights: np.ndarray, radius: float):
    """Exactly match Java: float32(acos(spherical cosine)*6378100) <= radius.

    Floating point identities are checked using the authors' saved solution.
    Coordinates are LATITUDE, LONGITUDE, in degrees (not planar coordinates).
    """
    n = len(weights)
    rad = np.deg2rad(coords)
    coslat = np.cos(rad[:, 0])
    sinlat = np.sin(rad[:, 0])
    lon = rad[:, 1]
    A = np.empty((n, n), np.bool_)
    chunk = max(32, min(256, 2_000_000 // max(n, 1)))
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        cos_dlon = np.cos(lon[lo:hi, None] - lon[None, :])
        dot = (sinlat[lo:hi, None] * sinlat[None, :] +
               coslat[lo:hi, None] * coslat[None, :] * cos_dlon)
        d_java_float = (np.arccos(np.clip(dot, -1.0, 1.0)) * 6378100.0).astype(np.float32)
        A[lo:hi] = d_java_float <= radius
    if not np.all(np.diag(A)):
        raise AssertionError('A facility must cover its own demand point')
    # Same orientation as the old cuda kernel: packed facility coverage in columns.
    pad = (-n) % 32
    packed = np.packbits(A, axis=1, bitorder='little')
    if pad:
        packed = np.pad(packed, ((0, 0), (0, pad // 8)))
    words = np.ascontiguousarray(packed).view('<u4').reshape(n, -1)
    words_t = np.ascontiguousarray(words.T)
    adj = csr_matrix(A, dtype=np.int32)
    return WeightedInstance(coords, weights, adj, words_t)


def diverse_seeds(base_sites: tuple[int, ...], n: int, p: int,
                  starts: int, seed: int, pool_idx: int):
    """Same straightforward pool diversification as the existing GR-ILS.

    Do not load the authors' best-known facilities as a search starting point.
    """
    rng = np.random.default_rng(seed + 1009 * pool_idx)
    output = []
    for b in range(starts):
        if pool_idx == 0 and b == 0:
            output.append(base_sites)
            continue
        # Pool 0: increasing diversity; later pools: all randomized.
        frac = 0.15 + 0.8 * (b + (pool_idx > 0)) / max(starts, 1)
        k = min(p, max(1, int(np.ceil(p * min(0.95, frac)))))
        keep = set(base_sites)
        drop = rng.choice(base_sites, size=k, replace=False)
        keep.difference_update(int(x) for x in drop)
        pool = np.asarray([i for i in range(n) if i not in keep], np.int32)
        add = rng.choice(pool, size=p - len(keep), replace=False)
        keep.update(int(x) for x in add)
        output.append(tuple(sorted(keep)))
    return output


if cuda is not None:
    @cuda.jit
    def _owners(bits, selected, counts, active, owner):
        t = cuda.grid(1)
        m = counts.shape[1]
        if t >= counts.shape[0] * m:
            return
        b = t // m
        j = t % m
        owner[b, j] = -1
        if active[b] == 0 or counts[b, j] != 1:
            return
        wd = j >> 5
        mask = np.uint32(1) << np.uint32(j & 31)
        for slot in range(selected.shape[1]):
            site = selected[b, slot]
            if site >= 0 and (bits[wd, site] & mask) != 0:
                owner[b, j] = slot
                break


    @cuda.jit
    def _losses_weighted(ptr, idx, weights, selected, counts, active, losses):
        t = cuda.grid(1)
        p = selected.shape[1]
        if t >= selected.shape[0] * p:
            return
        b = t // p
        slot = t % p
        if active[b] == 0:
            losses[b, slot] = 0
            return
        site = selected[b, slot]
        s = 0
        for k in range(ptr[site], ptr[site + 1]):
            j = idx[k]
            if counts[b, j] == 1:
                s += weights[j]
        losses[b, slot] = s


    @cuda.jit
    def _swap_tiles(ptr, idx, weights, chosen, counts, owner, losses,
                    active, tile_gains, tile_drops, tile_adds):
        b = cuda.blockIdx.x
        tile = cuda.blockIdx.y
        t = cuda.threadIdx.x
        p = losses.shape[1]
        n = chosen.shape[1]
        best_gain = NEG
        best_slot = -1
        best_add = -1
        # Each thread handles two or fewer add-site candidates in its tile.
        if active[b] != 0:
            for site in range(tile * TILE + t, min((tile + 1) * TILE, n), cuda.blockDim.x):
                if chosen[b, site] != 0:
                    continue
                overlaps = cuda.local.array(128, dtype=np.int32)
                for s in range(p):
                    overlaps[s] = 0
                uncovered_gain = 0
                for k in range(ptr[site], ptr[site + 1]):
                    j = idx[k]
                    c = counts[b, j]
                    if c == 0:
                        uncovered_gain += weights[j]
                    elif c == 1:
                        s = owner[b, j]
                        if s >= 0:
                            overlaps[s] += weights[j]
                for s in range(p):
                    gain = uncovered_gain + overlaps[s] - losses[b, s]
                    if (gain > best_gain or (gain == best_gain and
                                             (best_slot < 0 or s < best_slot or
                                              (s == best_slot and site < best_add)))):
                        best_gain, best_slot, best_add = gain, s, site
        gs = cuda.shared.array(THREADS, dtype=np.int32)
        ds = cuda.shared.array(THREADS, dtype=np.int32)
        adds = cuda.shared.array(THREADS, dtype=np.int32)
        gs[t], ds[t], adds[t] = best_gain, best_slot, best_add
        cuda.syncthreads()
        stride = cuda.blockDim.x // 2
        while stride:
            if t < stride:
                g, s, a = gs[t + stride], ds[t + stride], adds[t + stride]
                if (g > gs[t] or (g == gs[t] and s >= 0 and
                                  (ds[t] < 0 or s < ds[t] or (s == ds[t] and a < adds[t])))):
                    gs[t], ds[t], adds[t] = g, s, a
            cuda.syncthreads()
            stride //= 2
        if t == 0:
            tile_gains[b, tile] = gs[0]
            tile_drops[b, tile] = ds[0]
            tile_adds[b, tile] = adds[0]


    @cuda.jit
    def _apply_weighted_swap(selected, chosen, values, best_values,
                             best_selected, active, kicks, max_kicks,
                             tile_gains, tile_drops, tile_adds,
                             drops, adds, changed, kick_pending, swap_counts):
        b = cuda.grid(1)
        if b >= selected.shape[0]:
            return
        changed[b] = 0
        kick_pending[b] = 0
        if active[b] == 0:
            return
        gain, slot, add = NEG, -1, -1
        for tile in range(tile_gains.shape[1]):
            g, r, a = tile_gains[b, tile], tile_drops[b, tile], tile_adds[b, tile]
            if a >= 0 and (g > gain or (g == gain and
                                        (slot < 0 or r < slot or (r == slot and a < add)))):
                gain, slot, add = g, r, a
        if gain > 0:
            old = selected[b, slot]
            drops[b], adds[b] = old, add
            selected[b, slot] = add
            for s in range(1, selected.shape[1]):
                v = selected[b, s]
                t = s - 1
                while t >= 0 and selected[b, t] > v:
                    selected[b, t + 1] = selected[b, t]
                    t -= 1
                selected[b, t + 1] = v
            chosen[b, old] = 0
            chosen[b, add] = 1
            values[b] += gain
            changed[b] = 1
            swap_counts[b] += 1
            if values[b] > best_values[b]:
                best_values[b] = values[b]
                for s in range(selected.shape[1]):
                    best_selected[b, s] = selected[b, s]
        elif kicks[b] < max_kicks:
            kick_pending[b] = 1
            kicks[b] += 1
        else:
            active[b] = 0


    @cuda.jit
    def _repair_tiles(ptr, idx, weights, counts, chosen, tabu,
                      pending, tile_gain, tile_site):
        b = cuda.blockIdx.x
        tile = cuda.blockIdx.y
        t = cuda.threadIdx.x
        n = chosen.shape[1]
        best_gain = -1
        best_site = -1
        if pending[b] != 0:
            for site in range(tile * TILE + t, min((tile + 1) * TILE, n), cuda.blockDim.x):
                if chosen[b, site] != 0 or tabu[b, site] != 0:
                    continue
                gain = 0
                for k in range(ptr[site], ptr[site + 1]):
                    j = idx[k]
                    if counts[b, j] == 0:
                        gain += weights[j]
                if gain > best_gain or (gain == best_gain and (best_site < 0 or site < best_site)):
                    best_gain, best_site = gain, site
        gs = cuda.shared.array(THREADS, dtype=np.int32)
        ss = cuda.shared.array(THREADS, dtype=np.int32)
        gs[t], ss[t] = best_gain, best_site
        cuda.syncthreads()
        stride = cuda.blockDim.x // 2
        while stride:
            if t < stride:
                g, s = gs[t + stride], ss[t + stride]
                if g > gs[t] or (g == gs[t] and s >= 0 and (ss[t] < 0 or s < ss[t])):
                    gs[t], ss[t] = g, s
            cuda.syncthreads()
            stride //= 2
        if t == 0:
            tile_gain[b, tile] = gs[0]
            tile_site[b, tile] = ss[0]


    @cuda.jit
    def _select_repair(pending, tile_gain, tile_site, repair_add):
        b = cuda.grid(1)
        if b >= pending.shape[0] or pending[b] == 0:
            return
        gain, site = -1, -1
        for t in range(tile_gain.shape[1]):
            g, s = tile_gain[b, t], tile_site[b, t]
            if s >= 0 and (g > gain or (g == gain and (site < 0 or s < site))):
                gain, site = g, s
        repair_add[b] = site


    @cuda.jit
    def _finish_weighted(counts, weights, selected, values, best_values,
                         best_selected, pending):
        b = cuda.grid(1)
        if b >= selected.shape[0] or pending[b] == 0:
            return
        obj = 0
        for j in range(counts.shape[1]):
            if counts[b, j] > 0:
                obj += weights[j]
        for s in range(1, selected.shape[1]):
            v = selected[b, s]
            i = s - 1
            while i >= 0 and selected[b, i] > v:
                selected[b, i + 1] = selected[b, i]
                i -= 1
            selected[b, i + 1] = v
        values[b] = obj
        if obj > best_values[b]:
            best_values[b] = obj
            for s in range(selected.shape[1]):
                best_selected[b, s] = selected[b, s]
        pending[b] = 0


class WeightedResidentILS:
    """Keep weighted coverage counts and all trajectories GPU-resident."""

    def __init__(self, inst: WeightedInstance, p: int, starts: int, seed: int,
                 k: int, max_kicks: int):
        if not cuda.is_available():
            raise RuntimeError('A supported CUDA GPU is required (or use NUMBA_ENABLE_CUDASIM=1 for tests).')
        if p > 128:
            raise ValueError('p>128 not supported by local overlap array; use p<=128')
        self.inst, self.p, self.starts, self.seed = inst, p, starts, seed
        self.k, self.max_kicks = k, max_kicks if k else 0
        self.base_seed = inst.greedy(p)
        self.ntiles = (inst.n + TILE - 1) // TILE
        self.grid_counts = (starts * inst.n + THREADS - 1) // THREADS
        self.grid_slots = (starts * p + THREADS - 1) // THREADS
        self.grid_trajectories = (starts + THREADS - 1) // THREADS
        t0 = time.perf_counter()
        self.fac = cuda.to_device(inst.words_t)
        self.ptr = cuda.to_device(np.asarray(inst.adj.indptr, np.int32))
        self.idx = cuda.to_device(np.asarray(inst.adj.indices, np.int32))
        self.weights = cuda.to_device(inst.weights)
        b, n = starts, inst.n
        self.selected = cuda.device_array((b, p), np.int32)
        self.best_selected = cuda.device_array((b, p), np.int32)
        self.chosen = cuda.device_array((b, n), np.uint8)
        self.counts = cuda.device_array((b, n), np.int32)
        self.owner = cuda.device_array((b, n), np.int32)
        self.losses = cuda.device_array((b, p), np.int32)
        self.values = cuda.device_array(b, np.int32)
        self.best_values = cuda.device_array(b, np.int32)
        self.active = cuda.device_array(b, np.int32)
        self.kicks = cuda.device_array(b, np.int32)
        self.swaps = cuda.device_array(b, np.int32)
        self.changed = cuda.device_array(b, np.int32)
        self.kick_pending = cuda.device_array(b, np.int32)
        self.drops = cuda.device_array(b, np.int32)
        self.adds = cuda.device_array(b, np.int32)
        self.tabu = cuda.device_array((b, n), np.uint8)
        self.removed = cuda.device_array((b, max(1, k)), np.int32)
        self.random_state = cuda.device_array(b, np.uint32)
        self.gain_tile = cuda.device_array((b, self.ntiles), np.int32)
        self.drop_tile = cuda.device_array((b, self.ntiles), np.int32)
        self.add_tile = cuda.device_array((b, self.ntiles), np.int32)
        self.repair_gain_tile = cuda.device_array((b, self.ntiles), np.int32)
        self.repair_site_tile = cuda.device_array((b, self.ntiles), np.int32)
        self.repair_adds = cuda.device_array(b, np.int32)
        self.reset_pool(0)
        # All CUDA kernels are compiled before the time-limited search.
        self.one_round()
        cuda.synchronize()
        self.reset_pool(0)
        cuda.synchronize()
        self.setup_sec = time.perf_counter() - t0

    def reset_pool(self, pool_idx: int):
        selections = diverse_seeds(self.base_seed, self.inst.n, self.p,
                                   self.starts, self.seed, pool_idx)
        vals = np.asarray([self.inst.value(s) for s in selections], np.int32)
        s = np.asarray(selections, np.int32)
        chosen = np.zeros((self.starts, self.inst.n), np.uint8)
        for b, sites in enumerate(selections):
            chosen[b, list(sites)] = 1
        self.selected.copy_to_device(s)
        self.best_selected.copy_to_device(s)
        self.chosen.copy_to_device(chosen)
        self.values.copy_to_device(vals)
        self.best_values.copy_to_device(vals)
        self.active.copy_to_device(np.ones(self.starts, np.int32))
        self.kicks.copy_to_device(np.zeros(self.starts, np.int32))
        self.swaps.copy_to_device(np.zeros(self.starts, np.int32))
        self.tabu.copy_to_device(np.zeros((self.starts, self.inst.n), np.uint8))
        self.removed.copy_to_device(np.full((self.starts, max(1, self.k)), -1, np.int32))
        rng = np.random.default_rng(self.seed + 1009 * pool_idx + 717)
        self.random_state.copy_to_device(
            rng.integers(1, 2 ** 31, size=self.starts, dtype=np.uint32))
        base._init_counts[self.grid_counts, THREADS](self.fac, self.selected, self.counts)
        self.current_seeds = selections
        self.current_values = vals
        return selections, vals

    def one_round(self):
        _owners[self.grid_counts, THREADS](
            self.fac, self.selected, self.counts, self.active, self.owner)
        _losses_weighted[self.grid_slots, THREADS](
            self.ptr, self.idx, self.weights, self.selected, self.counts,
            self.active, self.losses)
        _swap_tiles[(self.starts, self.ntiles), THREADS](
            self.ptr, self.idx, self.weights, self.chosen, self.counts,
            self.owner, self.losses, self.active,
            self.gain_tile, self.drop_tile, self.add_tile)
        _apply_weighted_swap[self.grid_trajectories, THREADS](
            self.selected, self.chosen, self.values, self.best_values,
            self.best_selected, self.active, self.kicks, self.max_kicks,
            self.gain_tile, self.drop_tile, self.add_tile, self.drops,
            self.adds, self.changed, self.kick_pending, self.swaps)
        base._update_swap_counts[self.grid_counts, THREADS](
            self.fac, self.counts, self.changed, self.drops, self.adds)
        if self.k and self.max_kicks:
            base._destroy[self.grid_trajectories, THREADS](
                self.selected, self.chosen, self.random_state, self.kick_pending,
                self.removed, self.tabu, self.k)
            base._remove_counts[self.grid_counts, THREADS](
                self.fac, self.counts, self.kick_pending, self.removed, self.k)
            for _ in range(self.k):
                _repair_tiles[(self.starts, self.ntiles), THREADS](
                    self.ptr, self.idx, self.weights, self.counts,
                    self.chosen, self.tabu, self.kick_pending,
                    self.repair_gain_tile, self.repair_site_tile)
                _select_repair[self.grid_trajectories, THREADS](
                    self.kick_pending, self.repair_gain_tile,
                    self.repair_site_tile, self.repair_adds)
                base._insert[self.grid_trajectories, THREADS](
                    self.selected, self.chosen, self.kick_pending, self.repair_adds)
                base._add_counts[self.grid_counts, THREADS](
                    self.fac, self.counts, self.kick_pending, self.repair_adds)
            _finish_weighted[self.grid_trajectories, THREADS](
                self.counts, self.weights, self.selected, self.values,
                self.best_values, self.best_selected, self.kick_pending)

    def run(self, seconds: float, check_every: int, max_pools: int,
            max_rounds: int):
        initial_best = int(self.current_values.max())
        best, best_sites = initial_best, tuple(
            self.current_seeds[int(np.argmax(self.current_values))])
        history = [[0., best]]
        time_to_best = 0.
        t0 = time.perf_counter()
        rounds = 0
        pools_started, pools_completed = 1, 0
        swaps, kicks = 0, 0
        status = 'TIME_LIMIT'
        while True:
            if time.perf_counter() - t0 >= seconds:
                break
            if rounds >= max_rounds:
                status = 'ROUND_LIMIT'
                break
            for _ in range(min(check_every, max_rounds - rounds)):
                self.one_round()
                rounds += 1
            cuda.synchronize()
            elapsed = time.perf_counter() - t0
            vals = self.best_values.copy_to_host()
            winning = int(np.argmax(vals))
            current = int(vals[winning])
            if current > best:
                best = current
                best_sites = tuple(sorted(int(i) for i in
                                          self.best_selected[winning:winning + 1].copy_to_host()[0]))
                time_to_best = elapsed
                history.append([round(elapsed, 6), best])
            live = self.active.copy_to_host()
            if np.any(live):
                continue
            pools_completed += 1
            swaps += int(self.swaps.copy_to_host().sum())
            kicks += int(self.kicks.copy_to_host().sum())
            if max_pools and pools_completed >= max_pools:
                status = 'POOL_LIMIT'
                break
            if elapsed >= seconds:
                break
            pool_idx = pools_started
            seeds, seed_values = self.reset_pool(pool_idx)
            pools_started += 1
            ib = int(np.argmax(seed_values))
            if int(seed_values[ib]) > best:
                best = int(seed_values[ib])
                best_sites = seeds[ib]
                time_to_best = time.perf_counter() - t0
                history.append([round(time_to_best, 6), best])
        if pools_started > pools_completed:
            swaps += int(self.swaps.copy_to_host().sum())
            kicks += int(self.kicks.copy_to_host().sum())
        elapsed = time.perf_counter() - t0
        if self.inst.value(best_sites) != best:
            raise AssertionError(f'GPU weighted objective mismatch: {best} vs '
                                 f'{self.inst.value(best_sites)}')
        return dict(status=status, objective=best, facilities=list(best_sites),
                    initial_best=initial_best, time_sec=elapsed,
                    time_to_best_checkpoint_sec=time_to_best,
                    setup_sec=self.setup_sec, cold_total_sec=self.setup_sec + elapsed,
                    starts_per_pool=self.starts, pools_started=pools_started,
                    pools_completed=pools_completed, rounds=rounds,
                    accepted_1swaps=swaps, fixed_kicks=kicks,
                    history=history, backend='gpu_resident_weighted_sparse',
                    check_every=check_every, kick_size=self.k,
                    max_kicks_per_start=self.max_kicks,
                    gpu_device=str(getattr(cuda.current_context().device, 'name', 'CUDA simulator')))
