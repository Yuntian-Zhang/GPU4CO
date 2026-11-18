from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
from . import gpu_search as core

@dataclass(frozen=True)
class Elite:
    objective: int
    facilities: tuple[int, ...]


class DiverseEliteArchive:
    """Top-quality solutions with a minimum exchange distance.

    Distance d(A,B) = p - |A intersect B|. Near-duplicates are replaced only
    if improved. An archive entry never comes from published BKS solutions.
    """

    def __init__(self, p: int, capacity: int, min_distance: int,
                 relative_band: float = 0.03):
        if not (1 <= capacity and 0 <= min_distance <= p and
                0.0 < relative_band <= 1.0):
            raise ValueError('Invalid elite archive settings')
        self.p = p
        self.capacity = capacity
        self.min_distance = min_distance
        self.relative_band = relative_band
        self.items: list[Elite] = []
        self.offers = 0
        self.accepts = 0

    @staticmethod
    def distance(x: tuple[int, ...], y: tuple[int, ...]) -> int:
        return len(x) - len(set(x).intersection(y))

    def offer(self, sites, value: int) -> bool:
        self.offers += 1
        s = tuple(sorted(int(x) for x in sites))
        if len(s) != self.p or len(set(s)) != self.p:
            raise AssertionError('Invalid search result offered to elite archive')
        candidate = Elite(int(value), s)
        # Enforce diversity against *all* entries, not merely the closest one.
        # Exact duplicates are excluded even when min_distance is set to zero.
        threshold = max(1, self.min_distance)
        near = [i for i, item in enumerate(self.items)
                if self.distance(s, item.facilities) < threshold]
        if near:
            if any(candidate.objective <= self.items[i].objective for i in near):
                return False
            self.items = [item for i, item in enumerate(self.items) if i not in near]
            self.items.append(candidate)
        elif self.items and candidate.objective < int(
                self.items[0].objective * (1.0-self.relative_band)):
            return False
        elif len(self.items) < self.capacity:
            self.items.append(candidate)
        elif candidate.objective > self.items[-1].objective:
            self.items[-1] = candidate
        else:
            return False
        self.items.sort(key=lambda x: (-x.objective, x.facilities))
        self.accepts += 1
        # Once a stronger incumbent arrives, remove very poor archive members.
        if self.items:
            floor = int(self.items[0].objective * (1.0-self.relative_band))
            self.items = [x for x in self.items if x.objective >= floor]
        return True

    def info(self):
        distances = [self.distance(a.facilities,b.facilities)
                     for i,a in enumerate(self.items) for b in self.items[i+1:]]
        return dict(size=len(self.items),
                    mean_pairwise_distance=(float(np.mean(distances)) if distances else None),
                    min_pairwise_distance=(min(distances) if distances else None),
                    best=(self.items[0].objective if self.items else None),
                    worst=(self.items[-1].objective if self.items else None),
                    objectives=[x.objective for x in self.items],
                    offers=self.offers, accepts=self.accepts)


class EliteRestartSampler:
    """Generate a deterministic mix of fresh, elite-mutation, and recombined seeds.

    The restart distribution is fixed (not adaptive). The CPU performs seed
    construction only between GPU search pools. No reference BKS facilities
    or paper solutions enter this sampler.
    """

    def __init__(self, inst: core.WeightedInstance, p: int, seed: int,
                 base_sites: tuple[int, ...], elite_fraction: float,
                 recombine_fraction: float, perturb_sizes: tuple[int, ...],
                 rcl: int):
        if not (0 <= elite_fraction <= 1 and 0 <= recombine_fraction <= elite_fraction):
            raise ValueError('Require 0 <= recombine_fraction <= elite_fraction <= 1')
        if not perturb_sizes or any(not 1 <= k <= p for k in perturb_sizes):
            raise ValueError('Invalid elite restart perturb sizes')
        if rcl < 1:
            raise ValueError('rcl must be positive')
        self.inst, self.p, self.seed = inst, p, seed
        self.base_sites = base_sites
        self.elite_fraction = elite_fraction
        self.recombine_fraction = recombine_fraction
        self.perturb_sizes = tuple(perturb_sizes)
        self.rcl = rcl
        strengths = np.asarray(inst.adj @ inst.weights).ravel()
        # Search-relevant extra candidates: prefer sites with high *standalone*
        # weighted coverage without excluding an exploratory uniform component.
        self.strong_sites = np.argsort(-strengths, kind='stable')[:max(256, inst.n//4)].astype(np.int32)
        self.counts = {'fresh': 0, 'mutate': 0, 'recombine': 0, 'fallback': 0}

    def choose_parent(self, archive: DiverseEliteArchive, rng: np.random.Generator):
        # Favor high quality without collapsing the entire population to the #1 elite.
        if len(archive.items) == 1:
            return archive.items[0]
        idx = rng.integers(len(archive.items), size=2)
        return archive.items[min(map(int, idx))]

    def mutate(self, parent: Elite, rng: np.random.Generator, size: int):
        sites = set(parent.facilities)
        deleted = rng.choice(parent.facilities, size=size, replace=False)
        for s in deleted:
            sites.remove(int(s))
        banned = set(map(int, deleted))
        # Remove a fairly large fixed number of facilities to explore a new
        # basin. New sites come from strong-site and uniform distributions.
        n = self.inst.n
        for _ in range(size):
            for attempt in range(100):
                if rng.random() < .70:
                    cand = int(self.strong_sites[rng.integers(len(self.strong_sites))])
                else:
                    cand = int(rng.integers(n))
                if cand not in sites and (cand not in banned or attempt >= 40):
                    sites.add(cand)
                    break
            else:
                available = [i for i in range(n) if i not in sites]
                sites.add(int(rng.choice(available)))
        return tuple(sorted(sites))

    def recombine(self, one: Elite, two: Elite, rng: np.random.Generator):
        """Coverage-aware union recombination, using ONLY discovered elites.

        Retain the parents' intersection, greedily choose remaining sites from
        their union (with RCL randomness). This is a new search initialization,
        not an additional within-trajectory neighborhood operator.
        """
        one_set, two_set = set(one.facilities), set(two.facilities)
        selected = set(one_set & two_set)
        if len(selected) == self.p:
            return self.mutate(one, rng, self.perturb_sizes[0])
        candidates = sorted((one_set | two_set) - selected)
        remaining = self.inst.weights.copy()
        indptr, indices = self.inst.adj.indptr, self.inst.adj.indices
        for s in selected:
            remaining[indices[indptr[s]:indptr[s+1]]] = 0
        while len(selected) < self.p:
            # Only a small (<=2p) subset participates in recombination.
            vals = np.asarray(self.inst.adj[candidates] @ remaining).reshape(-1)
            top = np.argsort(-vals, kind='stable')[:min(self.rcl, len(candidates))]
            rank = int(rng.integers(len(top)))
            pos = int(top[rank])
            site = int(candidates.pop(pos))
            selected.add(site)
            remaining[indices[indptr[site]:indptr[site+1]]] = 0
        return tuple(sorted(selected))

    def generate(self, archive: DiverseEliteArchive, pool_idx: int, starts: int):
        # Same fresh-seed distribution as the old solver, with a distinct
        # deterministic RNG for elite-guided operations.
        independent = core.diverse_seeds(
            self.base_sites, self.inst.n, self.p, starts, self.seed, pool_idx)
        if pool_idx == 0 or not archive.items or self.elite_fraction == 0:
            self.counts['fresh'] += starts
            return independent
        rng = np.random.default_rng(self.seed + 1009*pool_idx + 781937)
        order = rng.permutation(starts)
        elite_n = min(starts, int(round(starts*self.elite_fraction)))
        recomb_n = min(elite_n, int(round(starts*self.recombine_fraction)))
        out = list(independent)
        for rank, slot in enumerate(order[:elite_n]):
            slot = int(slot)
            if rank < recomb_n and len(archive.items) > 1:
                parent = self.choose_parent(archive, rng)
                # Select a structurally different partner when possible.
                others = sorted(
                    (x for x in archive.items if x.facilities != parent.facilities),
                    key=lambda x: (-archive.distance(parent.facilities,x.facilities),-x.objective))
                if others:
                    partner = others[int(rng.integers(min(3,len(others))))]
                    out[slot] = self.recombine(parent, partner, rng)
                    self.counts['recombine'] += 1
                    continue
                self.counts['fallback'] += 1
            parent = self.choose_parent(archive, rng)
            size = self.perturb_sizes[(pool_idx+rank) % len(self.perturb_sizes)]
            out[slot] = self.mutate(parent, rng, size)
            self.counts['mutate'] += 1
        self.counts['fresh'] += starts-elite_n
        assert all(len(s)==self.p and len(set(s))==self.p and
                   all(0<=j<self.inst.n for j in s) for s in out)
        return out


class EliteWeightedResidentILS(core.WeightedResidentILS):
    """Original GPU-resident weighted solver + host elite pool at restarts.

    Intra-pool GPU kernels are IDENTICAL to the independently validated solver.
    """

    def __init__(self, inst, p, starts, seed, k, max_kicks,
                 restart_policy='elite', elite_size=12, min_distance=10,
                 elite_fraction=.75, recombine_fraction=.25,
                 perturb_sizes=(6,10,14,20), rcl=4):
        setup_started = time.perf_counter()
        self.restart_policy = restart_policy
        if restart_policy not in ('elite','independent'):
            raise ValueError('restart_policy must be elite or independent')
        self.archive = DiverseEliteArchive(p,elite_size,min_distance)
        self.sampler = EliteRestartSampler(inst,p,seed,inst.greedy(p),
                elite_fraction,recombine_fraction,perturb_sizes,rcl)
        self.pool_log = []
        super().__init__(inst,p,starts,seed,k,max_kicks)
        # Include host-side archive/sampler preparation in reported cold setup.
        self.cuda_engine_setup_sec = self.setup_sec
        self.setup_sec = time.perf_counter()-setup_started
        # The parent constructor initializes pool 0 twice for kernel warm-up;
        # count only the real pool-0 starts in search statistics.
        self.sampler.counts = {'fresh':starts, 'mutate':0,
                               'recombine':0, 'fallback':0}

    def reset_pool(self, pool_idx: int):
        # Respect the base constructor, which warms up with pool 0 twice.
        if self.restart_policy == 'independent':
            return super().reset_pool(pool_idx)
        selections = self.sampler.generate(self.archive,pool_idx,self.starts)
        vals = np.asarray([self.inst.value(s) for s in selections],np.int32)
        # In the base constructor self.inst exists before reset_pool(0).
        s = np.asarray(selections,np.int32)
        chosen = np.zeros((self.starts,self.sampler.inst.n),np.uint8)
        for b, sites in enumerate(selections):
            chosen[b,list(sites)] = 1
        self.selected.copy_to_device(s)
        self.best_selected.copy_to_device(s)
        self.chosen.copy_to_device(chosen)
        self.values.copy_to_device(vals)
        self.best_values.copy_to_device(vals)
        self.active.copy_to_device(np.ones(self.starts,np.int32))
        self.kicks.copy_to_device(np.zeros(self.starts,np.int32))
        self.swaps.copy_to_device(np.zeros(self.starts,np.int32))
        self.tabu.copy_to_device(np.zeros((self.starts,self.sampler.inst.n),np.uint8))
        self.removed.copy_to_device(np.full((self.starts,max(1,self.k)),-1,np.int32))
        rng = np.random.default_rng(self.seed+1009*pool_idx+717)
        self.random_state.copy_to_device(
            rng.integers(1,2**31,size=self.starts,dtype=np.uint32))
        core.base._init_counts[self.grid_counts,core.THREADS](
            self.fac,self.selected,self.counts)
        self.current_seeds = selections
        self.current_values = vals
        return selections,vals

    def _record_pool_elites(self):
        values = self.best_values.copy_to_host()
        selections = self.best_selected.copy_to_host()
        # Verify GPU state with independent sparse weighted evaluator.
        for val,sites in zip(values,selections):
            obj = int(self.inst.value(sites))
            if obj != int(val):
                raise AssertionError(f'GPU/CPU weighted objective mismatch: {int(val)} != {obj}')
            if self.restart_policy == 'elite':
                self.archive.offer(sites,int(val))
        return values, selections

    def run(self, seconds, check_every, max_pools, max_rounds,
            reference_bks=None, stop_at_bks=False, log_every_pools=25):
        if seconds <= 0 or check_every<1 or max_rounds<1:
            raise ValueError('Invalid budgets')
        initial_best = int(self.current_values.max())
        best, best_sites = initial_best, tuple(
            self.current_seeds[int(np.argmax(self.current_values))])
        history = [[0., best]]
        time_to_best = 0.0
        time_first_bks = 0.0 if reference_bks is not None and best>=reference_bks else None
        t0 = time.perf_counter()
        rounds = swaps = kicks = 0
        pools_started, pools_completed = 1, 0
        status = 'TIME_LIMIT'

        while True:
            if time.perf_counter()-t0 >= seconds:
                break
            if rounds >= max_rounds:
                status = 'ROUND_LIMIT'
                break
            for _ in range(min(check_every,max_rounds-rounds)):
                self.one_round()
                rounds += 1
            core.cuda.synchronize()
            elapsed = time.perf_counter()-t0
            vals = self.best_values.copy_to_host()
            wi = int(np.argmax(vals))
            current = int(vals[wi])
            if current > best:
                best = current
                best_sites = tuple(sorted(int(x) for x in
                    self.best_selected[wi:wi+1].copy_to_host()[0]))
                time_to_best = elapsed
                history.append([round(elapsed,6),best])
            if (reference_bks is not None and time_first_bks is None
                    and best>=reference_bks):
                time_first_bks = elapsed
                print(f'BKS REACHED: {best:,} at search t={elapsed:.3f}s',flush=True)
            if stop_at_bks and time_first_bks is not None:
                status = 'BKS_REACHED'
                break
            active = self.active.copy_to_host()
            if np.any(active):
                continue
            pools_completed += 1
            swaps += int(self.swaps.copy_to_host().sum())
            kicks += int(self.kicks.copy_to_host().sum())
            # Discover local-optimum elites from *all* GPU trajectories.
            self._record_pool_elites()
            self.pool_log.append([pools_completed, round(elapsed,4),best,
                                  len(self.archive.items)])
            if log_every_pools and pools_completed%log_every_pools==0:
                print(f'  completed pools={pools_completed}, '
                      f'best={best:,}, elites={len(self.archive.items)}, '
                      f't={elapsed:.1f}s',flush=True)
            if max_pools and pools_completed>=max_pools:
                status = 'POOL_LIMIT'
                break
            if elapsed >= seconds or rounds >= max_rounds:
                if rounds>=max_rounds:
                    status='ROUND_LIMIT'
                break
            new_seeds,new_vals = self.reset_pool(pools_started)
            pools_started += 1
            new_wi = int(np.argmax(new_vals))
            if int(new_vals[new_wi])>best:
                best = int(new_vals[new_wi])
                best_sites = new_seeds[new_wi]
                time_to_best = time.perf_counter()-t0
                history.append([round(time_to_best,6),best])
        if pools_started > pools_completed:
            swaps += int(self.swaps.copy_to_host().sum())
            kicks += int(self.kicks.copy_to_host().sum())
            self._record_pool_elites()
        elapsed = time.perf_counter()-t0
        if self.inst.value(best_sites) != best:
            raise AssertionError('Final returned objective and facilities disagree')
        return dict(status=status, objective=int(best), facilities=list(best_sites),
            initial_best=initial_best, time_sec=elapsed,
            time_to_best_checkpoint_sec=time_to_best,
            time_first_reach_bks_checkpoint_sec=time_first_bks,
            setup_sec=self.setup_sec,cuda_engine_setup_sec=self.cuda_engine_setup_sec,
            cold_total_sec=self.setup_sec+elapsed,
            starts_per_pool=self.starts,pools_started=pools_started,
            pools_completed=pools_completed,rounds=rounds,
            accepted_1swaps=swaps,fixed_kicks=kicks,
            history=history, backend='gpu_resident_weighted_sparse',
            restart_policy=self.restart_policy,elite_archive=self.archive.info(),
            seed_mixture=dict(self.sampler.counts),
            pool_log=self.pool_log,check_every=check_every,kick_size=self.k,
            max_kicks_per_start=self.max_kicks,
            gpu_device=str(getattr(core.cuda.current_context().device,'name','CUDA simulator')))

