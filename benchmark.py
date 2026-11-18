from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

from mclp.benchmarks import build_instance, check_reference, resolve_city

CASES = (
    'Kings:50', 'Kings:60', 'Kings:70', 'Kings:80', 'Kings:90',
    'SanFrancisco:50', 'SanFrancisco:70', 'Bronx:50', 'Bronx:70',
)
FIELDS = (
    'city', 'n', 'p', 'radius_m', 'method', 'seed', 'bks', 'objective',
    'gap_to_bks_pct', 'bks_hit', 'time_to_best_sec', 'search_sec', 'setup_sec',
    'objective_at_30s', 'objective_at_60s', 'objective_at_120s',
    'objective_at_300s', 'status',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=list(CASES),
                        metavar='CITY:P', help='e.g. Kings:50 Bronx:70')
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=2027)
    parser.add_argument('--seconds', type=float, default=300.0)
    parser.add_argument('--starts', type=int, default=32)
    parser.add_argument('--kick-size', type=int, default=4)
    parser.add_argument('--max-kicks', type=int, default=20)
    parser.add_argument('--check-every', type=int, default=5)
    parser.add_argument('--max-rounds', type=int, default=100_000_000)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('urban_benchmarks.zip'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-only', action='store_true',
                        help='Validate the published benchmark solutions without CUDA')
    parser.add_argument('--self-test', action='store_true',
                        help='Small exact-swap and elite checks (also checks GPU if CUDA is available)')
    parser.add_argument('--summarize-only', action='store_true')
    return parser.parse_args()


def parse_cases(tokens):
    allowed = {
        'Kings': {50, 60, 70, 80, 90},
        'SanFrancisco': {50, 70},
        'Bronx': {50, 70},
    }
    selected = []
    for token in tokens:
        if ':' not in token:
            raise ValueError(f'Expected CITY:P, received {token!r}')
        name, num = token.rsplit(':', 1)
        city = resolve_city(name)
        p = int(num)
        if city.key not in allowed or p not in allowed[city.key]:
            raise ValueError(f'{city.key}:p={p} is not among the nine paper configurations')
        if (city.key, p) not in selected:
            selected.append((city.key, p))
    if not selected:
        raise ValueError('At least one case is required')
    return selected


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_config(args):
    """Reject mixing runs produced by incompatible settings or source revisions."""
    base = Path(__file__).resolve().parent
    files = ['benchmark.py'] + sorted(str(p.relative_to(base)) for p in (base / 'mclp').glob('*.py'))
    return dict(
        data_sha256=sha256(args.data),
        source_sha256={name: sha256(base / name) for name in files},
        seconds=args.seconds, starts=args.starts,
        kick_size=args.kick_size, max_kicks=args.max_kicks,
        check_every=args.check_every, max_rounds=args.max_rounds,
        seed_start=args.seed_start,
        restart='elite', archive_size=12, min_distance=10,
        elite_fraction=0.75, recombine_fraction=0.25,
        perturb_sizes=[6, 10, 14, 20], recombine_rcl=4,
    )


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.writing-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def at_time(history, t):
    available = [int(v) for sec, v in history if float(sec) <= t]
    return max(available) if available else None


def make_row(city, p, method, seed, bks, result, inst, budget):
    objective = int(result['objective'])
    sites = result['facilities']
    if len(sites) != p or len(set(sites)) != p or objective != inst.value(sites):
        raise AssertionError(f'Invalid returned solution for {city.key}, p={p}, {method}')
    history = result['history']
    if not history or history[-1][1] != objective:
        raise AssertionError('Anytime history and final incumbent disagree')
    if any(b[0] < a[0] or b[1] < a[1] for a, b in zip(history, history[1:])):
        raise AssertionError('Anytime history is not monotone')
    elapsed = float(result['time_sec'])
    row = dict(city=city.key, n=city.n, p=p, radius_m=city.radius_m,
               method=method, seed=seed, bks=bks, objective=objective,
               gap_to_bks_pct=100.0 * (bks-objective) / bks,
               bks_hit=int(objective >= bks),
               time_to_best_sec=float(result['time_to_best_checkpoint_sec']),
               search_sec=elapsed, setup_sec=float(result['setup_sec']),
               status=result['status'])
    for checkpoint in (30, 60, 120, 300):
        row[f'objective_at_{checkpoint}s'] = at_time(history, checkpoint) if elapsed >= checkpoint else None
    return row


def summarize(output):
    rows = []
    for p in sorted((output / 'runs').glob('*.json')):
        record = json.loads(p.read_text(encoding='utf-8'))
        rows.append(record['row'])
    if not rows:
        print('No completed benchmark runs to summarize.')
        return
    rows.sort(key=lambda r: (r['city'], int(r['p']), r['method'], int(r['seed'])))
    with (output / 'runs.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for r in rows:
        groups.setdefault((r['city'], int(r['p']), r['method']), []).append(r)
    summary = []
    for (city, p, method), group in sorted(groups.items()):
        gaps = [float(r['gap_to_bks_pct']) for r in group]
        times = [float(r['time_to_best_sec']) for r in group]
        summary.append(dict(city=city, p=p, method=method, runs=len(group),
            bks=group[0]['bks'], mean_gap_pct=statistics.mean(gaps),
            std_gap_pct=statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
            bks_hits=sum(int(r['bks_hit']) for r in group),
            mean_time_to_best_sec=statistics.mean(times)))
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'Saved {len(rows)} runs to {output / "runs.csv"}; {len(summary)} groups to summary.csv')


def run_gpu(inst, p, seed, args):
    """Run the paper's GPU-resident search with elite-guided restarts."""
    from mclp.elite import EliteWeightedResidentILS
    engine = EliteWeightedResidentILS(
        inst, p, args.starts, seed, args.kick_size, args.max_kicks,
        restart_policy='elite', elite_size=12, min_distance=10,
        elite_fraction=.75, recombine_fraction=.25,
        perturb_sizes=(6, 10, 14, 20), rcl=4,
    )
    return engine.run(args.seconds, args.check_every, 0, args.max_rounds,
                      reference_bks=None, stop_at_bks=False, log_every_pools=0)


def self_test():
    from scipy.sparse import csr_matrix
    from mclp.gpu_search import WeightedInstance
    from mclp.elite import DiverseEliteArchive, EliteRestartSampler
    rng = np.random.default_rng(997)
    n, p = 14, 4
    A = rng.random((n, n)) < .30
    np.fill_diagonal(A, True)
    weights = rng.integers(2, 40, size=n, dtype=np.int32)
    packed = np.packbits(A, axis=1, bitorder='little')
    packed = np.pad(packed, ((0, 0), (0, (-n) % 32 // 8)))
    words = np.ascontiguousarray(packed).view('<u4').reshape(n, -1)
    inst = WeightedInstance(np.zeros((n, 2)), weights,
                            csr_matrix(A, dtype=np.int32),
                            np.ascontiguousarray(words.T))
    sites = np.asarray(inst.greedy(p), dtype=np.int32)
    # Small independent mathematical check: compare the paper's exact
    # G_s + H_rs - L_r formula with directly recomputed objective differences.
    # This is validation only, not a separate CPU search implementation.
    counts = np.asarray(A[sites].sum(axis=0), dtype=np.int32)
    value = int(weights[counts > 0].sum(dtype=np.int64))
    assert value == inst.value(sites)
    for r in range(p):
        drop = int(sites[r])
        loss = int(weights[A[drop] & (counts == 1)].sum(dtype=np.int64))
        for add in range(n):
            if add in sites:
                continue
            gain = int(weights[A[add] & (counts == 0)].sum(dtype=np.int64))
            overlap = int(weights[A[add] & A[drop] & (counts == 1)].sum(dtype=np.int64))
            formula = gain + overlap - loss
            changed = [add if j == r else int(v) for j, v in enumerate(sites)]
            direct = inst.value(changed) - value
            assert formula == direct, (drop, add, formula, direct)
    archive = DiverseEliteArchive(p, capacity=4, min_distance=2)
    first = (0, 1, 2, 3)
    assert archive.offer(first, 100)
    assert not archive.offer((0, 1, 2, 4), 99)
    assert archive.offer((0, 1, 2, 4), 110)
    sampler = EliteRestartSampler(inst, p, 2027, tuple(sites), .75, .25,
                                  (1, 2, 3), 3)
    seeds = sampler.generate(archive, 1, starts=8)
    assert all(len(x) == p and len(set(x)) == p for x in seeds)
    print('PASS: exact swap-gain formula and elite-restart checks')
    from numba import cuda
    if cuda.is_available():
        from mclp.elite import EliteWeightedResidentILS
        engine = EliteWeightedResidentILS(inst, p, 2, 2027, 2, 2, min_distance=1, perturb_sizes=(1, 2, 3))
        engine.one_round()
        cuda.synchronize()
        selected = engine.selected.copy_to_host()
        current_counts = engine.counts.copy_to_host()
        current_values = engine.values.copy_to_host()
        for row, c, v in zip(selected, current_counts, current_values):
            assert len(set(row.tolist())) == p
            assert np.array_equal(c, A[row].sum(axis=0))
            assert v == inst.value(row)
        print('PASS: GPU best-swap / coverage-count invariant')
    else:
        print('GPU test skipped: CUDA device unavailable (set NUMBA_ENABLE_CUDASIM=1 to simulate)')


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.seconds <= 0 or args.reps < 1 or args.seed_start < 0 or args.starts < 1:
        raise ValueError('Invalid search budget or repetitions')
    if not (1 <= args.kick_size <= min(p for _, p in parse_cases(args.cases))):
        raise ValueError('Invalid kick-size')
    if args.max_kicks < 0 or args.check_every < 1 or args.max_rounds < 1:
        raise ValueError('Invalid CUDA search settings')
    cases = parse_cases(args.cases)
    plan = [(city, p, args.seed_start + i)
            for city, p in cases for i in range(args.reps)]
    if args.dry_run:
        print(f'{len(cases)} configurations, GPU only, '
              f'{args.reps} seeds: {len(plan)} runs; {args.seconds:g}s each')
        print('Cases:', ', '.join(f'{city}:p{p}' for city, p in cases))
        return
    if args.summarize_only:
        summarize(args.output)
        return
    if not args.data.is_file():
        raise FileNotFoundError(f'Benchmark ZIP not found: {args.data}')
    config = experiment_config(args)
    manifest = args.output / 'manifest.json'
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding='utf-8')) != config:
            raise RuntimeError('Existing result directory has different experiment settings. '
                               'Choose another --output directory.')
    else:
        write_json(manifest, config)
    loaded_city = None
    inst = None
    for idx, (name, p, seed) in enumerate(plan, start=1):
        method = 'gpu'  # Retain the paper-compatible method label in CSV/JSON.
        result_file = args.output / 'runs' / f'{name}_p{p}_{method}_s{seed}.json'
        if result_file.is_file() and not args.verify_only:
            print(f'[{idx}/{len(plan)}] already done: {result_file.stem}')
            continue
        city = resolve_city(name)
        if loaded_city != name:
            t0 = time.perf_counter()
            inst, cached = build_instance(args.data, city, args.output / 'cache')
            print(f'{name}: {city.n} sites, R={city.radius_m} m, '
                  f'cache={cached}, preparation={time.perf_counter()-t0:.1f}s', flush=True)
            loaded_city = name
        assert inst is not None
        bks = check_reference(inst, args.data, city, p)['reference_obj']
        if args.verify_only:
            print(f'VERIFIED {city.key} p={p} BKS={bks:,}')
            continue
        print(f'[{idx}/{len(plan)}] {name} p={p} {method} seed={seed}', flush=True)
        result = run_gpu(inst, p, seed, args)
        row = make_row(city, p, method, seed, bks, result, inst, args.seconds)
        write_json(result_file, dict(row=row, result=result))
        summarize(args.output)
        print(f'  objective={row["objective"]:,}, gap={row["gap_to_bks_pct"]:.4f}%, '
              f't_best={row["time_to_best_sec"]:.2f}s', flush=True)
    if not args.verify_only:
        summarize(args.output)


if __name__ == '__main__':
    main()
