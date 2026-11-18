from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.sparse import csr_matrix

from . import gpu_search as core

P_VALUES = (50, 60, 70, 80, 90, 100)

@dataclass(frozen=True)
class City:
    key: str
    data_file: str
    n: int
    radius_m: int
    paper_name: str

CITIES = {
    "Kings": City("Kings", "Kings", 7730, 800, "Kings"),
    "SanFrancisco": City("SanFrancisco", "SanFrancisco", 5137, 600, "San Francisco"),
    "Bronx": City("Bronx", "Bronx", 3839, 600, "Bronx"),
    "Manhattan": City("Manhattan", "NewYork", 2713, 400, "Manhattan"),
}
_ALIASES = {"kings": "Kings", "sanfrancisco": "SanFrancisco",
            "san-francisco": "SanFrancisco", "san_francisco": "SanFrancisco",
            "sf": "SanFrancisco", "bronx": "Bronx", "newyork": "Manhattan",
            "new-york": "Manhattan", "manhattan": "Manhattan"}

# Published AILS-PR Table 4: BKS, mean final objective, best of 30 runs,
# mean time (seconds) to final solution (not time-to-BKS) with 300 s/run.
# These are immutable literature baselines, NOT inputs to the optimizer.
_PAPER = {
    "Kings": [
        (2022077, 2022077.00, 2022077, 99.5),
        (2227025, 2226850.97, 2227025, 163.2),
        (2380660, 2380502.47, 2380660, 173.0),
        (2467701, 2467311.57, 2467701, 166.1),
        (2502669, 2501635.90, 2502222, 230.6),
        (2504700, 2504692.47, 2504700, 202.7)],
    "SanFrancisco": [
        (617592, 617592.00, 617592, 11.5),
        (683139, 683139.00, 683139, 72.4),
        (733772, 733769.00, 733772, 145.8),
        (769781, 769757.77, 769781, 154.2),
        (791562, 791276.37, 791562, 175.6),
        (802307, 801845.93, 802211, 260.0)],
    "Bronx": [
        (1205051, 1205051.00, 1205051, 48.7),
        (1290635, 1290616.80, 1290635, 102.5),
        (1348232, 1348231.73, 1348232, 77.6),
        (1376061, 1375669.07, 1376061, 196.1),
        (1384376, 1384268.43, 1384364, 228.8),
        (1385099, 1385075.73, 1385091, 252.1)],
    "Manhattan": [
        (1153640, 1153640.00, 1153640, 2.4),
        (1288174, 1288174.00, 1288174, 22.1),
        (1396333, 1396333.00, 1396333, 12.0),
        (1481434, 1481434.00, 1481434, 49.4),
        (1539992, 1539989.97, 1539992, 81.8),
        (1574256, 1574142.10, 1574229, 219.8)],
}
PAPER_SOURCE = ("Maximo, Cordeau and Nascimento, A Hybrid Adaptive Iterated "
                "Local Search Heuristic for the Maximal Covering Location "
                "Problem, International Transactions in Operational Research, "
                "Table 4 (30 runs, 300 s/run)")


def resolve_city(name: str) -> City:
    key = _ALIASES.get(name.strip().lower())
    if key is None:
        raise ValueError(f"Unknown city {name!r}; choose from {', '.join(CITIES)}")
    return CITIES[key]


def published_row(city: City, p: int) -> dict:
    if p not in P_VALUES:
        raise ValueError(f"Unsupported p={p}; valid values are {P_VALUES}")
    bks, avg, best, tt = _PAPER[city.key][P_VALUES.index(p)]
    return dict(bks=int(bks), ailspr_avg=float(avg),
                ailspr_best=int(best), ailspr_time_sec=float(tt),
                source=PAPER_SOURCE, runs=30, time_limit_sec=300,
                time_definition="mean time to final solution (not time-to-BKS)")


def read_city_text(repo_zip: Path, city: City) -> str:
    return core.load_repo_file(repo_zip, f"Instancias/{city.data_file}.txt")


def parse_city_text(raw: str, expected_n: int):
    # Mirrors the earlier Kings parser and keeps the original city coordinates.
    lines = [line.strip() for line in raw.splitlines()]
    dim = next(int(line.split(':', 1)[1]) for line in lines
               if line.startswith('DIMENSION'))
    kind = next(line.split(':', 1)[1].strip() for line in lines
                if line.startswith('EDGE_WEIGHT_TYPE'))
    if dim != expected_n or kind != 'COORD':
        raise ValueError(f"Unexpected DIMENSION / EDGE_WEIGHT_TYPE: {dim}, {kind}")
    c, d = lines.index('NODE_COORD_SECTION'), lines.index('DEMAND_SECTION')
    coords = np.asarray([[float(x) for x in line.split()]
                         for line in lines[c + 1:c + 1 + dim]], dtype=np.float64)
    weights = np.asarray([int(line) for line in lines[d + 1:d + 1 + dim]],
                         dtype=np.int32)
    if (coords.shape != (dim, 2) or weights.shape != (dim,) or
            not np.isfinite(coords).all() or (weights < 0).any()):
        raise ValueError("Invalid geographic coordinates / demand weights")
    return coords, weights


def read_reference(repo_zip: Path, city: City, p: int) -> tuple[int, tuple[int, ...]]:
    """Read a saved paper solution for validation only; never seed the solver."""
    raw = core.load_repo_file(
        repo_zip, f"OtimosFinais/{city.data_file}_{p}_{city.radius_m}.txt")
    objective = re.search(r"(?m)^\s*f\s*:\s*(\d+)", raw)
    chosen = re.search(r"(?m)^\s*Facilidade\s*:\s*([\d\s]+)", raw)
    if not objective or not chosen:
        raise ValueError(f"Malformed reference for {city.key}, p={p}")
    sites = tuple(sorted(int(s) for s in chosen.group(1).split()))
    if len(sites) != p or len(set(sites)) != p or any(
            s < 0 or s >= city.n for s in sites):
        raise ValueError(f"Infeasible repository reference for {city.key}, p={p}")
    return int(objective.group(1)), sites


def check_reference(inst: core.WeightedInstance, repo_zip: Path,
                    city: City, p: int) -> dict:
    expected, sites = read_reference(repo_zip, city, p)
    calculated = inst.value(sites)
    published = published_row(city, p)
    if expected != calculated or expected != published['bks']:
        raise AssertionError(
            f"{city.key} p={p}: saved objective={expected}, "
            f"recomputed={calculated}, paper BKS={published['bks']}. "
            "This likely indicates a distance/weight parsing mismatch.")
    return dict(reference_obj=expected, recomputed_obj=calculated,
                matched=True, reference_facilities_used_for_search=False,
                reference_source="AILS-MCLP repository OtimosFinais")


def build_instance(repo_zip: Path, city: City, cache_dir: Path | None = None):
    """Build or load a cache keyed by the actual repo data + coverage radius.

    The core coverage builder (including Java-compatible float32 distances)
    is unchanged. Cache is safe across seeds and values of p.
    """
    raw = read_city_text(repo_zip, city)
    fingerprint = sha256((raw + f"|radius={city.radius_m}|v1").encode()).hexdigest()[:20]
    cache_path = (cache_dir / f"{city.key}_r{city.radius_m}_{fingerprint}.npz"
                  if cache_dir is not None else None)
    if cache_path is not None and cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                coords = data['coords']
                weights = data['weights']
                words = data['words_t']
                indices = data['indices']
                indptr = data['indptr']
                nn = int(data['n'])
            if (nn != city.n or len(weights) != city.n or
                    words.shape != ((city.n + 31) // 32, city.n)):
                raise ValueError("Cache shape mismatch")
            adj = csr_matrix((np.ones(len(indices), dtype=np.int32),
                              indices, indptr), shape=(nn, nn))
            return core.WeightedInstance(coords, weights, adj, words), True
        except (OSError, ValueError, KeyError, EOFError):
            print(f"Invalid cache {cache_path}; rebuilding", flush=True)
    coords, weights = parse_city_text(raw, city.n)
    inst = core.build_adjacency(coords, weights, float(city.radius_m))
    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Temporary file and atomic replacement keep partially written caches
        # from being considered valid in the next run.
        import os
        import tempfile
        fd, tmp_name = tempfile.mkstemp(prefix='mclp_cache_', suffix='.npz',
                                        dir=cache_dir)
        os.close(fd)
        try:
            np.savez_compressed(tmp_name, coords=coords, weights=weights,
                                words_t=inst.words_t,
                                indices=inst.adj.indices,
                                indptr=inst.adj.indptr,
                                n=np.int64(inst.n))
            os.replace(tmp_name, cache_path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
    return inst, False


def list_benchmarks() -> str:
    rows = [f"{'City':<15} {'n':>6} {'Radius (m)':>11}  p values"]
    for c in CITIES.values():
        rows.append(f"{c.key:<15} {c.n:>6} {c.radius_m:>11}  " +
                    ", ".join(map(str, P_VALUES)))
    return '\n'.join(rows)
