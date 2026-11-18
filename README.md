# GPU4CO

Replication code for *Covering Location Optimization at GPU Speed* by Yun-Tian Zhang. The paper is available on arXiv: [arXiv:2609.28084](https://arxiv.org/abs/2609.28084).

## To run the algorithm

Run from the project directory. To run a single benchmark case:

```bash
python benchmark.py --cases Kings:50 --reps 1 --seconds 300
```

All nine cases, GPU search only, ten independent seeds per case, each with 300 seconds of search time:

```bash
python benchmark.py
```

The default case list is Kings (p=50,60,70,80,90), SanFrancisco (p=50,70), and Bronx (p=50,70). Default seeds are 2027–2036. GPU settings: 32 resident trajectories, 256 add-site candidates per tile, 128 threads per CUDA block, kick size 4, max 20 kicks, and one status check per 5 rounds. Elite archive capacity is 12 with minimum exchange distance 10; restart seeds use 50% mutation, 25% recombination, and 25% independent starts.

For a shorter GPU smoke test:

```bash
python benchmark.py --cases Bronx:50 --reps 1 --seconds 10
```

Use `python benchmark.py --help` for all options.

The benchmark produces `results/runs.csv` and `results/summary.csv` plus run JSON records. A matching configuration can be resumed without rerunning completed seeds. Data construction/cache and GPU compilation/setup are excluded from the timed search. The reported time-to-best is the first time a run reaches *its own* final incumbent, not necessarily a published BKS.

## Data attribution

Urban benchmark source: Maximo, V. R., Cordeau, J. F., & Nascimento, M. C. (2025). A hybrid adaptive iterated local search heuristic for the maximal covering location problem. International Transactions in Operational Research, 32(1), 176-193. Best-known solutions are feasible reference objectives, not necessarily proven global optima.
