# profiles/_pipeline_test

- points with perf capture: 0
- points with PMU sweep:    0
- hotspot rows:             0

## topdown

_(no PMU data yet)_

## IPC

_(no PMU data yet)_

## files

| file | content |
|---|---|
| `hotspots.csv` | perf self-overhead hotspots, tagged by point |
| `topdown.csv` | 920B topdown buckets + sub-metrics per point |
| `ipc.csv` | cycles/instructions/IPC + confidence per point |
| `profile_index.json` | per-point profiler artifacts + phase shares |

Raw `perf.data` stays on a3-22 only (see plan/COORDINATION.md §5).
