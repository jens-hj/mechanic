#!/bin/zsh
root=docs/performance-results/2026-09-14-cpu-scale/ccd-followup
binary_root=.physics-reference/ccd-followup
python3 scripts/run-cpu-scale-benchmark.py --baseline "$binary_root/baseline" --candidate "$binary_root/candidate" --output "$root/cold" --assemblies connected --copies 1 2 5 10 --repeats 3 --warmup 0 --ticks 120
python3 scripts/run-cpu-scale-benchmark.py --baseline "$binary_root/baseline" --candidate "$binary_root/candidate" --output "$root/connected" --assemblies connected --copies 1 2 5 10 --repeats 3 --warmup 600 --ticks 3600
python3 scripts/run-cpu-scale-benchmark.py --baseline "$binary_root/baseline" --candidate "$binary_root/candidate" --output "$root/separated" --assemblies separated --copies 10 --repeats 2 --warmup 600 --ticks 3600
python3 /tmp/mechanic-ccd/analyze.py > /tmp/mechanic-ccd/analysis.log
