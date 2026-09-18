# Documentation

Start with the [repository README](../README.md) for setup and commands, and
[`AGENTS.md`](../AGENTS.md) for the crate map and the conventions every change
follows.

## Design

- [Architecture invariants](architecture.md): crate ownership, persistence, clocks, kernel order, and the GPU ABI.
- [Compiled machine dynamics](compiled-machine-dynamics.md): how a construction graph becomes a reduced-coordinate machine.
- [CPU physics](physics-cpu.md): the soft-step solver the app runs and the exact reference solver.
- [Gameplay design](gameplay-design.md): what the game is for and what that rules in and out.

## Features

- [Independent suspension](suspension.md)
- [Linear bearings](linear-bearings.md)
- [Feature weld placement](feature-weld-placement.md)
- [Live construction editing and dimension-link freezing](live-construction-editing.md)
- [Multitool guide](multitool/GUIDE.md)
- [Tool effects](tool-effects/README.md)

## Operating the app and tools

- [Environment variables](environment.md): every `MECHANIC_*` switch, by binary.
- [Performance profiling](performance-profiling.md): captures, render experiments, and the scripts under `scripts/`.

## Status and measurements

- [Milestone status](milestone-status.md): what is done against the scale gates.
- [Terrain response status](terrain-response-status.md)
- [CPU construction scaling](cpu-scaling.md)
- [Vehicle performance checklist](performance-todo.md): dated working notes; the entries record history and do not gate current work.
- [Performance results](performance-results/README.md): retained benchmark evidence.
