# Shared memory parallelism in Julia — ANF Julia4HPC, Fréjus 2026

Material for the session *Parallélisation à mémoire partagée : Threads.jl et
DataFlowTasks.jl* (Tuesday 13 October 2026, 13:30–17:30).

- `talk/`: Beamer slides (`frejus_2026.tex`, figures in `talk/figures/`)
- `scripts/`: the examples shown in the slides, to run yourself

## Setup

Julia ≥ 1.12. Part 1 only needs the standard library:

```bash
julia --project -t auto scripts/threads_loop.jl
```

Part 2 needs DataFlowTasks.jl (v0.3), GraphViz, GLMakie and BenchmarkTools:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Run the examples from the repository root; each script says how many threads
to use in its first lines, e.g.

```bash
julia --project -t 4 scripts/accumulate.jl
```

`scripts/lu_profile.jl` opens an interactive trace (hover a task, zoom, pan)
and writes its figures to `figures/`. GLMakie needs OpenGL: on a machine
without a display (e.g. over SSH), replace `GLMakie` by `CairoMakie` in
`Project.toml` and in the script, and look at the saved figures instead.
