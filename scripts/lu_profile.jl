# Profiling the DataFlowTasks LU: summary, DAG (4x4 tiles) and an interactive
# trace (16x16 tiles).
# Run with: julia --project -t 8 scripts/lu_profile.jl
# In the trace window: hover a task to see its predecessors (blue), successors
# (orange) and lifecycle; scroll or drag to zoom, right-drag to pan, Ctrl+click
# to reset. Close the window to exit.
# Also writes figures/lu-dag.svg and figures/lu-trace.png. The slides use copies
# in talk/figures/: `rsvg-convert -f pdf lu-dag.svg -o lu-dag.pdf`, and the png
# as is (GLMakie saves it at 96 dpi: the slide's `trim` assumes this).

include("lu_dft.jl")
import DataFlowTasks as DFT
using GLMakie, GraphViz
outdir = mkpath(normpath(joinpath(@__DIR__, "..", "figures")))
categories = ["lu", "ldiv", "rdiv", "mul"]

# small problem: a DAG you can read
A = rand(1000, 1000) + 1000I
lu_dft!(tiles(copy(A), 250)) # compile
T = tiles(copy(A), 250) # outside @log: the copy is not part of the trace
log_small = DFT.@log lu_dft!(T)
DFT.savedag(joinpath(outdir, "lu-dag.svg"), GraphViz.Graph(log_small))

# the real problem: summary and trace
A = rand(4000, 4000) + 4000I
lu_dft!(tiles(copy(A), 250)) # warm up
T = tiles(copy(A), 250)
log_info = DFT.@log lu_dft!(T)
DFT.describe(log_info; categories)
fig = plot(log_info; categories)
save(joinpath(outdir, "lu-trace.png"), fig)
println("figures written to ", outdir)

wait(display(fig)) # interactive window, until closed
