# CPU-bound work: factor 10 matrices.
# Run with: julia --project -t 1 scripts/parallelism.jl   (no gain)
#      and: julia --project -t 4 scripts/parallelism.jl   (speedup)

using LinearAlgebra
using Base.Threads: @spawn

BLAS.set_num_threads(1) # parallelism comes from our tasks, not from BLAS

println("threads: ", Threads.nthreads(), " (+", Threads.nthreads(:interactive), " interactive)")

As = [rand(2000, 2000) for _ in 1:10]

# sequential
lu_seq(As) = map(lu, As)

# one task per matrix: a computing task never waits, so it only runs faster
# if there are other threads to run on
lu_tasks(As) = fetch.(map(A -> @spawn(lu(A)), As))

lu_seq(As[1:2]); lu_tasks(As[1:2]) # compile

print("sequential: "); @time lu_seq(As)
print("tasks:      "); @time lu_tasks(As)
