# The tempting (and wrong) way to give each thread a buffer.
# Run with: julia --project -t 4 scripts/threadid.jl

using LinearAlgebra
using Base.Threads: @threads, maxthreadid, nthreads, threadid

BLAS.set_num_threads(1)

println("threads: ", nthreads(), " (+", nthreads(:interactive), " interactive)")

N, m = 100, 1000
As = [rand(m, m) + m * I for _ in 1:N]
bs = [rand(m) for _ in 1:N]
xs = similar(bs)

# one buffer per thread id (ids can exceed nthreads(): interactive threads)
Fs = [lu(As[1]) for _ in 1:maxthreadid()]
@threads for i in 1:N
    F = Fs[threadid()]
    lu!(F, As[i])
    xs[i] = F \ bs[i]
end

# looks fine...
println("max error: ", maximum(i -> norm(As[i] * xs[i] - bs[i]), 1:N))
# ...but only because nothing in the loop body yields: a task can migrate, or
# share its thread with another task, at any yield point.
