# Buffers with threads: one buffer per task (chunk), or a pool of buffers shared by many tasks.
# Run with: julia --project -t 4 scripts/buffers_threads.jl
#      and: julia --project -t 8 scripts/buffers_threads.jl

using LinearAlgebra
using Base.Threads: @spawn, @threads, nthreads
using Base.Iterators: partition

BLAS.set_num_threads(1)

println("threads: ", nthreads())

N, m = 100, 1000
As = [rand(m, m) + m * I for _ in 1:N]
bs = [rand(m) for _ in 1:N]

function solve_seq(As, bs)
    return map((A, b) -> lu(A) \ b, As, bs)
end

# @threads, allocating a factorization per iteration
function solve_alloc(As, bs)
    xs = similar(bs)
    @threads for i in eachindex(As)
        xs[i] = lu(As[i]) \ bs[i]
    end
    return xs
end

# one task per chunk, one buffer per task
function solve_chunks(As, bs)
    xs = similar(bs)
    n = cld(length(As), nthreads()) # chunk size
    @sync for idxs in partition(eachindex(As), n)
        @spawn begin
            F = lu(As[1]) # this task's buffer
            for i in idxs
                lu!(F, As[i])
                xs[i] = F \ bs[i]
            end
        end
    end # @sync: wait for all tasks spawned inside
    return xs
end

# @threads, one buffer per task: created on first use by each task (Julia >= 1.12)
function solve_once(As, bs)
    xs = similar(bs)
    F0 = lu(As[1])
    buf = OncePerTask(() -> copy(F0))
    @threads for i in eachindex(As)
        F = buf() # this task's buffer
        lu!(F, As[i])
        xs[i] = F \ bs[i]
    end
    return xs
end

# @threads hides its tasks: borrow a buffer from a pool (a Channel) instead
function solve_pool(As, bs)
    xs = similar(bs)
    # factor once (serial!), then copy: copies are cheap.
    # NOTE: not called `F`! The loop body below would then assign to this captured
    # variable, shared by all tasks: a data race.
    F0 = lu(As[1])
    pool = Channel{typeof(F0)}(nthreads())
    for _ in 1:nthreads()
        put!(pool, copy(F0))
    end
    @threads for i in eachindex(As)
        F = take!(pool) # wait if empty
        lu!(F, As[i])
        xs[i] = F \ bs[i]
        put!(pool, F)   # give it back
    end
    return xs
end

for f in (solve_seq, solve_alloc, solve_chunks, solve_once, solve_pool) # compile
    f(As[1:4], bs[1:4])
end

for f in (solve_seq, solve_alloc, solve_chunks, solve_once, solve_pool) # check
    xs = f(As, bs)
    println(rpad(f, 13), "max residual: ", maximum(i -> norm(As[i] * xs[i] - bs[i]), eachindex(As)))
end

print(rpad("solve_seq:", 14)); @time solve_seq(As, bs)
print(rpad("solve_alloc:", 14)); @time solve_alloc(As, bs)
print(rpad("solve_chunks:", 14)); @time solve_chunks(As, bs)
print(rpad("solve_once:", 14)); @time solve_once(As, bs)
print(rpad("solve_pool:", 14)); @time solve_pool(As, bs)
