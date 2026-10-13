# Independent iterations: one task per iteration, then @threads (one task per chunk).
# Run with: julia --project -t 4 scripts/threads_loop.jl

using LinearAlgebra
using Base.Threads: @spawn, @threads, nthreads
using Base.Iterators: partition

BLAS.set_num_threads(1) # parallelism comes from our tasks, not from BLAS

println("threads: ", nthreads())

As = [rand(2000, 2000) for _ in 1:10]

lu_seq(As) = map(lu, As)

# one task per iteration, then wait for all
function lu_spawn(As)
    Fs = Vector{Any}(undef, length(As))
    ts = Task[]
    for i in eachindex(As)
        t = @spawn Fs[i] = lu(As[i])
        push!(ts, t)
    end
    waitall(ts) # Fs is ready only now
    return Fs
end

# same thing: @sync waits for every @spawn in its scope
function lu_sync(As)
    Fs = Vector{Any}(undef, length(As))
    @sync for i in eachindex(As)
        @spawn Fs[i] = lu(As[i])
    end
    return Fs
end

function lu_threads(As)
    Fs = Vector{Any}(undef, length(As))
    @threads for i in eachindex(As)
        Fs[i] = lu(As[i])
    end
    return Fs
end

# simplified @threads: one task per chunk, then wait for all
function lu_chunks(As)
    Fs = Vector{Any}(undef, length(As))
    chunks = partition(eachindex(As), cld(length(As), nthreads()))
    tasks = map(chunks) do idxs
        @spawn for i in idxs
            Fs[i] = lu(As[i])
        end
    end
    waitall(tasks)
    return Fs
end

lu_seq(As[1:2]); lu_spawn(As[1:2]); lu_sync(As[1:2]); lu_threads(As[1:2]); lu_chunks(As[1:2]) # compile

print("sequential:         "); @time lu_seq(As)
print("task per iteration: "); @time lu_spawn(As)
print("same, with @sync:   "); @time lu_sync(As)
print("@threads:           "); @time lu_threads(As)
print("chunks by hand:     "); @time lu_chunks(As)
