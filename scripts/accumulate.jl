# Accumulating in a threaded loop: sum of log-determinants.
# Run with: julia --project -t 4 scripts/accumulate.jl

using LinearAlgebra
using Base.Threads: @spawn, @threads, Atomic, atomic_add!, nthreads
using Base.Iterators: partition

BLAS.set_num_threads(1)

println("threads: ", nthreads())

N, m = 100, 1000
As = [rand(m, m) + m * I for _ in 1:N]
bs = [rand(m) for _ in 1:N]

function logdet_seq(As, bs)
    xs = similar(bs)
    ld = 0.0
    for i in eachindex(As)
        F = lu(As[i])
        xs[i] = F \ bs[i]
        ld += logdet(F)
    end
    return ld
end

# WRONG: all tasks update the same variable (`ld` is captured by the loop body:
# one variable, "boxed" and shared by all the tasks)
function logdet_naive(As, bs)
    xs = similar(bs)
    ld = 0.0
    @threads for i in eachindex(As)
        F = lu(As[i])
        xs[i] = F \ bs[i]
        ld += logdet(F) # read, add, write: race!
    end
    return ld
end

# fix 1: a lock around the update (Lockable = value + lock; a Ref, since a Float64 cannot be mutated)
function logdet_lock(As, bs)
    xs = similar(bs)
    ld = Lockable(Ref(0.0))
    @threads for i in eachindex(As)
        F = lu(As[i])
        xs[i] = F \ bs[i]
        d = logdet(F)          # compute outside the lock
        @lock ld ld[][] += d   # one task at a time
    end
    return @lock ld ld[][]
end

# fix 2: atomic update, read-add-write as one indivisible step
function logdet_atomic(As, bs)
    xs = similar(bs)
    ld = Atomic{Float64}(0.0)
    @threads for i in eachindex(As)
        F = lu(As[i])
        xs[i] = F \ bs[i]
        atomic_add!(ld, logdet(F))
    end
    return ld[]
end

# fix 3: no shared state, each task returns its partial sum (chunk pattern)
function logdet_partial(As, bs)
    xs = similar(bs)
    n = cld(length(As), nthreads()) # chunk size
    ts = map(partition(eachindex(As), n)) do idxs
        @spawn begin
            s = 0.0
            for i in idxs
                F = lu(As[i])
                xs[i] = F \ bs[i]
                s += logdet(F)
            end
            s # returned by the task
        end
    end
    return sum(fetch.(ts))
end

ref = logdet_seq(As, bs)
println("sequential: ", ref)
for run in 1:5
    ld = logdet_naive(As, bs)
    println("naive, run $run: ", ld, "   relative error = ", abs(ld - ref) / abs(ref))
end

for f in (logdet_lock, logdet_atomic, logdet_partial)
    f(As[1:4], bs[1:4]) # compile
    lds = [f(As, bs) for _ in 1:5]
    println(f, ": relative errors = ", [abs(ld - ref) / abs(ref) for ld in lds])
end

print(rpad("logdet_seq:", 16)); @time logdet_seq(As, bs)
print(rpad("logdet_lock:", 16)); @time logdet_lock(As, bs)
print(rpad("logdet_atomic:", 16)); @time logdet_atomic(As, bs)
print(rpad("logdet_partial:", 16)); @time logdet_partial(As, bs)
