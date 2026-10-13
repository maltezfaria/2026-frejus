# Granularity: one task per item vs one task per chunk, for cheap items.
# Run with: julia --project -t 4 scripts/granularity.jl

using Base.Threads: @spawn, nthreads
using Base.Iterators: partition

println("threads: ", nthreads())

N = 10^6
xs = rand(N)
ys = similar(xs)

f(x) = sin(x)^2 + cos(x)^2 # cheap: a few ns

function per_item!(ys, xs)
    @sync for i in eachindex(xs)
        @spawn ys[i] = f(xs[i])
    end
end

function per_chunk!(ys, xs, nchunks)
    n = cld(length(xs), nchunks) # chunk size
    @sync for idxs in partition(eachindex(xs), n)
        @spawn for i in idxs
            ys[i] = f(xs[i])
        end
    end
end

map!(f, ys, xs); per_item!(ys, xs); per_chunk!(ys, xs, nthreads()) # compile

print(rpad("sequential:", 14)); @time map!(f, ys, xs)
print(rpad("task per item:", 14)); @time per_item!(ys, xs)
for nchunks in (nthreads(), 16 * nthreads(), 1000)
    print(rpad("$nchunks chunks:", 14))
    @time per_chunk!(ys, xs, nchunks)
end
