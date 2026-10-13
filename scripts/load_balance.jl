# Load balancing: the number of chunks matters when iterations have unequal cost.
# Run with: julia --project -t 4 scripts/load_balance.jl

using LinearAlgebra
using Base.Threads: @spawn, nthreads
using Base.Iterators: partition

BLAS.set_num_threads(1)

println("threads: ", nthreads())

N = 100
ms = round.(Int, range(200, 1600, N)) # growing sizes: cost of `lu` ∝ m³
As = [rand(m, m) + m * I for m in ms]
bs = [rand(m) for m in ms]

solve_seq(As, bs) = map((A, b) -> lu(A) \ b, As, bs)

function solve_chunks(As, bs, nchunks)
    xs = similar(bs)
    n = cld(length(As), nchunks) # chunk size
    ts = map(partition(eachindex(As), n)) do idxs
        @spawn for i in idxs
            xs[i] = lu(As[i]) \ bs[i]
        end
    end
    waitall(ts)
    return xs
end

solve_seq(As[1:2], bs[1:2]); solve_chunks(As[1:4], bs[1:4], 2) # compile

print("sequential: "); @time solve_seq(As, bs)
for nchunks in (4, 20, 100)
    print("$nchunks chunks: "); @time solve_chunks(As, bs, nchunks)
end
