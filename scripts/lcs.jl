# Longest common subsequence (LCS): a tiled wavefront with DataFlowTasks.
# Run with: julia --project -t 4 scripts/lcs.jl

using DataFlowTasks

println("threads: ", Threads.nthreads())

# L[i+1, j+1] = length of the LCS of x[1:i] and y[1:j]
function fill_lengths!(L, x, y, irange, jrange)
    @inbounds for j in jrange, i in irange
        L[i+1, j+1] = x[i] == y[j] ? L[i, j] + 1 : max(L[i+1, j], L[i, j+1])
    end
end

# walk back from the bottom-right corner to recover one LCS
function backtrack(L, x, y)
    i, j = lastindex(x), lastindex(y)
    subseq = Char[]
    while L[i+1, j+1] != 0
        if x[i] == y[j]
            pushfirst!(subseq, x[i])
            i, j = i - 1, j - 1
        elseif L[i+1, j] > L[i, j+1]
            j -= 1
        else
            i -= 1
        end
    end
    return String(subseq)
end

function lcs_seq!(L, x, y)
    L[:, 1] .= 0
    L[1, :] .= 0
    fill_lengths!(L, x, y, eachindex(x), eachindex(y))
    return backtrack(L, x, y)
end

# split a range into n chunks
tiles(r, n) = Iterators.partition(r, cld(length(r), n))

function lcs_tiled!(L, x, y, n)
    L[:, 1] .= 0
    L[1, :] .= 0
    for jr in tiles(eachindex(y), n), ir in tiles(eachindex(x), n)
        fill_lengths!(L, x, y, ir, jr)
    end
    return backtrack(L, x, y)
end

# tasks with dependencies by hand: tile (i,j) waits for (i-1,j) and (i,j-1)
using Base.Threads: @spawn
function lcs_tasks!(L, x, y, n)
    L[:, 1] .= 0
    L[1, :] .= 0
    I, J = collect(tiles(eachindex(x), n)), collect(tiles(eachindex(y), n))
    T = Matrix{Task}(undef, length(I), length(J))
    for j in eachindex(J), i in eachindex(I)
        T[i, j] = @spawn begin
            i > 1 && wait(T[i-1, j])
            j > 1 && wait(T[i, j-1])
            fill_lengths!(L, x, y, I[i], J[j])
        end
    end
    wait(T[end, end])
    return backtrack(L, x, y)
end

# same loops, each tile is a task: dependencies come from the annotations
function lcs_dft!(L, x, y, n)
    @dspawn fill!(@W(view(L, :, 1)), 0) label = "init"
    @dspawn fill!(@W(view(L, 1, :)), 0) label = "init"
    for (j, jr) in enumerate(tiles(eachindex(y), n)), (i, ir) in enumerate(tiles(eachindex(x), n))
        @dspawn begin
            @R view(L, ir, jr)               # reads the tiles above and to the left
            @RW view(L, ir .+ 1, jr .+ 1)    # writes its own tile
            fill_lengths!(L, x, y, ir, jr)
        end label = "tile ($i,$j)"
    end
    t = @dspawn backtrack(@R(L), x, y) label = "backtrack"
    return fetch(t)
end

x = rand("ATCG", 8192)
y = rand("ATCG", 16384)
L = Matrix{Int32}(undef, length(x) + 1, length(y) + 1) # 0.5 GB
n = 16 # tiles per dimension

ref = lcs_seq!(L, x, y)
@assert lcs_tiled!(L, x, y, n) == ref
@assert lcs_tasks!(L, x, y, n) == ref
@assert lcs_dft!(L, x, y, n) == ref
println("LCS length: ", length(ref))

for (name, f) in (("sequential", () -> lcs_seq!(L, x, y)),
                  ("tiled", () -> lcs_tiled!(L, x, y, n)),
                  ("tasks by hand", () -> lcs_tasks!(L, x, y, n)),
                  ("DataFlowTasks", () -> lcs_dft!(L, x, y, n)))
    t = minimum(@elapsed(f()) for _ in 1:5)
    println(rpad(name, 15), round(t, digits = 3), " s")
end
