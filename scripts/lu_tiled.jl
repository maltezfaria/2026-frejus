# Tiled LU factorization (no pivoting): sequential, fork-join, and tasks with a
# last-writer table.
# Run with: julia --project -t 8 scripts/lu_tiled.jl

using LinearAlgebra
using Base.Threads: @spawn, nthreads

BLAS.set_num_threads(1) # parallelism comes from our tasks, not from BLAS

# T[i, j] is a view of tile (i, j) of A; tiles of size ts
function tiles(A, ts)
    n = size(A, 1) ÷ ts
    r(i) = ((i - 1) * ts + 1):(i * ts)
    return [view(A, r(i), r(j)) for i in 1:n, j in 1:n]
end

function lu_tiled!(T)
    n = size(T, 1)
    for k in 1:n
        lu!(T[k, k], NoPivot())                          # diagonal tile
        for j in (k + 1):n
            ldiv!(UnitLowerTriangular(T[k, k]), T[k, j]) # row panel
        end
        for i in (k + 1):n
            rdiv!(T[i, k], UpperTriangular(T[k, k]))     # column panel
        end
        for j in (k + 1):n, i in (k + 1):n
            mul!(T[i, j], T[i, k], T[k, j], -1, 1)       # trailing update
        end
    end
    return
end

# fork-join: parallel loops, a barrier (@sync) after each phase
function lu_forkjoin!(T)
    n = size(T, 1)
    for k in 1:n
        lu!(T[k, k], NoPivot())
        @sync begin # panels
            for j in (k + 1):n
                @spawn ldiv!(UnitLowerTriangular(T[k, k]), T[k, j])
            end
            for i in (k + 1):n
                @spawn rdiv!(T[i, k], UpperTriangular(T[k, k]))
            end
        end
        @sync for j in (k + 1):n, i in (k + 1):n # updates
            @spawn mul!(T[i, j], T[i, k], T[k, j], -1, 1)
        end
    end
    return
end

# run `f` once all `deps` are done (deps are evaluated *now*, at the call)
spawn_after(f, deps...) = @spawn (waitall(deps); f())

# tasks, no barriers: W[i, j] is the last task that wrote tile (i, j).
# A task waits for the last writers of the tiles it touches, then becomes the
# last writer of the tile it writes.
function lu_tasks!(T)
    n = size(T, 1)
    W = fill(@spawn(nothing), n, n)
    for k in 1:n
        W[k, k] = spawn_after(W[k, k]) do
            lu!(T[k, k], NoPivot())
        end
        for j in (k + 1):n # panels
            W[k, j] = spawn_after(W[k, k], W[k, j]) do
                ldiv!(UnitLowerTriangular(T[k, k]), T[k, j])
            end
            W[j, k] = spawn_after(W[k, k], W[j, k]) do
                rdiv!(T[j, k], UpperTriangular(T[k, k]))
            end
        end
        for j in (k + 1):n, i in (k + 1):n # updates
            W[i, j] = spawn_after(W[i, k], W[k, j], W[i, j]) do
                mul!(T[i, j], T[i, k], T[k, j], -1, 1)
            end
        end
    end
    return waitall(W)
end

# relative residual ‖LU - A‖ / ‖A‖ of the factorization computed by f!
function check(f!, A, ts)
    F = copy(A)
    f!(tiles(F, ts))
    return norm(UnitLowerTriangular(F) * UpperTriangular(F) - A) / norm(A)
end

# best of 3 runs, each on a fresh copy of A
bench(f!, A, ts) = minimum(_ -> (T = tiles(copy(A), ts); @elapsed f!(T)), 1:3)

# time and check each version on an m×m matrix, with tiles of size ts
function compare(versions; m = 4000, ts = 250)
    println("threads: ", nthreads())
    A = rand(m, m) + m * I # diagonally dominant: no pivoting needed
    for f! in versions
        err = check(f!, A, ts) # also compiles
        println(rpad(f!, 14), round(bench(f!, A, ts), digits = 3), " s   error: ", err)
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__ # run as a script, not included
    compare((lu_tiled!, lu_forkjoin!, lu_tasks!))
end
