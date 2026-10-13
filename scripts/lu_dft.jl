# Tiled LU with DataFlowTasks: the sequential loops + data annotations.
# Run with: julia --project -t 8 scripts/lu_dft.jl

include("lu_tiled.jl") # tiles, lu_tiled!, lu_forkjoin!, lu_tasks!, compare
using DataFlowTasks

function lu_dft!(T)
    n = size(T, 1)
    for k in 1:n
        @dspawn lu!(@RW(T[k, k]), NoPivot()) label = "lu ($k,$k)"
        for j in (k + 1):n # panels
            @dspawn ldiv!(UnitLowerTriangular(@R(T[k, k])), @RW(T[k, j])) label = "ldiv ($k,$j)"
            @dspawn rdiv!(@RW(T[j, k]), UpperTriangular(@R(T[k, k]))) label = "rdiv ($j,$k)"
        end
        for j in (k + 1):n, i in (k + 1):n # updates
            @dspawn mul!(@RW(T[i, j]), @R(T[i, k]), @R(T[k, j]), -1, 1) label = "mul ($i,$j)"
        end
    end
    # a last task reading the whole matrix: it runs after every write, so
    # waiting for it is waiting for the factorization
    A = parent(T[1, 1]) # the matrix itself (@R(T) is a matrix of views)
    return wait(@dspawn @R(A) label = "done")
end

if abspath(PROGRAM_FILE) == @__FILE__ # run as a script, not included
    compare((lu_tiled!, lu_forkjoin!, lu_tasks!, lu_dft!))
end
