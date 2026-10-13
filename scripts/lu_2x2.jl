# 2×2 tiled LU factorization with hand-written task dependencies.
# Run with: julia --project -t 4 scripts/lu_2x2.jl

using LinearAlgebra
using Base.Threads: @spawn
BLAS.set_num_threads(1) # parallelism comes from our tasks, not from BLAS

println("threads: ", Threads.nthreads())

function tiles(A)
    m = size(A, 1); n = m ÷ 2
    return view(A, 1:n, 1:n), view(A, 1:n, n+1:m),
           view(A, n+1:m, 1:n), view(A, n+1:m, n+1:m)
end

function lu_2x2_seq!(A)
    A11, A12, A21, A22 = tiles(A)
    lu!(A11, NoPivot())                          # 1
    ldiv!(UnitLowerTriangular(A11), A12)         # 2
    rdiv!(A21, UpperTriangular(A11))             # 3
    mul!(A22, A21, A12, -1, 1); lu!(A22, NoPivot()) # 4
    return A
end

function lu_2x2_tasks!(A)
    A11, A12, A21, A22 = tiles(A)
    t1 = @spawn lu!(A11, NoPivot())
    t2 = @spawn (wait(t1); ldiv!(UnitLowerTriangular(A11), A12))
    t3 = @spawn (wait(t1); rdiv!(A21, UpperTriangular(A11)))
    t4 = @spawn begin
        wait(t2); wait(t3)
        mul!(A22, A21, A12, -1, 1)  # A22 -= A21 * A12
        lu!(A22, NoPivot())
    end
    wait(t4)
    return A
end

m = 4000
A = rand(m, m) + m * I # diagonally dominant: no pivoting needed
F = lu(A, NoPivot())

lu_2x2_seq!(copy(A)); lu_2x2_tasks!(copy(A)) # warm-up

B = copy(A); print("sequential: "); @time lu_2x2_seq!(B)
println("sequential error: ", norm(B - F.factors) / norm(F.factors))

B = copy(A); print("tasks:      "); @time lu_2x2_tasks!(B)
println("tasks error:      ", norm(B - F.factors) / norm(F.factors))
