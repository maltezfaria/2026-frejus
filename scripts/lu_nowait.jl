# 2×2 block LU with tasks but NO coordination: a silent, nondeterministic bug.
# Run with: julia --project -t 1 scripts/lu_nowait.jl   (correct, by luck)
#      and: julia --project -t 4 scripts/lu_nowait.jl   (garbage, changes every run)

using LinearAlgebra
using Base.Threads: @spawn

BLAS.set_num_threads(1)

function lu_nowait!(A)
    m = size(A, 1); n = m ÷ 2
    A11, A12 = view(A, 1:n, 1:n), view(A, 1:n, n+1:m)
    A21, A22 = view(A, n+1:m, 1:n), view(A, n+1:m, n+1:m)
    t1 = @spawn lu!(A11, NoPivot())                       # 1
    t2 = @spawn ldiv!(UnitLowerTriangular(A11), A12)      # 2: needs t1!
    t3 = @spawn rdiv!(A21, UpperTriangular(A11))          # 3: needs t1!
    t4 = @spawn begin                                     # 4: needs t2, t3!
        mul!(A22, A21, A12, -1, 1)
        lu!(A22, NoPivot())
    end
    waitall((t1, t2, t3, t4))
    return A
end

println("threads: ", Threads.nthreads())

m = 1000
A = rand(m, m) + m * I # diagonally dominant: no pivoting needed
F = lu(A, NoPivot())   # reference

for run in 1:5
    B = lu_nowait!(copy(A))
    err = norm(B - F.factors) / norm(F.factors)
    println("run $run: relative error = ", err)
end
