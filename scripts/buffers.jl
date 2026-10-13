# Many systems: reuse a buffer instead of allocating a factorization per system.
# Run with: julia --project scripts/buffers.jl

using LinearAlgebra

BLAS.set_num_threads(1)

N, m = 100, 1000
As = [rand(m, m) + m * I for _ in 1:N]
bs = [rand(m) for _ in 1:N]

# allocate: every `lu` creates a new factorization
function solve_alloc(As, bs)
    xs = similar(bs)
    for i in eachindex(As)
        F = lu(As[i])
        xs[i] = F \ bs[i]
    end
    return xs
end

# buffer: allocate one factorization, then overwrite it
function solve_buffer(As, bs)
    xs = similar(bs)
    F = lu(As[1])
    for i in eachindex(As)
        lu!(F, As[i]) # writes into F, leaves As[i] untouched
        xs[i] = F \ bs[i]
    end
    return xs
end

solve_alloc(As, bs); solve_buffer(As, bs) # compile

print("allocate: "); @time solve_alloc(As, bs)
print("buffer:   "); @time solve_buffer(As, bs)
