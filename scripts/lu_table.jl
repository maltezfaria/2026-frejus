# The results table of the tiled LU: each version, for several tile sizes, and
# LAPACK as a reference.
# Run with: julia --project -t 8 scripts/lu_table.jl  (about a minute)

include("lu_dft.jl") # the four versions, tiles, check, bench

m = 4000
A = rand(m, m) + m * I # diagonally dominant: no pivoting needed
sizes = (500, 250, 200, 125, 100)
versions = (lu_tiled!, lu_forkjoin!, lu_tasks!, lu_dft!)

println("$m×$m, ", nthreads(), " threads, time in s (best of 3)")
println(rpad("tile size", 14), join(lpad.(sizes, 6)))
for f! in versions
    print(rpad(f!, 14))
    for ts in sizes
        err = check(f!, A, ts) # also compiles
        err < 1.0e-12 || error("$f!, tile size $ts: error $err")
        print(lpad(round(bench(f!, A, ts), digits = 2), 6))
    end
    println()
end

# reference: LAPACK (with pivoting), parallelized by BLAS
BLAS.set_num_threads(nthreads())
lu(A) # compile
t = minimum(_ -> (B = copy(A); @elapsed lu!(B)), 1:3)
println("LAPACK lu! (", BLAS.get_num_threads(), " BLAS threads): ", round(t, digits = 2))
