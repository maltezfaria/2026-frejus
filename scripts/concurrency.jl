# Concurrency: several tasks in progress on a single thread.
# Run with: julia --project -t 1 scripts/concurrency.jl

using Downloads: download
using Base.Threads: @spawn

println("threads: ", Threads.nthreads(), " (+", Threads.nthreads(:interactive), " interactive)")

# HAL (French open archive) search API: publications produced in a given year
api = "https://api.archives-ouvertes.fr/search"
hal(year) = "$api/?q=producedDateY_i:$year"  # publications of a given year
urls = hal.(2005:2024)

# sequential: one request after the other
download_seq(urls) = map(download, urls)

# concurrent: a task waiting for the network lets the others run
# (spawn all tasks first, then fetch: `map(u -> fetch(@spawn ...), urls)` is sequential!)
download_tasks(urls) = fetch.(map(u -> @spawn(download(u)), urls))

download_seq(urls[1:2]); download_tasks(urls[1:2]) # compile

print("sequential: "); @time download_seq(urls)
# "lock conflicts": Downloads protects its shared state with locks, and a task
# can be suspended while holding one -- contention even on a single thread
print("tasks:      "); @time download_tasks(urls)
