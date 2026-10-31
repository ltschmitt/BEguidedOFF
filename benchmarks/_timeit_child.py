"""Run a command, report wall time + peak child RSS (for index builds)."""
import resource
import subprocess
import sys
import time

t0 = time.perf_counter()
rc = subprocess.call(sys.argv[1:])
wall = time.perf_counter() - t0
peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(f"BENCH_META exit={rc} wall_s={wall:.1f} peak_rss_gb={peak_kb / 1048576:.2f}")
sys.exit(rc)
