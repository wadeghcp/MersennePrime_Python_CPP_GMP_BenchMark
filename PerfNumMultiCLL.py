#!/usr/bin/env python3
"""Mersenne prime search / machine benchmark driver (Python 3, 2026 port of the 2015-2018 tool).

Splits Lucas-Lehmer tests over N workers. Two executors:
  --workers threads    (default) a ThreadPool. The C++ test releases the GIL, so this scales on
                       any Python 3 and scales fully on the free-threaded 3.14t build.
  --workers processes  a ProcessPool, the 2018 behaviour, for comparison.

  PerfNumMultiCLL.py -t 10 -r 10001        # test every prime p <= 10001, 10 workers
  PerfNumMultiCLL.py -p 11213              # one exponent
  PerfNumMultiCLL.py -l 521 607 1279       # a list
  PerfNumMultiCLL.py -n 2 10001            # just count the primes in a range
  PerfNumMultiCLL.py -r 10001 --json out.json
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import lucaslehmer as ll

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = 256


def known_exponents():
    try:
        with open(os.path.join(HERE, "known_mp_list.txt")) as f:
            return {int(x) for x in f.read().split() if x.strip()}
    except OSError:
        return set()


ENGINE = "fft"

def test_one(p, depth=0):
    t0 = time.perf_counter()
    if ENGINE == "fft":
        r = ll.lucas_lehmer_fft(p)          # IBDWT / FFT squaring; exact (GMP redo on rounding trouble)
    else:
        r = ll.lucas_lehmer_split(p, depth) if depth else ll.lucas_lehmer(p)
    return p, r, time.perf_counter() - t0

SPLIT_THREADS = {0: 1, 1: 3, 2: 9}   # threads one test uses at each split depth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--threads", type=int, default=min(10, os.cpu_count() or 1), help="number of workers (default 10 or cpu count)")
    ap.add_argument("--workers", choices=("threads", "processes"), default="threads", help="executor kind (default threads)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("-r", "--prime_range", type=int, default=0, help="test every prime exponent p from 2 to this number")
    g.add_argument("-p", "--prime_value", type=int, default=0, help="test one exponent")
    g.add_argument("-l", "--prime_list", type=int, nargs="+", default=[], help="test these exponents")
    g.add_argument("-n", "--return_num_primes_in_range", type=int, nargs=2, metavar=("LO", "HI"), help="just count the primes in [LO, HI]")
    ap.add_argument("--all", action="store_true", help="print every tested exponent, not only the Mersenne hits")
    ap.add_argument("--json", metavar="FILE", help="write results + timings to FILE")
    ap.add_argument("--progress", type=float, default=10.0, metavar="SEC", help="print a progress line every SEC seconds (0 = off)")
    ap.add_argument("--order", choices=("sorted", "done"), default="sorted",
                    help="sorted (default): results stream out in ascending p, each printed as soon as every smaller "
                         "exponent has finished; done: print in completion order, whatever finishes first")
    ap.add_argument("--engine", choices=("fft", "gmp", "cuda"), default="fft",
                    help="fft = floating-point IBDWT squaring (FFTW/oneMKL, uses the vector units, default); "
                         "gmp = integer GMP squaring; cuda = exact integer NTT on the GPU, one thread block per exponent "
                         "(needs the lucaslehmer_cuda extension: make cuda). All three are exact.")
    ap.add_argument("--cuda-batch", type=int, default=2048, metavar="N", help="cuda: exponents per launch (default 2048)")
    ap.add_argument("--split", type=int, choices=(0, 1, 2), default=0,
                    help="MAX split depth used in the TAIL of a run: 0 = never split (default), 1 = up to 3 threads "
                         "per test, 2 = up to 9. Measured: no gain on ranges; use for a few large exponents. Every test starts plain while the queue is long; once fewer "
                         "unstarted exponents remain than a third of the worker threads, idle threads are folded into the "
                         "remaining tests as split threads. Nothing is reserved up front.")
    ap.add_argument("--gap", type=float, default=3.0, metavar="SEC",
                    help="sorted mode: coalesce output into blocks, flushing a block once it is SEC seconds old "
                         "or 200 lines long (0 = print each line immediately)")
    a = ap.parse_args(argv)
    global ENGINE; ENGINE = a.engine
    if a.engine == "fft" and a.split: print("note: --split applies to the gmp engine only", file=sys.stderr)

    if a.return_num_primes_in_range:
        lo, hi = a.return_num_primes_in_range
        n = len(ll.primes(lo, hi)); print(f"primes in [{lo}, {hi}]: {n}"); return 0
    if a.prime_value: exps = [a.prime_value]
    elif a.prime_list: exps = a.prime_list
    else: exps = ll.primes(2, a.prime_range or 10001)
    if a.threads < 1 or a.threads > MAX_WORKERS:
        print(f"workers must be 1..{MAX_WORKERS}", file=sys.stderr); return 1

    known = known_exponents()
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"python {sys.version.split()[0]}  gil={'on' if gil else 'off'}  gmp {ll.gmp_version}  "
          f"workers={a.threads} ({a.workers})  exponents={len(exps)}  max p={max(exps)}", flush=True)
    hdr = f"{'exponent p':>12} | {'2^p-1 prime?':^13} | {'LL time s':>10} | {'from start s':>12} | {'digits':>8} | known"
    print(hdr); print("-" * len(hdr), flush=True)
    t_start = time.perf_counter(); hits = []; rows = []; done = 0; last_prog = t_start
    Ex = ThreadPoolExecutor if a.workers == "threads" else ProcessPoolExecutor

    def line(p, r, dt, at):
        return (f"{p:>12} | {('YES' if r else 'no'):^13} | {dt:>10.4f} | {at:>12.3f} | "
                f"{ll.mersenne_digits(p) if r else '':>8} | {'yes' if p in known else ('NEW?!' if r else '')}")

    ordered = a.order == "sorted"
    exps_sorted = sorted(exps)
    if a.engine == "cuda":
        # GPU path: one thread block per exponent, launched in ascending groups that share a
        # transform length, in chunks so the ordered output streams. Exponents the kernel cannot
        # take (p < 64, or above its shared-memory limit) run on the CPU FFT engine.
        import lucaslehmer_cuda as lc
        name, sms, mem = lc.device_info()
        print(f"cuda: {name}, {sms} SMs, kernel max p = {lc.max_exponent()}", file=sys.stderr, flush=True)
        chunks = []
        for p in exps_sorted:
            N = lc.ntt_length(p) if p >= 64 else 0
            if chunks and chunks[-1][0] == N and len(chunks[-1][1]) < a.cuda_batch: chunks[-1][1].append(p)
            else: chunks.append((N, [p]))
        block = []
        for N, ps in chunks:
            if N:
                res = {p: (r, dt) for p, r, dt in lc.lucas_lehmer_batch(ps)}
            else:
                res = {}
            for p in ps:
                if p in res and res[p][1] >= 0: r, dt = res[p]
                else:
                    t0 = time.perf_counter(); r = ll.lucas_lehmer_fft(p); dt = time.perf_counter() - t0
                now = time.perf_counter(); at = now - t_start; done += 1
                rows.append({"p": p, "mersenne": r, "ll_s": round(dt, 6)})
                if r: hits.append(p)
                if r or a.all: block.append(line(p, r, dt, at))
            if block: print("\n".join(block), flush=True); block = []
            if a.progress and time.perf_counter() - last_prog >= a.progress:
                print(f"    ... {done}/{len(exps)} done, {len(hits)} hits, {time.perf_counter()-t_start:.0f} s elapsed, "
                      f"last p={ps[-1]} N={N}", file=sys.stderr, flush=True); last_prog = time.perf_counter()
    else:
        # Tail-adaptive split: ONE pool of a.threads plain workers, every test starts on one thread
        # while the unstarted queue is deep (that is where throughput lives: a plain thread is 100%
        # efficient, a 3-way split 59%, a 9-way split 29%). When fewer unstarted exponents remain
        # than threads, the idle threads would go to waste, so tests that start from then on take
        # split threads: depth 1 when remaining*3 <= threads, depth 2 when remaining*9 <= threads,
        # so the split threads never oversubscribe the cores.
        import threading
        remaining = [len(exps)]; rlock = threading.Lock()
        def run_adaptive(p):
            with rlock:
                remaining[0] -= 1; left = remaining[0]
            depth = 0
            if a.split >= 1 and left <= a.threads // 3: depth = 1
            if a.split >= 2 and left <= max(1, a.threads // 9): depth = 2
            return test_one(p, depth)
        results = {}                     # p -> (r, dt, at)   completed, not yet printed (sorted mode)
        frontier = 0                     # index into exps_sorted of the next exponent to print
        block = []; block_born = None
        Ex = ThreadPoolExecutor if a.workers == "threads" else ProcessPoolExecutor
        submit_order = exps_sorted if ordered else sorted(exps, reverse=True)
        if a.workers == "processes":     # no shared counter across processes: split the last threads//2 by position
            n = len(submit_order); cut1 = n - a.threads // 3; cut2 = n - max(1, a.threads // 9)
            def depth_for(i): return 2 if (a.split >= 2 and i >= cut2) else (1 if (a.split >= 1 and i >= cut1) else 0)
        with Ex(max_workers=a.threads) as ex:
            if a.workers == "processes":
                futs = [ex.submit(test_one, p, depth_for(i)) for i, p in enumerate(submit_order)]
            else:
                futs = [ex.submit(run_adaptive, p) for p in submit_order]
            for f in as_completed(futs):
                p, r, dt = f.result(); done += 1
                now = time.perf_counter(); at = now - t_start
                rows.append({"p": p, "mersenne": r, "ll_s": round(dt, 6)})
                if r: hits.append(p)
                if not ordered:
                    if r or a.all: print(line(p, r, dt, at), flush=True)
                else:
                    results[p] = (r, dt, at)
                    while frontier < len(exps_sorted) and exps_sorted[frontier] in results:
                        q = exps_sorted[frontier]; rq, dq, aq = results.pop(q); frontier += 1
                        if rq or a.all:
                            if not block: block_born = now
                            block.append(line(q, rq, dq, aq))
                    if block and (a.gap == 0 or len(block) >= 200):
                        print("\n".join(block), flush=True); block = []
                if a.progress and now - last_prog >= a.progress:
                    print(f"    ... {done}/{len(exps)} done, {len(hits)} hits, {now-t_start:.0f} s elapsed"
                          + (f", next in order: p={exps_sorted[frontier]}" if ordered and frontier < len(exps_sorted) else ""),
                          file=sys.stderr, flush=True); last_prog = now
                if ordered and block and a.gap and now - block_born >= a.gap:
                    print("\n".join(block), flush=True); block = []
        while ordered and frontier < len(exps_sorted):
            q = exps_sorted[frontier]; rq, dq, aq = results.pop(q); frontier += 1
            if rq or a.all: block.append(line(q, rq, dq, aq))
        if block: print("\n".join(block), flush=True)
    total = time.perf_counter() - t_start
    hits.sort()
    print("-" * len(hdr), flush=True)
    print(f"tested {len(exps)} exponents in {total:.3f} s wall, {sum(r['ll_s'] for r in rows):.3f} s of LL work, "
          f"{len(hits)} Mersenne primes: {hits}")
    missing = [p for p in known if p <= max(exps) and p not in hits and p >= min(exps)]
    if missing: print(f"WARNING: known Mersenne exponents in range NOT found: {missing}", file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"python": sys.version.split()[0], "gil": gil, "gmp": ll.gmp_version, "workers": a.threads,
                       "executor": a.workers, "engine": a.engine, "fft_backend": ll.fft_backend, "split": a.split, "order": a.order, "wall_s": round(total, 3), "hits": hits, "rows": rows}, f, indent=1)
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
