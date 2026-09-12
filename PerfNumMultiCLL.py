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


def test_one(p, depth=0):
    t0 = time.perf_counter()
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
    ap.add_argument("--split", type=int, choices=(0, 1, 2), default=1,
                    help="threads per test for the LARGEST exponents: 0 = plain GMP (1 thread), 1 = 3 threads, "
                         "2 = 9 threads (default 1). The big exponents run on a reserved pool with this many "
                         "threads each, so the tail of a run keeps the machine busy instead of going single-threaded.")
    ap.add_argument("--big-share", type=float, default=0.25, metavar="FRAC",
                    help="fraction of exponents (the largest) treated as big and run split on the reserved pool")
    ap.add_argument("--gap", type=float, default=3.0, metavar="SEC",
                    help="sorted mode: coalesce output into blocks, flushing a block once it is SEC seconds old "
                         "or 200 lines long (0 = print each line immediately)")
    a = ap.parse_args(argv)

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
    # Level 1 + 2 scheduling: the largest --big-share of the exponents go to a "big" pool where each
    # test is itself split over SPLIT_THREADS[--split] threads and starts LARGEST FIRST; everything
    # else goes to a "small" pool of plain tests that walks up from the bottom so the in-order
    # frontier advances at once. Both pools feed one result stream.
    per = SPLIT_THREADS[a.split]
    nbig = int(len(exps_sorted) * a.big_share) if (a.split and len(exps_sorted) >= 8) else 0
    big = exps_sorted[len(exps_sorted) - nbig:] if nbig else []
    small = exps_sorted[:len(exps_sorted) - nbig]
    small_threads = max(1, min(len(small), a.threads // 8 if big else a.threads))
    big_workers = max(1, (a.threads - small_threads) // per) if big else 0
    if big:
        print(f"scheduling: {len(small)} small exponents on {small_threads} plain threads (ascending); "
              f"{len(big)} big exponents (p >= {big[0]}) on {big_workers} slots x {per} threads each (descending)", flush=True)
    results = {}                     # p -> (r, dt, at)   completed, not yet printed (sorted mode)
    frontier = 0                     # index into exps_sorted of the next exponent to print
    block = []; block_born = None
    pools = []
    Ex = ThreadPoolExecutor if a.workers == "threads" else ProcessPoolExecutor
    futs = []
    if big:
        pb = Ex(max_workers=big_workers); pools.append(pb)
        futs += [pb.submit(test_one, p, a.split) for p in sorted(big, reverse=True)]
    ps_ = Ex(max_workers=small_threads); pools.append(ps_)
    futs += [ps_.submit(test_one, p, 0) for p in (small if ordered else sorted(small, reverse=True))]
    try:
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
    finally:
        for pl in pools: pl.shutdown(wait=True)
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
                       "executor": a.workers, "split": a.split, "order": a.order, "wall_s": round(total, 3), "hits": hits, "rows": rows}, f, indent=1)
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
