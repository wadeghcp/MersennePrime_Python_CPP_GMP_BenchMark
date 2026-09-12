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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import lucaslehmer as ll

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = 256


def known_exponents():
    try:
        with open(os.path.join(HERE, "known_mp_list.txt")) as f:
            return {int(x) for x in f.read().split() if x.strip()}
    except OSError:
        return set()


def test_one(p):
    t0 = time.perf_counter()
    r = ll.lucas_lehmer(p)
    return p, r, time.perf_counter() - t0


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
          f"workers={a.threads} ({a.workers})  exponents={len(exps)}  max p={max(exps)}")
    hdr = f"{'exponent p':>12} | {'2^p-1 prime?':^13} | {'LL time s':>10} | {'from start s':>12} | {'digits':>8} | known"
    print(hdr); print("-" * len(hdr))
    t_start = time.perf_counter(); hits = []; rows = []
    Ex = ThreadPoolExecutor if a.workers == "threads" else ProcessPoolExecutor
    with Ex(max_workers=a.threads) as ex:
        for p, r, dt in ex.map(test_one, sorted(exps, reverse=True)):   # biggest first: better load balance
            rows.append({"p": p, "mersenne": r, "ll_s": round(dt, 6)})
            if r: hits.append(p)
            if r or a.all:
                print(f"{p:>12} | {('YES' if r else 'no'):^13} | {dt:>10.4f} | {time.perf_counter()-t_start:>12.3f} | "
                      f"{ll.mersenne_digits(p) if r else '':>8} | {'yes' if p in known else ('NEW?!' if r else '')}")
    total = time.perf_counter() - t_start
    hits.sort()
    print("-" * len(hdr))
    print(f"tested {len(exps)} exponents in {total:.3f} s wall, {sum(r['ll_s'] for r in rows):.3f} s of LL work, "
          f"{len(hits)} Mersenne primes: {hits}")
    missing = [p for p in known if p <= max(exps) and p not in hits and p >= min(exps)]
    if missing: print(f"WARNING: known Mersenne exponents in range NOT found: {missing}", file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"python": sys.version.split()[0], "gil": gil, "gmp": ll.gmp_version, "workers": a.threads,
                       "executor": a.workers, "wall_s": round(total, 3), "hits": hits, "rows": rows}, f, indent=1)
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
