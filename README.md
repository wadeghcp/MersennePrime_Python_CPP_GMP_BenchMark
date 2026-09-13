# MersennePrime_Python_CPP_GMP_BenchMark

A Mersenne-prime finder and a quick, honest machine benchmark: a Python 3 driver that spreads
Lucas-Lehmer tests over N workers, wrapping a small C++ core on the GNU Multiple Precision
library (GMP). Originally written 2015-2018 for Python 2.7; ported in 2026 to Python 3.9+
including the **free-threaded CPython 3.14t** build, where the worker threads run truly in
parallel because the C++ test releases the GIL.

Mersenne primes have the form M_p = 2^p - 1 with p prime. Lucas-Lehmer decides M_p's primality
in p-2 squarings modulo M_p, which is where all the time goes; GMP's FFT multiplication is what
makes it feasible. 52 Mersenne primes are known as of 2026 (`known_mp_list.txt`); the largest,
2^136,279,841 - 1, has 41 million digits and was found by GIMPS in October 2024.

## What changed in the 2026 port

| | 2018 | 2026 |
|---|---|---|
| Python | 2.7 | 3.9+, tested on 3.12 and free-threaded 3.14.7t |
| binding | PyBindGen, `PyObject*` in/out, big ints round-tripped as decimal strings | pybind11, plain C++ (`unsigned long` in, `bool` out), GIL released in the loop |
| big ints | boost::multiprecision over GMP | `gmp.h` directly |
| mod 2^p-1 | `mpz_mod` | shift-and-add reduction (same result, cheaper) |
| prime sieve | O(n^2)-ish vector-erase loop | sieve of Eratosthenes |
| parallelism | `multiprocessing.Process` per exponent | `ThreadPoolExecutor` (default) or `ProcessPoolExecutor` |
| build | hand Makefile against `/usr/include/python2.7` | `make` for the Python on your PATH, or `pip install .` |
| licence | none | MIT |

The 2018 sources are kept under `legacy/` for reference. A thin `LucasLehmer` class with the old
method names is still exported for anyone who imported it.

## Build

Requires a C++17 compiler, GMP and FFTW3 headers (`libgmp-dev libfftw3-dev` on Debian/Ubuntu), and
`pybind11`. `make FFT=mkl` builds the FFT engine on oneMKL's FFTW3 wrapper instead (AVX-512 FFTs;
`MKLROOT` defaults to `/opt/intel/oneapi/mkl/latest`).

```sh
python3 -m pip install pybind11            # once, into the interpreter you will use
make                                       # -> lucaslehmer.<abi>.so beside the sources
make test                                  # pytest
# or, as a package:
pip install .
```

For the free-threaded interpreter with [uv](https://docs.astral.sh/uv/):

```sh
uv python install 3.14t
uv venv --python 3.14t .venv && uv pip install pybind11 pytest
make PYTHON=.venv/bin/python
.venv/bin/python -c "import sys; print(sys._is_gil_enabled())"   # False
```

## Run

```
PerfNumMultiCLL.py -t 32 -r 100001         # ordered output, blocks every 3 s, FFT engine on every thread
PerfNumMultiCLL.py -t 32 -r 100001 --engine gmp        # integer GMP squaring instead
PerfNumMultiCLL.py -t 32 -r 100001 --engine gmp --split 2 --gap 5   # tail tests split 9-way once cores go idle
PerfNumMultiCLL.py -t 32 -r 100001 --order done   # completion order
PerfNumMultiCLL.py -t 16 -r 10001          # every prime p <= 10001 on 16 workers
PerfNumMultiCLL.py -p 11213                # one exponent
PerfNumMultiCLL.py -l 521 607 1279         # a list
PerfNumMultiCLL.py -n 2 10001              # just count primes in a range (1229)
PerfNumMultiCLL.py -r 10001 --workers processes --json run.json
```

Output is a table of the Mersenne hits (add `--all` for every exponent), the Lucas-Lehmer time
per exponent, and a wall-clock total; the exit status is 2 if a known Mersenne exponent in the
range was missed, which makes the run its own correctness check.

## Measured on the port's development box

Xeon w5-3435X, GMP 6.3.0, g++ 13.3, `-r 10001` (1229 exponents, 22 Mersenne primes):

| interpreter | workers | wall |
|---|---|---|
| 3.14.7 free-threaded, 16 threads | threads | 0.79 s |
| 3.14.7 free-threaded, 16 processes | processes | 0.87 s |
| 3.12.3 (GIL), 16 threads | threads | 0.79 s |

The GIL build keeps up because the extension releases the GIL for the whole Lucas-Lehmer loop;
what the free-threaded build removes is the serialisation of everything around it. Sixteen
tests near p = 20,000 run 11x faster on sixteen threads than serially on 3.14t. The 2018 README
recorded "reasonable time" for the first 200,000 whole numbers on a 12-core server; that range
is now a few minutes.

`-r 100001` (9592 exponents, 28 Mersenne primes), free-threaded 3.14.7t, threads:

| host | threads | wall |
|---|---|---|
| Xeon w5-3435X, 16 cores / 32 threads, 3.4 GHz base | 32 | 1526.7 s (`--engine gmp`) |
| Xeon Platinum 8480+ (Sapphire Rapids), 2.0 GHz base | 128 | 337 s (`--engine gmp`) |

The 8480+ does p = 44,497 in 1.55 s on one thread against 2.9 s on the w5, but a 128-thread run
slows every test 1.57x (SMT siblings and all-core clocks), so the 4x thread count buys 4.5x.

## The FFT engine (`--engine fft`, default)

GMP squares with the integer carry chain (`mulx`/`adcx` on 64-bit limbs), which the vector units
cannot help. Prime95 and gpuowl get their speed from a different algorithm: the Crandall-Fagin
irrational-base discrete weighted transform (IBDWT). The residue is held as N balanced digits in
a variable base of ceil((j+1)p/N) - ceil(jp/N) bits each and pre-weighted by 2^(ceil(jp/N) - jp/N);
one cyclic convolution of the weighted digits is then the squaring *modulo 2^p - 1* with no
separate reduction. The convolution is a real FFT, a pointwise complex square and an inverse FFT
(FFTW3 or oneMKL), followed by rounding to integers and a carry pass. That is where AVX-512 goes
to work.

It stays exact. Every iteration measures the worst distance of any digit from an integer; a test
that ever exceeds 0.35 is reported unreliable and redone with GMP, and the final residue is
rebuilt as an exact integer with GMP before the zero test. At the default 18 bits per word the
worst error seen over a full p = 110,503 test is 4e-3. `lucaslehmer.set_fft_bits_per_word()`
trades FFT length for rounding margin; `lucaslehmer.fft_info(p)` returns (N, bits/word, worst
error) for a full run of p.

| one test, one thread | GMP | FFT (FFTW3, AVX2 build) | FFT (oneMKL, AVX-512) |
|---|---|---|---|
| p = 44,497 | 1.32 s | 0.45 s (2.9x) | 0.45 s (2.9x) |
| p = 86,243 | 6.58 s | 1.95 s (3.4x) | 1.79 s (3.7x) |
| p = 110,503 | 11.54 s | 3.00 s (3.8x) | 2.76 s (4.2x) |

`-r 30001` on 32 threads: 28.3 s GMP, 15.2 s FFT. FFT_100K_ROW

Prime95's hand-written assembly is another 2-3x beyond a library FFT at these sizes; that gap is
two decades of GIMPS tuning, not algorithm.

## Splitting one test across threads (`--split`)

GMP multiplies on one thread, so a run's tail, the last few huge exponents, used to leave the
machine idle. `lucas_lehmer_split(p, depth)` squares by Karatsuba on halves: with s = a·2^k + b,
s² = a²·2^2k + 2ab·2^k + b² and 2ab = (a+b)² − a² − b², three independent half-size squarings on
three threads (depth 1) or nine quarter-size ones on nine (depth 2), then the same shift-and-add
reduction modulo 2^p − 1. Persistent workers with a spinning barrier keep the per-iteration cost
near a microsecond, which matters because a 100 kbit squaring is itself only ~100 µs.

| one test | plain | `--split 1`, 3 threads | `--split 2`, 9 threads |
|---|---|---|---|
| p = 44,497 | 1.32 s | 0.83 s (1.58x) | 0.71 s (1.87x) |
| p = 86,243 | 6.58 s | 3.71 s (1.78x) | 2.70 s (2.44x) |
| p = 110,503 | 11.54 s | 6.49 s (1.78x) | 4.49 s (2.57x) |

The split applies to the GMP engine. A split thread does less useful work than a plain one (3-way is 59% efficient, 9-way 29%), so
the driver never splits while the queue is deep: every test starts on one thread. Only when
fewer unstarted exponents remain than a third of the threads do new tests take 3 threads, and
under a ninth, 9, so idle cores are folded into the tail without oversubscribing. `--split`
caps that depth and defaults to 0 because on a range it measures as a wash: `-r 100001` on 32
threads took 1526.7 s plain and 1540.8 s with `--split 2`, the tail being under 5% of the wall
and the split's overhead eating what it recovers. More cores, not more split.
Split pays when one exponent dominates, p in the millions, or when a handful of exponents run
on a big machine.
`lucaslehmer.bench_square(bits, iters, depth)` measures one squaring at any size.

`AnalyzeKnown.py` plots ratios between successive known exponents (needs numpy, matplotlib,
scipy, natsort: `pip install .[analyze]`).

## Notes kept from 2018

The bottleneck is the squaring inside Lucas-Lehmer, executed p-2 times per exponent; there is
no memoisation across exponents because the intermediate values are as large as M_p itself. A
single test is inherently single-threaded; parallelism is across exponents. FPGA and huge-cache
Xeon speculation from the original README stands as written; GIMPS remains the tool that
actually finds records.
