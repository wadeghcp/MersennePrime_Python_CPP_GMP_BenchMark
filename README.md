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
`pybind11`. `make FFT=mkl` builds the FFT engine on oneMKL's FFTW3 wrapper instead (AVX-512 FFTs, ~8%
faster per test here; `MKLROOT` defaults to `/opt/intel/oneapi/mkl/latest`; with Debian/Ubuntu's
`libmkl-dev` use `make FFT=mkl MKLINC=/usr/include/mkl MKLLIB=/usr/lib/x86_64-linux-gnu`). It links
`libmkl_rt` only: the split MKL libraries cannot be loaded from a Python extension ("cannot load
libmkl_avx512.so ... undefined symbol"). Note that installing Debian's MKL packages offers to make
MKL the system-wide BLAS/LAPACK alternative; decline that on a machine you care about, this build
does not need it.

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
| Xeon w5-3435X, 16 cores / 32 threads, 3.4 GHz base | 32 | 1526.7 s (`--engine gmp`), 442.5 s (`--engine fft`) |
| Xeon Platinum 8480+ (Sapphire Rapids), 2.0 GHz base | 128 | 337 s (`--engine gmp`) |
| Xeon Platinum 8480+, `make FFT=mkl` | 128 | 100 s (`--engine fft`); `-r 200001` (17,984 exponents, 29 primes): 777 s |

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

`-r 30001` on 32 threads: 28.3 s GMP, 15.2 s FFT. `-r 100001` on 32 threads: 1526.7 s GMP, 442.5 s FFT
(3.45x), same 28 primes.

Prime95's hand-written assembly is another 2-3x beyond a library FFT at these sizes; that gap is
two decades of GIMPS tuning, not algorithm.

## The CUDA engine (`--engine cuda`)

`make cuda` builds `lucaslehmer_cuda` (nvcc, `CUDA_ARCH=sm_86` by default, `NVCC_CCBIN=gcc-12`
if your CUDA wants an older host compiler). It is the same IBDWT idea as the FFT engine, done in
a finite field instead of floating point, so it is exact by construction and needs no fp64, which
GeForce parts only have at 1/64 rate.

- Field: GF(q), q = 2^64 - 2^32 + 1. 2^32 divides q - 1, so power-of-two NTTs exist, and 2 has
  order 192, so an N-th root of 2 exists for every N | 2^26. That root supplies the "irrational"
  weights 2^(ceil(jp/N) - jp/N) as field elements, and one cyclic convolution of the weighted
  digits is the squaring modulo 2^p - 1 with no reduction step.
- One thread block per exponent runs the whole test on-device: forward NTT (decimation in
  frequency), pointwise square, inverse NTT (decimation in time, so no bit reversal), unweight,
  balanced carry in parallel chunks with the top carry wrapping into word 0. The digits never
  leave shared memory until the end; the final residue is rebuilt with GMP on the host.
- Exactness: every convolution digit is an integer below N * 2^(2B - 1), and N is chosen so that
  is under q/2, so the field value is the integer. Nothing to round, no error bound to watch.
- Limit of this version: shared memory, N <= 8192, so p <= 204,800 on sm_86. Larger p needs a
  global-memory (four-step) NTT, which is also where a GPU starts to win outright.

Measured, same exponents and the same 28 primes every time:

| range | CPU FFT engine, 32 Xeon w5-3435X threads | CUDA engine, RTX 3070 Ti (48 SMs, 8 GB) |
|---|---|---|
| `-r 30001` | 15.2 s | 11.1 s |
| `-r 100001` | 442.5 s | 394.2 s |

A 2021 consumer card with no usable fp64 edging out a 16-core Sapphire Rapids workstation with
AVX-512 MKL transforms, exactly, with a kernel that has had no tuning yet (radix-2 stages, a
`__syncthreads` per stage, twiddles read from shared memory). GMP on the same Xeon: 1526.7 s.

## Self-checking runs: res64, `--verify`, `--loop` (a GIMPS-style stress test)

Every result line now carries **res64**, the low 64 bits of the final Lucas-Lehmer residue,
the same fingerprint GIMPS uses to cross-check runs between machines. It is 0 for a prime and
otherwise a value that depends on every one of the p-2 squarings: a single flipped bit anywhere
in the run, in any engine, changes it. All four engines (GMP, split GMP, FFT, CUDA) produce
identical res64 for every exponent, which is how they are checked against each other.

That turns the driver into a stress test in the prime95 torture-test family, with a known-good
answer for every exponent rather than only for the known primes:

```
PerfNumMultiCLL.py -t 128 -r 200001 --json ref.json               # reference pass on a known-good box
PerfNumMultiCLL.py -t 128 -r 200001 --verify ref.json             # any mismatch -> reported, exit code 3
PerfNumMultiCLL.py -t 128 -r 200001 --loop 0                      # pass 1 = reference, then verify forever
```

The exponent picks the footprint. The FFT engine's working set is about 20 bytes per transform
point per thread, so `-r 100001` (transform 6,144) lives in each core's L2, `-p 1000003` (65,536
points, ~1.5 MB) sits at the L2/L3 boundary, and `-p 10000019` (~12 MB per thread) drives L3 and
the memory controllers; 128 threads of that is a memory-subsystem workload. The CUDA engine is a
shared-memory and L2 workload: the residue never leaves the SM, host traffic is a burst per launch,
and DRAM is idle.

## Cross-checked against prime95 and gpuowl at 3 million bits, and a bug found upstream

`lucaslehmer.gmp_res64_after(p, iters)` and `fft_res64_after(p, iters)` run only the first
`iters` squarings and return the residue, so an engine can be checked against another program's
interim residues without running a multi-hour test to the end. That check found a wrong answer in
the reference GPU implementation. The chain of evidence, in order:

**1. The observation.** PRPLL (gpuowl master 4d0e759, 2025-12-14), run as a five-minute smoke test
on M37 (p = 3,021,377, a known prime) with its default transform for that size, reported the prime
as **composite**: `-ll 3021377` -> "status":"C", res64 `a0ec9abad5afd56e`, in 3 min 5 s on an RTX
3070 Ti. The same residue on every rerun, so deterministic, not a hardware fault.

**2. The oracle.** The residue after 20,000 squarings from s_0 = 4 is a checkable fingerprint:

| implementation | res64 after 20,000 squarings |
|---|---|
| prime95 v30.19 b20 (`InterimResidues=20000`; it counts the seed as iteration 2, so its "iteration 20002" line) | `CE811E3772129824` |
| this repo, GMP engine, exact integer arithmetic | `ce811e3772129824` |
| this repo, FFT engine, fp64, worst rounding error 7e-4 | `ce811e3772129824` |
| PRPLL, "FP32+M61" 256K transform, RTX 3070 Ti | `d7979bd162116bee` |

Three independent implementations agree to the bit, and prime95 is the author's own CPU code.
It is not an off-by-one: our residues at 19,999 and 20,001 match neither. PRPLL's own `-prp` mode
on the same configuration trips its Gerbicz error check at the first block, so the program can
detect the error; plain Lucas-Lehmer has no check and reports the wrong residue as a verdict.

**3. Localization.** Same exponent, every transform type PRPLL offers, 20,000 iterations each:
fp64, M61-only and FP32+M31+M61 are correct; M31+M61 and FP32+M61, every variant, are wrong with
the identical residue. A bits-per-word sweep on FP32+M61 (a prime near 3.3M, 3.5M, 3.8M, 4.1M,
4.4M, 4.7M, 5.0M on the same 256K transform) is wrong at 11.53 and 12.59 bits per word and right
from 13.35 up, and `-carry long`, which routes through the older expanded carry kernels, is right
everywhere. So: the fused carry kernel, the two types that hold 96-bit intermediates, low bits per
word.

**4. The mechanism.** In the fused kernel, `carryStepSignedSloppy(i96)` returns a full 32-bit word
instead of an nBits-wide one whenever a digit exceeds 2^31, on the grounds that the transform is
rated for 34 bits per word. The transform copes, but the convolution digits then reach ~2^63
instead of ~2^(2 BPW + 9), and the carry that `carryFinal` computes into a hard-coded `i32`
temporary needs about 63 - 2 nBits bits: more than 31 once nBits <= 15. A probe on that temporary,
reported through the round-off statistic, measured over the sampled iterations:

| bits per word | bits the i32 temporary must hold | fits | PRPLL result |
|---|---|---|---|
| 11.53 | 39.5 | no | wrong |
| 12.59 | 36.2 | no | wrong |
| 13.35 | 33.9 | no | wrong (marginal: LL passed 20,000 iterations once, PRP and STATS runs fail) |
| 14.50 | 29.5 | yes | right |
| 17.93 | 23.4 | yes | right |
| any of these, with the fix | 9.4 | yes | right |

The boundary is where 63 - 2 nBits crosses 31, which is the observed transition. The author had
guarded the identical branch in the 64-bit variant with `EXP / NWORDS >= 23` and left the comment
"for reasons I don't fully understand the sloppy case fails if BPW is too low"; the 96-bit variant
had no guard.

**5. The fix.** The same one-line guard on the 96-bit variant (derived requirement nBits >= 16;
23 keeps the author's margin). Verified on the 3070 Ti: both types correct at
11.53, 12.59 and 13.35 bits per word, `-ll 3021377` returns "3021377 is PRIME!" with res64 0,
`-prp` runs clean to 100,000 iterations with the error margin up seven orders of magnitude, 19.07
bits per word unchanged at 59 us/iteration. Submitted upstream from
`wadeghcp/gpuowl`, branch `fix-i96-sloppy-carry32`.

**6. Why it matters, and why nobody had hit it.** The code is three months old and unreleased,
production exponents sit near 19 bits per word where the temporary fits, and GIMPS runs PRP with
the check, so no prime was ever at risk. What breaks is the small case: the five-minute known-prime
run that a validation engineer uses to decide whether to trust a new GPU, a new driver, or a
day-long test on the same machine. A self-check that fails only at the small end is the worst place
for a weak link, because that is the case everything else is trusted on. That is the case this
repo's engines are built to make exact, and it is why every engine here carries res64.

Per-iteration cost at that size, this box (Xeon w5-3435X), is the honest picture of where a
single monolithic transform stands against the state of the art:

| p = 3,021,377, one squaring | ms |
|---|---|
| prime95, AVX-512 192K FFT, 1 thread | 0.54 |
| prime95, 10 threads | 0.10 |
| GMP engine (exact) | 4.1 |
| FFT engine (FFTW, one 196,608-point transform) | 6.6 |
| PRPLL on the RTX 3070 Ti (wrong answer before the fix; 59 us after) | 0.06 |

At p = 44,497 the FFT engine is within 1.4x of prime95 (4,608-point FFT: prime95 7 us/iteration,
FFT engine 10 us); at p ~ 1e5 it trails by about 3x (6K FFT: prime95 8.5 us, FFT engine 25 us).
At 3e6 it trails by 12x and falls behind GMP: a 3 MB working set no longer fits a core's L2, and
one big transform streams it from L3 on every pass, where prime95's Pass1/Pass2 split keeps each
pass in cache and fuses weighting and carry into the transform. That two-pass ("four-step")
structure is the next engine, on the CPU and on the GPU alike.

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
