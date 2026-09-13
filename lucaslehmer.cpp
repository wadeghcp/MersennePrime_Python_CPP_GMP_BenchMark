// lucaslehmer: Lucas-Lehmer Mersenne-prime test on GMP, exposed to Python via pybind11.
//
// 2015-2018 original (Python 2.7 + PyBindGen + boost::multiprecision): PyObject* in, PyObject*
// out, big ints round-tripped through decimal strings. 2026 port: plain C++ on gmp.h, pybind11,
// Python 3.9+ including the free-threaded 3.14t build. The LL loop releases the GIL, so on a
// free-threaded interpreter (and even on a GIL build) a ThreadPool of tests runs in parallel
// without multiprocessing.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gmp.h>
#include <vector>
#include <cstdint>
#include <string>
#include <tuple>
#include <stdexcept>

namespace py = pybind11;

// GIMPS-style res64: the low 64 bits of the final Lucas-Lehmer residue s_{p-2} mod 2^p - 1.
// Zero for a prime, otherwise a fingerprint of the whole computation: any silent data corruption
// in any of the p-2 squarings changes it, which is what makes a run self-checking against a
// reference. Each engine stores it here (thread-local) and the *_res() bindings return it.
static thread_local unsigned long long g_last_res64 = 0;
static inline void note_res64(const mpz_t s) { g_last_res64 = mpz_get_ui(s); }   // low limb (64-bit GMP)


// Lucas-Lehmer: M_p = 2^p - 1 is prime iff s_{p-2} == 0 with s_0 = 4, s_{k+1} = s_k^2 - 2 (mod M_p).
// p must itself be prime for the test to mean anything; p == 2 is handled (M_2 = 3 is prime).
static bool lucas_lehmer(unsigned long p) {
    if (p < 2) return false;
    if (p == 2) return true;
    if (p % 2 == 0) return false;
    mpz_t M, s;
    mpz_init(M); mpz_init(s);
    mpz_set_ui(M, 1); mpz_mul_2exp(M, M, p); mpz_sub_ui(M, M, 1);   // M = 2^p - 1
    mpz_set_ui(s, 4);
    {
        py::gil_scoped_release release;   // the hot loop runs without the GIL
        for (unsigned long i = 0; i + 2 < p; ++i) {
            mpz_mul(s, s, s);
            mpz_sub_ui(s, s, 2);
            // s mod (2^p - 1) via the bit trick: s = (s & M) + (s >> p), repeated once more if needed.
            // Cheaper than mpz_mod for this special modulus; result identical.
            mpz_t hi; mpz_init(hi);
            mpz_tdiv_q_2exp(hi, s, p);          // hi = s >> p
            mpz_tdiv_r_2exp(s, s, p);           // s  = s & M
            mpz_add(s, s, hi);
            if (mpz_cmp(s, M) >= 0) mpz_sub(s, s, M);
            mpz_clear(hi);
        }
    }
    note_res64(s);
    bool prime = (mpz_cmp_ui(s, 0) == 0);
    mpz_clear(M); mpz_clear(s);
    return prime;
}

// Digits of M_p in base 10, without materialising the number in Python.
static std::string mersenne_str(unsigned long p) {
    mpz_t M; mpz_init(M);
    mpz_set_ui(M, 1); mpz_mul_2exp(M, M, p); mpz_sub_ui(M, M, 1);
    char* c = mpz_get_str(nullptr, 10, M);
    std::string out(c);
    void (*freefn)(void*, size_t); mp_get_memory_functions(nullptr, nullptr, &freefn); freefn(c, out.size() + 1);
    mpz_clear(M);
    return out;
}

// Sieve of Eratosthenes: primes in [lo, hi]. The 2018 version was an O(n^2)-ish vector-erase loop.
static std::vector<unsigned long> primes_in_range(unsigned long lo, unsigned long hi) {
    std::vector<unsigned long> out;
    if (hi < 2) return out;
    std::vector<bool> comp(hi + 1, false);
    for (unsigned long i = 2; i * i <= hi; ++i)
        if (!comp[i]) for (unsigned long j = i * i; j <= hi; j += i) comp[j] = true;
    for (unsigned long i = (lo < 2 ? 2 : lo); i <= hi; ++i) if (!comp[i]) out.push_back(i);
    return out;
}


// ---------------------------------------------------------------------------------------------
// Level 2: one Lucas-Lehmer test on several threads. GMP's multiplication is single-threaded, so
// split the squaring by hand: with s = a*2^k + b (k = half the bit length),
//     s^2 = a^2 * 2^2k + 2ab * 2^k + b^2,   and Karatsuba gives 2ab = (a+b)^2 - a^2 - b^2,
// i.e. THREE half-size SQUARINGS that are independent -> three threads, then a cheap combine
// and the same shift-and-add reduction. With depth 2 each half-size squaring is split again:
// nine quarter-size squarings on nine threads. Persistent workers with a per-iteration barrier,
// so the (p-2) iterations do not pay thread creation.
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>

namespace {
struct Pool {   // fixed set of workers; a spinning barrier per "generation" (wake-up ~1 us, not ~50 us)
    std::vector<std::thread> th; std::vector<std::function<void()>> jobs;
    std::atomic<unsigned long> gen{0}; std::atomic<int> pending{0}; std::atomic<bool> stop{false};
    explicit Pool(int n) : jobs(n) {
        for (int i = 0; i < n; ++i) th.emplace_back([this, i] {
            unsigned long seen = 0;
            for (;;) {
                unsigned spins = 0;
                while (gen.load(std::memory_order_acquire) == seen && !stop.load(std::memory_order_relaxed)) {
                    if (++spins < 4096) __builtin_ia32_pause(); else { std::this_thread::yield(); spins = 0; }
                }
                if (stop.load(std::memory_order_relaxed)) return;
                seen = gen.load(std::memory_order_acquire);
                jobs[i]();
                pending.fetch_sub(1, std::memory_order_acq_rel);
            }
        });
    }
    void start(std::vector<std::function<void()>> j) {
        jobs = std::move(j); pending.store((int)th.size(), std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
    }
    void wait() { unsigned spins = 0; while (pending.load(std::memory_order_acquire) != 0) { if (++spins < 4096) __builtin_ia32_pause(); else { std::this_thread::yield(); spins = 0; } } }
    void run(std::vector<std::function<void()>> j) { start(std::move(j)); wait(); }
    ~Pool() { stop.store(true); for (auto& t : th) t.join(); }
};

// r = x^2 using Karatsuba on halves; squarings of the halves delegated to `sq` (which may itself split).
static void square_split(mpz_t r, const mpz_t x, mpz_t a, mpz_t b, mpz_t t1, mpz_t t2, mpz_t t3,
                         const std::function<void(mpz_t, const mpz_t, int)>& sq, Pool* pool) {
    size_t L = mpz_sizeinbase(x, 2); size_t k = L / 2;
    if (L < 4096) { mpz_mul(r, x, x); return; }        // too small to be worth splitting
    mpz_tdiv_q_2exp(a, x, k);                            // a = x >> k
    mpz_tdiv_r_2exp(b, x, k);                            // b = x & (2^k-1)
    mpz_add(t3, a, b);                                   // t3 = a + b (squared below)
    if (pool) {
        pool->start({ [&] { sq(t1, a, 1); }, [&] { sq(t2, b, 2); } });   // two on the workers ...
        sq(t3, t3, 0);                                                    // ... one on THIS thread, concurrently
        pool->wait();
    } else { sq(t1, a, 0); sq(t2, b, 0); sq(t3, t3, 0); }
    mpz_sub(t3, t3, t1); mpz_sub(t3, t3, t2);            // t3 = 2ab
    mpz_mul_2exp(r, t1, 2 * k);                          // r = a^2 << 2k
    mpz_mul_2exp(t3, t3, k); mpz_add(r, r, t3);          // + 2ab << k
    mpz_add(r, r, t2);                                   // + b^2
}
} // namespace

// depth 1 -> 3 threads (2 workers + caller), depth 2 -> 9 threads (each half split again on its own pool)
static bool lucas_lehmer_split(unsigned long p, int depth) {
    if (depth <= 0) return lucas_lehmer(p);
    if (p < 2) return false; if (p == 2) return true; if (p % 2 == 0) return false;
    mpz_t M, s, sq_, a, b, t1, t2, t3; mpz_init(M); mpz_init(s); mpz_init(sq_);
    mpz_init(a); mpz_init(b); mpz_init(t1); mpz_init(t2); mpz_init(t3);
    mpz_set_ui(M, 1); mpz_mul_2exp(M, M, p); mpz_sub_ui(M, M, 1);
    mpz_set_ui(s, 4);
    {
        py::gil_scoped_release release;
        Pool top(2);
        // depth 2: each of the three lanes gets its own scratch + 2-worker pool
        struct Lane { mpz_t a, b, t1, t2, t3; Pool* pool; };
        std::vector<Lane> lanes(3); std::vector<Pool*> sub;
        for (auto& ln : lanes) { mpz_init(ln.a); mpz_init(ln.b); mpz_init(ln.t1); mpz_init(ln.t2); mpz_init(ln.t3);
                                 ln.pool = depth >= 2 ? new Pool(2) : nullptr; if (ln.pool) sub.push_back(ln.pool); }
        std::function<void(mpz_t, const mpz_t, int)> plain = [](mpz_t r, const mpz_t x, int) { mpz_mul(r, x, x); };
        std::function<void(mpz_t, const mpz_t, int)> lane_sq = [&](mpz_t r, const mpz_t x, int lane) {
            Lane& ln = lanes[lane]; square_split(r, x, ln.a, ln.b, ln.t1, ln.t2, ln.t3, plain, ln.pool); };
        const auto& sq = depth >= 2 ? lane_sq : plain;
        for (unsigned long i = 0; i + 2 < p; ++i) {
            square_split(sq_, s, a, b, t1, t2, t3, sq, &top);   // sq_ = s^2
            mpz_sub_ui(sq_, sq_, 2);
            mpz_tdiv_q_2exp(t1, sq_, p); mpz_tdiv_r_2exp(s, sq_, p); mpz_add(s, s, t1);   // mod 2^p-1
            if (mpz_cmp(s, M) >= 0) mpz_sub(s, s, M);
        }
        for (auto& ln : lanes) { mpz_clear(ln.a); mpz_clear(ln.b); mpz_clear(ln.t1); mpz_clear(ln.t2); mpz_clear(ln.t3); }
        for (auto* q : sub) delete q;
    }
    note_res64(s);
    bool prime = (mpz_cmp_ui(s, 0) == 0);
    mpz_clear(M); mpz_clear(s); mpz_clear(sq_); mpz_clear(a); mpz_clear(b); mpz_clear(t1); mpz_clear(t2); mpz_clear(t3);
    return prime;
}

// microbenchmark: seconds per squaring of a `bits`-bit number, plain vs split (depth 1/2)
static double bench_square(unsigned long bits, int iters, int depth) {
    mpz_t x, r, a, b, t1, t2, t3; mpz_init(x); mpz_init(r); mpz_init(a); mpz_init(b); mpz_init(t1); mpz_init(t2); mpz_init(t3);
    gmp_randstate_t st; gmp_randinit_default(st); mpz_urandomb(x, st, bits); gmp_randclear(st);
    double secs = 0;
    {
        py::gil_scoped_release release;
        Pool* top = depth >= 1 ? new Pool(2) : nullptr;
        struct Lane { mpz_t a, b, t1, t2, t3; Pool* pool; }; std::vector<Lane> lanes(3);
        for (auto& ln : lanes) { mpz_init(ln.a); mpz_init(ln.b); mpz_init(ln.t1); mpz_init(ln.t2); mpz_init(ln.t3); ln.pool = depth >= 2 ? new Pool(2) : nullptr; }
        std::function<void(mpz_t, const mpz_t, int)> plain = [](mpz_t r, const mpz_t x, int) { mpz_mul(r, x, x); };
        std::function<void(mpz_t, const mpz_t, int)> lane_sq = [&](mpz_t r, const mpz_t x, int lane) {
            Lane& ln = lanes[lane]; square_split(r, x, ln.a, ln.b, ln.t1, ln.t2, ln.t3, plain, ln.pool); };
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (depth == 0) mpz_mul(r, x, x);
            else square_split(r, x, a, b, t1, t2, t3, depth >= 2 ? lane_sq : plain, top);
        }
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
        for (auto& ln : lanes) { mpz_clear(ln.a); mpz_clear(ln.b); mpz_clear(ln.t1); mpz_clear(ln.t2); mpz_clear(ln.t3); delete ln.pool; }
        delete top;
    }
    mpz_clear(x); mpz_clear(r); mpz_clear(a); mpz_clear(b); mpz_clear(t1); mpz_clear(t2); mpz_clear(t3);
    return secs;
}


// ---------------------------------------------------------------------------------------------
// Floating-point engine: Crandall-Fagin irrational-base discrete weighted transform (IBDWT),
// the algorithm prime95 / gpuowl use. The residue is held as N balanced digits in a variable
// base (b_j = ceil((j+1)p/N) - ceil(jp/N) bits each) and pre-weighted by a_j = 2^(ceil(jp/N) - jp/N),
// so one cyclic convolution of the weighted digits IS the squaring modulo 2^p - 1: no separate
// reduction step. The convolution is a real FFT, a pointwise complex square and an inverse FFT
// (FFTW3, or oneMKL through its FFTW3 wrapper). Every iteration measures the worst rounding
// error; a test whose error ever exceeds 0.35 is reported unreliable and the caller redoes it
// exactly with GMP, so the answer is never trusted to floating point alone. The final residue
// is rebuilt as an exact integer with GMP before the zero test.
#include <fftw3.h>
#ifdef LL_FFT_MKL
#include <mkl_service.h>
#endif
#include <cmath>
#include <map>
#ifndef LL_FFT_BACKEND
#define LL_FFT_BACKEND "fftw3"
#endif

struct FftPlans { fftw_plan fwd, inv; };
static std::mutex g_plan_mx;
static std::map<size_t, FftPlans> g_plans;

static FftPlans plans_for(size_t N) {
    std::lock_guard<std::mutex> lk(g_plan_mx);          // planning is not thread-safe; executing is
    auto it = g_plans.find(N);
    if (it != g_plans.end()) return it->second;
    double* r = fftw_alloc_real(N); fftw_complex* c = fftw_alloc_complex(N / 2 + 1);
    FftPlans P{fftw_plan_dft_r2c_1d((int)N, r, c, FFTW_MEASURE | FFTW_DESTROY_INPUT),
               fftw_plan_dft_c2r_1d((int)N, c, r, FFTW_MEASURE | FFTW_DESTROY_INPUT)};
    fftw_free(r); fftw_free(c);
    g_plans.emplace(N, P);
    return P;
}

// Smallest N in {2^k, 3*2^k, 5*2^k} with p/N <= max_bpw bits per word. At 18 bits/word the worst
// rounding error measured over a full p = 110,503 test is 4e-3 against the 0.35 abort threshold.
static double g_max_bpw = 18.0;                       // tunable: lucaslehmer.set_fft_bits_per_word()
static size_t fft_length(unsigned long p, double max_bpw = 0.0) {
    if (max_bpw <= 0) max_bpw = g_max_bpw;
    size_t best = 0;
    for (size_t m : {1, 3, 5})
        for (size_t N = m * 16; N < ((size_t)1 << 40); N <<= 1)
            if ((double)p / N <= max_bpw) { if (!best || N < best) best = N; break; }
    return best;
}

// 1 = prime, 0 = composite, -1 = rounding error too large (redo exactly). err_out gets the worst error seen.
static int lucas_lehmer_fft_raw(unsigned long p, double* err_out = nullptr) {
    if (p < 2) return 0;
    if (p == 2) return 1;
    if (p % 2 == 0) return 0;
    if (p < 512) return -1;                               // too small for the word layout; GMP is faster anyway
    const size_t N = fft_length(p);
    std::vector<int>      bits(N);
    std::vector<double>   w(N), winv(N);
    for (size_t j = 0; j < N; ++j) {
        unsigned long lo = (j * p + N - 1) / N, hi = ((j + 1) * p + N - 1) / N;    // ceil(jp/N), ceil((j+1)p/N)
        bits[j] = (int)(hi - lo);
        double frac = (double)(lo * N - j * p) / (double)N;                        // in [0,1)
        w[j] = std::exp2(frac);
        winv[j] = 1.0 / (w[j] * (double)N);                                        // also undoes FFTW's scaling
    }
    std::vector<int64_t> x(N, 0);
    x[0] = 4;
    double* y = fftw_alloc_real(N);
    fftw_complex* Y = fftw_alloc_complex(N / 2 + 1);
    FftPlans P = plans_for(N);
    double maxerr = 0.0;
    int verdict = -1;
    {
        py::gil_scoped_release release;
        for (unsigned long it = 0; it + 2 < p; ++it) {
            for (size_t j = 0; j < N; ++j) y[j] = (double)x[j] * w[j];
            fftw_execute_dft_r2c(P.fwd, y, Y);
            for (size_t k = 0; k <= N / 2; ++k) {                                  // Y_k^2
                double re = Y[k][0], im = Y[k][1];
                Y[k][0] = re * re - im * im; Y[k][1] = 2.0 * re * im;
            }
            fftw_execute_dft_c2r(P.inv, Y, y);
            double err = 0.0;
            for (size_t j = 0; j < N; ++j) {                                       // unweight, round, error
                double v = y[j] * winv[j], r = std::nearbyint(v);
                double e = std::fabs(v - r); if (e > err) err = e;
                y[j] = r;
            }
            if (err > maxerr) maxerr = err;
            if (err > 0.35) { verdict = -1; goto done; }
            int64_t carry = -2;                                                    // the "- 2" of the recurrence
            for (size_t j = 0; j < N; ++j) {                                       // balanced carry propagation
                int64_t v = (int64_t)y[j] + carry;
                int b = bits[j];
                carry = (v + ((int64_t)1 << (b - 1))) >> b;                        // floor((v + base/2) / base)
                x[j] = v - (carry << b);
            }
            while (carry) {                                                        // wrap: 2^p == 1 (mod M_p)
                for (size_t j = 0; j < N && carry; ++j) {
                    int64_t v = x[j] + carry; int b = bits[j];
                    carry = (v + ((int64_t)1 << (b - 1))) >> b;
                    x[j] = v - (carry << b);
                }
            }
        }
        {   // exact final residue with GMP: s = sum x_j 2^(ceil(jp/N)) mod 2^p - 1
            mpz_t M, s, t; mpz_init(M); mpz_init(s); mpz_init(t);
            mpz_set_ui(M, 1); mpz_mul_2exp(M, M, p); mpz_sub_ui(M, M, 1);
            for (size_t j = 0; j < N; ++j) {
                if (!x[j]) continue;
                mpz_set_si(t, (long)x[j]); mpz_mul_2exp(t, t, (j * p + N - 1) / N); mpz_add(s, s, t);
            }
            mpz_mod(s, s, M);
            note_res64(s);
            verdict = (mpz_cmp_ui(s, 0) == 0) ? 1 : 0;
            mpz_clear(M); mpz_clear(s); mpz_clear(t);
        }
    }
done:
    fftw_free(y); fftw_free(Y);
    if (err_out) *err_out = maxerr;
    return verdict;
}

// Exact answer: FFT engine, redone with GMP if the rounding error ever got too large.
static bool lucas_lehmer_fft(unsigned long p) {
    int v = lucas_lehmer_fft_raw(p);
    return v < 0 ? lucas_lehmer(p) : v == 1;
}

// (N, bits per word, worst rounding error over the whole test) -- for tuning and the README.
static std::tuple<size_t, double, double> fft_info(unsigned long p) {
    size_t N = fft_length(p); double err = 0.0;
    lucas_lehmer_fft_raw(p, &err);
    return {N, (double)p / N, err};
}

PYBIND11_MODULE(lucaslehmer, m, py::mod_gil_not_used()) {
#ifdef LL_FFT_MKL
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);   // one FFT per Python thread; no OpenMP runtime needed
#endif
    m.doc() = "Lucas-Lehmer Mersenne prime test on GMP (releases the GIL; free-threading safe)";
    m.def("lucas_lehmer", &lucas_lehmer, py::arg("p"),
          "True iff 2^p - 1 is prime (Lucas-Lehmer). p should be prime.");
    m.def("lucas_lehmer_split", &lucas_lehmer_split, py::arg("p"), py::arg("depth") = 1,
          "Lucas-Lehmer with the squaring split across threads: depth 1 = 3 threads, depth 2 = 9 threads.");
    m.def("lucas_lehmer_fft", &lucas_lehmer_fft, py::arg("p"),
          "Lucas-Lehmer on the floating-point IBDWT engine (FFT squaring, exact: falls back to GMP on rounding trouble).");
    m.def("lucas_lehmer_res", [](unsigned long p) { bool r = lucas_lehmer(p); return std::make_pair(r, g_last_res64); }, py::arg("p"),
          "(is_prime, res64) on the GMP engine: res64 = low 64 bits of the final residue, GIMPS style.");
    m.def("lucas_lehmer_split_res", [](unsigned long p, int depth) { bool r = lucas_lehmer_split(p, depth); return std::make_pair(r, g_last_res64); },
          py::arg("p"), py::arg("depth") = 1, "(is_prime, res64) on the split GMP engine.");
    m.def("lucas_lehmer_fft_res", [](unsigned long p) { bool r = lucas_lehmer_fft(p); return std::make_pair(r, g_last_res64); }, py::arg("p"),
          "(is_prime, res64) on the FFT engine (exact; GMP redo on rounding trouble).");
    m.def("fft_info", &fft_info, py::arg("p"), "(FFT length, bits per word, worst rounding error) of an FFT-engine test of p.");
    m.attr("fft_backend") = std::string(LL_FFT_BACKEND);
    m.def("set_fft_bits_per_word", [](double b) { g_max_bpw = b; }, py::arg("bits"),
          "Max bits per FFT word (default 18; prime95 runs ~18-19.5). Higher = shorter FFT = faster, more rounding error.");
    m.def("bench_square", &bench_square, py::arg("bits"), py::arg("iters") = 20, py::arg("depth") = 0,
          "seconds per squaring of a random bits-bit number: depth 0 plain GMP, 1 = 3-thread split, 2 = 9-thread split");
    m.def("primes", &primes_in_range, py::arg("lo"), py::arg("hi"),
          "List of primes in [lo, hi] (sieve of Eratosthenes).");
    m.def("mersenne_digits", [](unsigned long p) { return mersenne_str(p).size(); }, py::arg("p"),
          "Number of decimal digits of 2^p - 1.");
    m.def("mersenne_str", &mersenne_str, py::arg("p"), "Decimal string of 2^p - 1.");
    m.attr("gmp_version") = std::string(gmp_version);
    // 2018-compatible class surface, for anyone who imported LucasLehmer().
    struct LucasLehmer {};
    py::class_<LucasLehmer>(m, "LucasLehmer")
        .def(py::init<>())
        .def("sa_lucaslehmer", [](LucasLehmer&, unsigned long p, py::object) { return lucas_lehmer(p); }, py::arg("N_"), py::arg("S_") = py::none())
        .def("sa_getListOfPrimes", [](LucasLehmer&, py::list primes, unsigned long lo, unsigned long hi) {
            for (auto v : primes_in_range(lo, hi)) primes.append(v);
            return true;
        });
}
