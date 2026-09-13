// Lucas-Lehmer on the GPU with an exact integer transform.
//
// One thread block owns one exponent p and runs the whole test on-device: p-2 squarings modulo
// 2^p - 1, each an integer IBDWT (Crandall-Fagin's irrational-base discrete weighted transform,
// done in a finite field instead of floating point):
//
//   * the residue is N balanced digits in a variable base, b_j = ceil((j+1)p/N) - ceil(jp/N) bits;
//   * the field is GF(q), q = 2^64 - 2^32 + 1 ("Goldilocks"): 2^32 | q-1 so power-of-two NTTs
//     exist, and 2 has order 192, so an N-th root of 2 exists for every N | 2^26. That root, w,
//     gives integer weights a_j = w^(ceil(jp/N)*N - jp)  (= "2^(ceil(jp/N) - jp/N)"), which is what
//     makes one cyclic convolution equal the squaring modulo 2^p - 1 with no reduction step;
//   * forward NTT (DIF) -> pointwise square -> inverse NTT (DIT) -> unweight -> balanced carry.
//     No bit reversal: the pointwise square does not care about order;
//   * every digit of the convolution is an integer below N * 2^(2B-1) (B = bits per word), and
//     N is chosen so that stays under q/2, so the field value IS the integer: no rounding, no
//     error bound to monitor. The final residue goes back to the host and is checked with GMP.
//
// Limits of this first version: everything lives in shared memory, so N <= 8192 (64 KB of
// digits), i.e. p <= 204,800 on sm_86. Larger p needs a multi-block or global-memory NTT.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <gmp.h>
#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstdio>

namespace py = pybind11;
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned __int128 u128;

static const u64 Q = 0xFFFFFFFF00000001ull;   // 2^64 - 2^32 + 1
static const u64 EPS = 0xFFFFFFFFull;          // 2^64 mod Q == 2^32 - 1

// ---- field arithmetic, host side (for tables) ------------------------------------------------
static u64 h_mul(u64 a, u64 b) { return (u64)(((u128)a * b) % Q); }
static u64 h_pow(u64 a, u64 e) { u64 r = 1; while (e) { if (e & 1) r = h_mul(r, a); a = h_mul(a, a); e >>= 1; } return r; }
static u64 h_inv(u64 a) { return h_pow(a, Q - 2); }

// ---- field arithmetic, device side --------------------------------------------------------------
__device__ __forceinline__ u64 f_add(u64 a, u64 b) {
    u64 s = a + b; if (s < a) s += EPS; if (s >= Q) s -= Q; return s;
}
__device__ __forceinline__ u64 f_sub(u64 a, u64 b) {
    u64 d = a - b; if (a < b) d -= EPS; return d;
}
__device__ __forceinline__ u64 f_mul(u64 a, u64 b) {
    u64 lo = a * b, hi = __umul64hi(a, b);
    u64 hh = hi >> 32, hl = hi & EPS;
    u64 t0 = lo - hh; if (lo < hh) t0 -= EPS;         // lo - hh   (2^96 == -1)
    u64 t1 = (hl << 32) - hl;                          // hl * (2^32 - 1)   (2^64 == 2^32 - 1)
    u64 r = t0 + t1; if (r < t0) r += EPS;
    if (r >= Q) r -= Q;
    return r;
}

struct TestArgs {
    u64 p;            // exponent
    u64 N;            // transform length (power of two)
    u64 log2N;
    const u64* wgt;   // [N] forward weights a_j
    const u64* wgt_inv;  // [N] a_j^-1 * N^-1
};

// Shared layout: s[N] digits/field values, tw[N/2] forward twiddles w_N^k, cout[T] carries.
extern __shared__ u64 smem[];

__global__ void __launch_bounds__(1024)
ll_kernel(const TestArgs* args, const u64* twiddles /*[N/2] for this launch's N*/,
          i64* out_digits /*[grid][N]*/, i64* out_ns /*[grid]*/) {
    const TestArgs A = args[blockIdx.x];
    const u64 p = A.p, N = A.N; const int T = blockDim.x, t = threadIdx.x;
    u64* s  = smem;
    u64* tw = smem + N;
    i64* cout = (i64*)(smem + N + N / 2);
    const u64 half = N / 2;

    u64 t_start; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
    for (u64 k = t; k < half; k += T) tw[k] = twiddles[k];
    for (u64 j = t; j < N; j += T) s[j] = 0;
    if (t == 0) s[0] = 4;                                   // s_0 = 4
    __syncthreads();

    const u64 E = N / T;                                    // digits per thread (contiguous chunk)
    for (u64 it = 0; it + 2 < p; ++it) {
        // 1. weight: y_j = x_j * a_j  (x_j is a signed digit stored as i64 bits)
        for (u64 j = t; j < N; j += T) {
            i64 x = (i64)s[j];
            u64 f = x < 0 ? Q - (u64)(-x) : (u64)x;
            s[j] = f_mul(f, __ldg(A.wgt + j));
        }
        __syncthreads();
        // 2. forward NTT, decimation in frequency (natural order in, bit-reversed out)
        for (u64 len = half; len >= 1; len >>= 1) {
            u64 tstep = half / len;
            for (u64 i = t; i < half; i += T) {
                u64 grp = i / len, j = i - grp * len;
                u64 a = grp * 2 * len + j, b = a + len;
                u64 u = s[a], v = s[b];
                s[a] = f_add(u, v);
                s[b] = f_mul(f_sub(u, v), tw[j * tstep]);
            }
            __syncthreads();
        }
        // 3. pointwise square
        for (u64 j = t; j < N; j += T) { u64 v = s[j]; s[j] = f_mul(v, v); }
        __syncthreads();
        // 4. inverse NTT, decimation in time (bit-reversed in, natural out), twiddle w^-k = -w^(N/2-k)
        for (u64 len = 1; len < N; len <<= 1) {
            u64 tstep = half / len;
            for (u64 i = t; i < half; i += T) {
                u64 grp = i / len, j = i - grp * len;
                u64 a = grp * 2 * len + j, b = a + len;
                u64 k = j * tstep;
                u64 wk = k ? (Q - tw[half - k]) : 1ull;     // w^-k
                u64 u = s[a], v = f_mul(s[b], wk);
                s[a] = f_add(u, v);
                s[b] = f_sub(u, v);
            }
            __syncthreads();
        }
        // 5. unweight (includes 1/N), reinterpret as a signed integer, store as i64 bits
        for (u64 j = t; j < N; j += T) {
            u64 v = f_mul(s[j], __ldg(A.wgt_inv + j));
            i64 sv = (v > (Q >> 1)) ? (i64)(v - Q) : (i64)v;
            s[j] = (u64)sv;
        }
        __syncthreads();
        // 6. "- 2" and balanced carry, base 2^b_j per word, wrap of the top carry into word 0
        //    pass 1: each thread carries its own chunk; pass 2+: feed chunk carries forward until none
        {
            i64 c = (t == 0) ? -2 : 0;
            u64 j0 = t * E;
            for (u64 j = j0; j < j0 + E; ++j) {
                u64 lo = (j * p + N - 1) / N, hi = ((j + 1) * p + N - 1) / N;
                int b = (int)(hi - lo);
                i64 v = (i64)s[j] + c;
                c = (v + ((i64)1 << (b - 1))) >> b;
                s[j] = (u64)(v - (c << b));
            }
            cout[t] = c;
            __syncthreads();
            while (__syncthreads_or(cout[t] != 0)) {
                i64 cin = cout[(t + T - 1) % T];             // thread 0 takes the wrap-around carry
                __syncthreads();
                for (u64 j = j0; j < j0 + E && cin; ++j) {
                    u64 lo = (j * p + N - 1) / N, hi = ((j + 1) * p + N - 1) / N;
                    int b = (int)(hi - lo);
                    i64 v = (i64)s[j] + cin;
                    cin = (v + ((i64)1 << (b - 1))) >> b;
                    s[j] = (u64)(v - (cin << b));
                }
                cout[t] = cin;
                __syncthreads();
            }
        }
    }
    for (u64 j = t; j < N; j += T) out_digits[(u64)blockIdx.x * N + j] = (i64)s[j];
    u64 t_end; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
    if (t == 0) out_ns[blockIdx.x] = (i64)(t_end - t_start);
}

// ---- host side ---------------------------------------------------------------------------------
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " #x); } while (0)

static int max_bits_per_word(u64 N) {                       // N * 2^(2B-1) < 2^62  ->  B <= (63 - log2 N) / 2
    int l = 0; while ((1ull << l) < N) ++l;
    return (63 - l) / 2;
}
static u64 ntt_length(u64 p) {
    for (u64 N = 64; N <= 8192; N <<= 1)
        if ((p + N - 1) / N <= (u64)max_bits_per_word(N)) return N;
    return 0;                                               // too big for the single-block kernel
}

struct Tables { u64 N; std::vector<u64> tw; u64 root2; u64 invN; };
static Tables make_tables(u64 N) {
    const u64 g = 7;
    u64 omega = h_pow(g, (Q - 1) / N);                      // primitive N-th root of unity
    Tables T; T.N = N; T.tw.resize(N / 2);
    u64 x = 1; for (u64 k = 0; k < N / 2; ++k) { T.tw[k] = x; x = h_mul(x, omega); }
    // N-th root of 2: om = primitive (192 N)-th root; 2 = (om^N)^s for some s < 192; w = om^s
    u64 om = h_pow(g, (Q - 1) / (192 * N)), r192 = h_pow(om, N), pw = 1; u64 s = 0;
    for (; s < 192; ++s) { if (pw == 2) break; pw = h_mul(pw, r192); }
    if (s == 192) throw std::runtime_error("no N-th root of 2");
    T.root2 = h_pow(om, s);
    T.invN = h_inv(N);
    return T;
}

// Run a batch of exponents that all share one transform length. Returns (p, is_prime, seconds) per test.
static std::vector<std::tuple<u64, bool, double, u64>> run_batch(const std::vector<u64>& ps, u64 N) {
    const size_t n = ps.size();
    Tables T = make_tables(N);
    // per-test weights
    std::vector<u64> hw(n * N), hwi(n * N);
    for (size_t i = 0; i < n; ++i) {
        u64 p = ps[i];
        for (u64 j = 0; j < N; ++j) {
            u64 c = (j * p + N - 1) / N;                    // ceil(jp/N)
            u64 e = c * N - j * p;                          // in [0, N)
            u64 a = h_pow(T.root2, e);
            hw[i * N + j] = a;
            hwi[i * N + j] = h_mul(h_inv(a), T.invN);
        }
    }
    u64 *d_tw, *d_w, *d_wi; TestArgs* d_args; i64 *d_out, *d_ns;
    CK(cudaMalloc(&d_tw, N / 2 * 8)); CK(cudaMalloc(&d_w, n * N * 8)); CK(cudaMalloc(&d_wi, n * N * 8));
    CK(cudaMalloc(&d_args, n * sizeof(TestArgs))); CK(cudaMalloc(&d_out, n * N * 8)); CK(cudaMalloc(&d_ns, n * 8));
    CK(cudaMemcpy(d_tw, T.tw.data(), N / 2 * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, hw.data(), n * N * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wi, hwi.data(), n * N * 8, cudaMemcpyHostToDevice));
    std::vector<TestArgs> args(n);
    int l = 0; while ((1ull << l) < N) ++l;
    for (size_t i = 0; i < n; ++i) args[i] = TestArgs{ps[i], N, (u64)l, d_w + i * N, d_wi + i * N};
    CK(cudaMemcpy(d_args, args.data(), n * sizeof(TestArgs), cudaMemcpyHostToDevice));

    int threads = (int)std::min<u64>(1024, std::max<u64>(32, N / 8));
    size_t smem = N * 8 + (N / 2) * 8 + threads * 8;
    CK(cudaFuncSetAttribute(ll_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    {
        py::gil_scoped_release release;
        ll_kernel<<<(unsigned)n, threads, smem>>>(d_args, d_tw, d_out, d_ns);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }
    std::vector<i64> out(n * N), ns(n);
    CK(cudaMemcpy(out.data(), d_out, n * N * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ns.data(), d_ns, n * 8, cudaMemcpyDeviceToHost));
    cudaFree(d_tw); cudaFree(d_w); cudaFree(d_wi); cudaFree(d_args); cudaFree(d_out); cudaFree(d_ns);

    // exact verdict: rebuild the residue with GMP and reduce mod 2^p - 1
    std::vector<std::tuple<u64, bool, double, u64>> res(n);
    mpz_t M, s, t; mpz_init(M); mpz_init(s); mpz_init(t);
    for (size_t i = 0; i < n; ++i) {
        u64 p = ps[i];
        mpz_set_ui(M, 1); mpz_mul_2exp(M, M, p); mpz_sub_ui(M, M, 1);
        mpz_set_ui(s, 0);
        for (u64 j = 0; j < N; ++j) {
            i64 x = out[i * N + j]; if (!x) continue;
            mpz_set_si(t, (long)x); mpz_mul_2exp(t, t, (j * p + N - 1) / N); mpz_add(s, s, t);
        }
        mpz_mod(s, s, M);
        res[i] = {p, mpz_cmp_ui(s, 0) == 0, ns[i] / 1e9, (u64)mpz_get_ui(s)};   // res64 = low limb
    }
    mpz_clear(M); mpz_clear(s); mpz_clear(t);
    return res;
}

// Public entry: any list of exponents; grouped by transform length, each group one launch.
// Exponents too small (< 64) or too large for the kernel are returned with is_prime = false and
// seconds < 0 so the caller can redo them on the CPU.
static std::vector<std::tuple<u64, bool, double, u64>> lucas_lehmer_batch(std::vector<u64> ps) {
    std::map<u64, std::vector<u64>> groups; std::vector<std::tuple<u64, bool, double, u64>> res;
    for (u64 p : ps) {
        u64 N = (p >= 64) ? ntt_length(p) : 0;
        if (!N) res.emplace_back(p, false, -1.0, 0ull); else groups[N].push_back(p);
    }
    for (auto& g : groups) { auto r = run_batch(g.second, g.first); res.insert(res.end(), r.begin(), r.end()); }
    return res;
}

static std::tuple<std::string, int, size_t> device_info() {
    int dev = 0; cudaDeviceProp pr; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&pr, dev));
    return {pr.name, pr.multiProcessorCount, pr.totalGlobalMem};
}

PYBIND11_MODULE(lucaslehmer_cuda, m, py::mod_gil_not_used()) {
    m.doc() = "Lucas-Lehmer on CUDA: exact integer IBDWT in GF(2^64 - 2^32 + 1), one thread block per exponent";
    m.def("lucas_lehmer_batch", &lucas_lehmer_batch, py::arg("exponents"),
          "[(p, is_prime, seconds, res64)] for a list of exponents; seconds < 0 marks one the kernel cannot take (redo on CPU).");
    m.def("ntt_length", &ntt_length, py::arg("p"), "Transform length the kernel would use for p (0 = too large).");
    m.def("max_exponent", []() { return (u64)8192 * max_bits_per_word(8192); }, "Largest p this kernel handles.");
    m.def("device_info", &device_info, "(name, SM count, global memory bytes)");
}
