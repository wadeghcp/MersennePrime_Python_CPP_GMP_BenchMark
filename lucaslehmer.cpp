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
#include <stdexcept>

namespace py = pybind11;

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

PYBIND11_MODULE(lucaslehmer, m, py::mod_gil_not_used()) {
    m.doc() = "Lucas-Lehmer Mersenne prime test on GMP (releases the GIL; free-threading safe)";
    m.def("lucas_lehmer", &lucas_lehmer, py::arg("p"),
          "True iff 2^p - 1 is prime (Lucas-Lehmer). p should be prime.");
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
