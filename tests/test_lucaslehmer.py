import lucaslehmer as ll

KNOWN = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423]

def test_primes_count_10001():
    assert len(ll.primes(2, 10001)) == 1229

def test_primes_bounds():
    assert ll.primes(0, 1) == [] and ll.primes(2, 2) == [2] and ll.primes(10, 20) == [11, 13, 17, 19]

def test_known_mersenne_exponents():
    found = [p for p in ll.primes(2, 4423) if ll.lucas_lehmer(p)]
    assert found == KNOWN

def test_non_mersenne():
    assert not ll.lucas_lehmer(11) and not ll.lucas_lehmer(23) and not ll.lucas_lehmer(4)

def test_digits():
    assert ll.mersenne_digits(127) == 39 and ll.mersenne_str(7) == "127"

def test_compat_class():
    c = ll.LucasLehmer(); out = []
    assert c.sa_lucaslehmer(31, 4) is True and c.sa_getListOfPrimes(out, 2, 10) is True and out == [2, 3, 5, 7]


def test_split_matches_plain():
    for d in (1, 2):
        assert [p for p in ll.primes(2, 4423) if ll.lucas_lehmer_split(p, d)] == KNOWN
        assert not ll.lucas_lehmer_split(4425, d)


def test_fft_engine_matches_known_and_gmp():
    assert [p for p in ll.primes(2, 4423) if ll.lucas_lehmer_fft(p)] == KNOWN
    assert all(ll.lucas_lehmer_fft(p) == ll.lucas_lehmer(p) for p in ll.primes(4425, 6000))
    n, bpw, err = ll.fft_info(9689)
    assert ll.lucas_lehmer_fft(9689) and err < 0.05 and bpw <= 18


def test_cuda_engine_if_built():
    try:
        import lucaslehmer_cuda as lc
    except ImportError:
        import pytest; pytest.skip("lucaslehmer_cuda not built")
    res = {p: r for p, r, s, res64 in lc.lucas_lehmer_batch(ll.primes(2, 4423)) if s >= 0}
    assert sorted(p for p, r in res.items() if r) == [p for p in KNOWN if p >= 64]
    assert all(res[p] == ll.lucas_lehmer(p) for p in res)


def test_res64_agrees_across_engines():
    for p in (4425, 9689, 20011):
        g = ll.lucas_lehmer_res(p); f = ll.lucas_lehmer_fft_res(p); s = ll.lucas_lehmer_split_res(p, 1)
        assert g == f == s
        assert (g[1] == 0) == g[0]           # res64 is zero exactly for a prime
