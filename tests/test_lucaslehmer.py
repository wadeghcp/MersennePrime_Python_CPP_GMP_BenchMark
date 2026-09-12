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
