#include "lucaslehmerpy.h"

#include "string.h"

#include <execinfo.h>
#include <cxxabi.h>
#include <vector>
#include <algorithm>

PyObject* LucasLehmer::_py_tmp = PyLong_FromLong( 0 );
PyObject* LucasLehmer::_py_maxx = PyLong_FromUnsignedLongLong( ((unsigned long long)(0xffffffffffffffffLL)));
PyObject* LucasLehmer::_py_zero = PyLong_FromLong( 0 );
PyObject* LucasLehmer::_py_one = PyLong_FromLong( 1 );
PyObject* LucasLehmer::_py_two = PyLong_FromLong( 2 );
mpz_t LucasLehmer::_mpz_zero;
mpz_t LucasLehmer::_mpz_one;
mpz_t LucasLehmer::_mpz_two;
mpz_t LucasLehmer::_mpz_four;

LucasLehmer::LucasLehmer() : timesum(0), timestd(0)
{
    mpz_init(_mpz_zero);
    mpz_set_ui(_mpz_zero, 0);

    mpz_init(_mpz_one);
    mpz_set_ui(_mpz_one, 1);

    mpz_init(_mpz_two);
    mpz_set_ui(_mpz_two, 2);

    mpz_init(_mpz_four);
    mpz_set_ui(_mpz_four, 4);

}

void LucasLehmer::mpz_convFromPy(mpz_t mv, PyObject* pv)
{
    PyObject* nv = PyObject_Str(pv);

    Py_ssize_t len = PyObject_Length(nv);
    char *buffer = PyString_AsString(nv);

    mpz_set_str(mv, buffer, 10);
}

PyObject* LucasLehmer::mpz_convToPy(mpz_t &mv)
{
    char *str = NULL;
    str = mpz_get_str(str, 10, mv);
    char *pend = NULL;
    PyObject* pobj = PyLong_FromString(str, &pend, 10);

    return pobj;
}

PyObject* LucasLehmer::sa_getListOfPrimes(PyObject* primes, PyObject* N_low, PyObject* N_high)
{
    unsigned long range_high = PyLong_AsLong(N_high);
    unsigned long range_low = PyLong_AsLong(N_low);

    primeVec_t pvec;
    pvec.push_back(2);

    primeVec_t rvec(range_high);

    std::fill(rvec.begin(), rvec.end(), 0);
    
    for ( size_t i = 3; i < range_high + 1; i++)
    {
        if (!(i % 2))
            continue;
	pvec.push_back(i);
    }

    unsigned long candidate = 0;

    size_t count = 0;

    while (pvec.size())
    {
	candidate = pvec[0];
        rvec[count++] = candidate;

	for (unsigned long i = candidate; i < range_high + 1; i += candidate)
	{
            auto it = std::lower_bound(pvec.begin(), pvec.end(), i);
            if (it != pvec.end() && *it == i)
                pvec.erase(it);
	}
    }

    PyObject* tmp;

    for (size_t i = 0; i < rvec.size(); i++)
    {
        unsigned long v = rvec[i];
        if (v < range_low)
            continue;
        if (!v)
            break;
	tmp = PyLong_FromUnsignedLong(v);
	if (PyList_Append(primes, tmp))
	{
	    std::cerr << "Exception: getListOfPrimes: PyList_Append failed" << std::endl;
	    throw;
	}
    }
    Py_RETURN_TRUE;
}

PyObject* LucasLehmer::sa_lucaslehmer(PyObject* N_, PyObject* S_)
{
    if (!PyObject_Compare(PyNumber_Remainder(N_, _py_two), _py_zero)) 
	return _py_two;
    
    mpz_t n, s;
    mpz_init(n);
    mpz_init(s);
    mpz_set(n, _mpz_zero);
    mpz_convFromPy(n, N_);
    mpz_convFromPy(s, S_);

    if (!mpz_cmp(n, _mpz_zero))
    {
        std::cerr << "Error: conversion from PyObject to mpz_t failed" << std::endl;
        throw;
    }

    mpz_t M, i, tmp1, tmp2;

    mpz_init(M);
    mpz_init(i);

    mpz_init(tmp1);
    mpz_init(tmp2);

    mpz_set(i, _mpz_zero);
    mpz_set(s, _mpz_four);

    mpz_sub(tmp1, n, _mpz_two);

    mpz_mul_2exp(tmp2, _mpz_one, PyLong_AsLong(N_));
    mpz_sub(M, tmp2, _mpz_one);

    do
    {
        mpz_mul(s, s, s);
        mpz_sub(s, s, _mpz_two);
        mpz_mod(s, s, M);
        mpz_add(i, i, _mpz_one);
    }
    while (mpz_cmp(i, tmp1) < 0);

    if (mpz_cmp(s, _mpz_zero) != 0)
    {
        Py_RETURN_FALSE;
    }

    Py_RETURN_TRUE;
}

PyObject* LucasLehmer::sa_gcd(PyObject* a, PyObject* b)
{
    PyObject* c = PyLong_FromLong( 0 );
    if (PyObject_Compare(a, _py_zero))
    {
	while (0 < PyObject_Compare(a, _py_zero))
	{
	    c = a; 
	    a = PyNumber_Remainder(b, a);  
	    b = c;
	}
    }

    b = PyNumber_Add(b, _py_zero);

    return b;
}

PyObject* LucasLehmer::sa_pow2mpcm(PyObject* y, PyObject* n, PyObject* c)
{
    tmp(y);
    tmp(PyNumber_Multiply(tmp(), tmp()));
    tmp(PyNumber_Remainder(tmp(), n));
    tmp(PyNumber_Add(tmp(), c));
    tmp(PyNumber_Remainder(tmp(), n));
    
    return tmp();
}
