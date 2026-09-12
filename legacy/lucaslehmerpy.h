#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iterator>
#include <sys/time.h>

#include <Python.h>

#include <boost/multiprecision/gmp.hpp>

using namespace boost::multiprecision;

class LucasLehmer
{
public:
    LucasLehmer();
    ~LucasLehmer() {}
    PyObject* sa_gcd(PyObject* a, PyObject* b);
    PyObject* sa_pow2mpcm(PyObject* y, PyObject* n, PyObject* c);
    PyObject* sa_lucaslehmer(PyObject* N_, PyObject* S_);
    PyObject* sa_getListOfPrimes(PyObject* primes, PyObject* N_low, PyObject* N_high);
    PyObject* tmp() { return _py_tmp; }
    void tmp(PyObject* t) { _py_tmp = t; }
private:
    typedef std::vector<unsigned long> primeVec_t;
    typedef primeVec_t::iterator primeVecIter_t;
    
    typedef struct timeStruct_t
    {
	timeStruct_t() : numsamps(0), average(0), stddev(0) {}
	long long numsamps;
	double average;
	double stddev;
    } tstruct;
    void mpz_convFromPy(mpz_t mv, PyObject* pv);
    PyObject* mpz_convToPy(mpz_t &mv);

    unsigned long getElapsed(const bool do_it = true)
    {
	if (do_it)
	{
	    clock_gettime(CLOCK_REALTIME,&t2);

	    unsigned long secd = t2.tv_sec - t1.tv_sec;
	    if (secd)
	    {
		secd *= 1e9;
	    }
	    return secd += t2.tv_nsec - t1.tv_nsec;
	}

	clock_gettime(CLOCK_REALTIME,&t1);

	return 0;
    }

    unsigned long long int rdtsc(void)
    {
	unsigned long long int x;
	unsigned a, d;
	__asm__ volatile("rdtsc" : "=a" (a), "=d" (d));
	
	return ((unsigned long long)a) | (((unsigned long long)d) << 32);;
    }

    static PyObject* _py_tmp;
    static PyObject* _py_maxx;
    static PyObject* _py_zero;
    static PyObject* _py_one;
    static PyObject* _py_two;
    static mpz_t _mpz_zero;
    static mpz_t _mpz_one;
    static mpz_t _mpz_two;
    static mpz_t _mpz_four;
    timespec t1,t2;
    double timesum;
    double timestd;
};

