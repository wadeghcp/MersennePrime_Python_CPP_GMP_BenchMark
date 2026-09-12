#!/usr/bin/python

import sys
import pybindgen

mod = pybindgen.Module('lucaslehmer')

mod.add_include('"lucaslehmerpy.h"')

lucaslehmerclass = mod.add_class('LucasLehmer')

lucaslehmerclass.add_constructor([])

lucaslehmerclass.add_method('sa_gcd', 
                      pybindgen.retval('PyObject*', caller_owns_return=True, is_const=False), 
                      [pybindgen.param('PyObject*','a',transfer_ownership=False),
                       pybindgen.param('PyObject*','b',transfer_ownership=False)])

lucaslehmerclass.add_method('sa_pow2mpcm', 
                      pybindgen.retval('PyObject*', caller_owns_return=True, is_const=False), 
                      [pybindgen.param('PyObject*','y',transfer_ownership=False),
                       pybindgen.param('PyObject*','n',transfer_ownership=False),
                       pybindgen.param('PyObject*','c',transfer_ownership=False)])

lucaslehmerclass.add_method('sa_getListOfPrimes', 
                      pybindgen.retval('PyObject*', caller_owns_return=True, is_const=False), 
                      [pybindgen.param('PyObject*','primes',transfer_ownership=False),
                       pybindgen.param('PyObject*','N_low',transfer_ownership=False),
                       pybindgen.param('PyObject*','N_high',transfer_ownership=False)])

lucaslehmerclass.add_method('sa_lucaslehmer', 
                      pybindgen.retval('PyObject*', caller_owns_return=True, is_const=False), 
                      [pybindgen.param('PyObject*','N_',transfer_ownership=False),
                       pybindgen.param('PyObject*','S_',transfer_ownership=False),
                      ])

lucaslehmerclass.add_method('tmp',
                      pybindgen.retval('PyObject*', caller_owns_return=True, is_const=False), 
                      [])

lucaslehmerclass.add_method('tmp',
                      None,
                      [pybindgen.param('PyObject*','t',transfer_ownership=False)])

mod.generate(sys.stdout)
