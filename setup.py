from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
setup(ext_modules=[Pybind11Extension("lucaslehmer", ["lucaslehmer.cpp"], libraries=["gmp"], cxx_std=17,
                                     extra_compile_args=["-O3", "-march=native"])],
      cmdclass={"build_ext": build_ext})
