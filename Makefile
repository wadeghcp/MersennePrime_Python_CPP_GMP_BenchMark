# Build the lucaslehmer extension for whichever python is first on PATH (or PYTHON=...).
#   make            -> lucaslehmer<abi>.so next to the sources
#   make test       -> pytest
#   make clean
PYTHON  ?= python3
CXX     ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -fPIC -fvisibility=hidden -Wall -pthread
PYINC   := $(shell $(PYTHON) -m pybind11 --includes)
EXT     := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET  := lucaslehmer$(EXT)
# FFT backend for the floating-point engine: fftw (libfftw3-dev, default) or mkl (oneMKL's FFTW3 wrapper, AVX-512)
FFT     ?= fftw
ifeq ($(FFT),mkl)
  # oneAPI install: MKLROOT=/opt/intel/oneapi/mkl/<ver>. Debian/Ubuntu libmkl-dev: MKLINC=/usr/include/mkl MKLLIB=/usr/lib/x86_64-linux-gnu
  MKLROOT ?= /opt/intel/oneapi/mkl/latest
  MKLINC  ?= $(MKLROOT)/include
  MKLLIB  ?= $(MKLROOT)/lib
  FFTFLAGS := -I$(MKLINC)/fftw -I$(MKLINC) -DLL_FFT_BACKEND='"mkl"' -DLL_FFT_MKL
  # mkl_rt (single dynamic library) is the only MKL link that works from a dlopen'ed Python extension:
  # the split libs make libmkl_core's dispatch fail with "cannot load libmkl_avx512.so ... undefined symbol".
  # The threading layer is forced to sequential at import, so libiomp5 is not needed.
  FFTLIBS  := -L$(MKLLIB) -Wl,-rpath,$(MKLLIB) -lmkl_rt -ldl
else
  FFTFLAGS :=
  FFTLIBS  := -lfftw3
endif

all: $(TARGET)
	@echo "Build done: $(TARGET)"

$(TARGET): lucaslehmer.cpp
	$(CXX) $(CXXFLAGS) $(FFTFLAGS) $(PYINC) -shared -o $@ $< -lgmp $(FFTLIBS) -lm

test: $(TARGET)
	$(PYTHON) -m pytest -q tests

clean:
	rm -f lucaslehmer*.so
	@echo Clean done

.PHONY: all test clean
