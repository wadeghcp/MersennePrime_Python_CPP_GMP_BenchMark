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
  MKLROOT ?= /opt/intel/oneapi/mkl/latest
  FFTFLAGS := -I$(MKLROOT)/include/fftw -DLL_FFT_BACKEND='"mkl"'
  FFTLIBS  := -L$(MKLROOT)/lib -Wl,-rpath,$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -ldl
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
