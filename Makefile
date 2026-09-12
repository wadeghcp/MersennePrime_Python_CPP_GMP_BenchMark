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

all: $(TARGET)
	@echo "Build done: $(TARGET)"

$(TARGET): lucaslehmer.cpp
	$(CXX) $(CXXFLAGS) $(PYINC) -shared -o $@ $< -lgmp

test: $(TARGET)
	$(PYTHON) -m pytest -q tests

clean:
	rm -f lucaslehmer*.so
	@echo Clean done

.PHONY: all test clean
