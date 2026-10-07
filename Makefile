# make build      configure and build the C++ library, tests and examples in build/cmake
# make test       run the C++ test suite
# make python     install the Python module in editable mode (builds the C++ through scikit-build-core)
# make notebooks  regenerate notebooks from tools/notebooks/ and execute every solution notebook
# make figures    regenerate 1_theory/figures/
# make format     clang-format all C++ sources
# make all        build + test + python + notebooks

PYTHON ?= python3
BUILD_DIR ?= build/cmake
JOBS ?= $(shell nproc 2>/dev/null || echo 2)

.PHONY: all build test python notebooks check-notebooks figures format clean

all: build test python notebooks

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j $(JOBS)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure -j $(JOBS)

python:
	$(PYTHON) -m pip install -e ".[notebooks]"

notebooks:
	$(PYTHON) tools/build_notebooks.py
	$(PYTHON) tools/execute_notebooks.py

check-notebooks:
	$(PYTHON) tools/build_notebooks.py --check
	$(PYTHON) tools/execute_notebooks.py --no-write

figures:
	$(PYTHON) tools/make_figures.py

format:
	clang-format -i $$(find cpp python -name '*.hpp' -o -name '*.cpp')

clean:
	rm -rf build
