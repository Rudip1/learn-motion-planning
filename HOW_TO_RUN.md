# How to run

Tested on Ubuntu 24.04 with GCC 13, CMake 3.28 and Python 3.12/3.13. Any C++17 compiler, CMake ≥ 3.20 and
Python ≥ 3.10 should work.

## Dependencies

```bash
sudo apt install build-essential cmake libeigen3-dev python3-venv
```

Eigen is optional: if CMake does not find Eigen 3.4 it downloads the headers. Catch2 (tests only) is always
downloaded by CMake.

## C++ library and tests

```bash
make build     # cmake -S . -B build/cmake && cmake --build build/cmake
make test      # ctest: every algorithm against an independent reference
```

The command-line examples are built next to the tests, e.g.

```bash
./build/cmake/cpp/examples/01_vehicle_kinematics 15 10 -90 > trajectory.csv
```

## Python module

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[notebooks]"        # builds the C++ with scikit-build-core and installs `motion_planning`
python -c "import motion_planning as mp; print(mp.wrap_angle(4.0))"
```

`pip install -e .` rebuilds the extension when the C++ changes and you reinstall; the Python files in
`python/motion_planning/` are picked up without reinstalling.

## Notebooks

Each chapter has two notebooks with the same cells:

- `2_notebooks/exercises/NN_<chapter>.ipynb` — the reader's copy; ✏️ cells are left for you, and the `assert`
  cell after each one checks your answer.
- `2_notebooks/solutions/NN_<chapter>.ipynb` — solved, executes top to bottom.

```bash
jupyter lab 2_notebooks/exercises/      # pip install jupyterlab, if needed
```

Both notebooks are generated from one source, `tools/notebooks/NN_<chapter>.py`; edit that file, then

```bash
make notebooks          # regenerate both notebooks and execute the solutions (stores outputs)
make check-notebooks    # what CI runs: sources and notebooks in sync, every solution executes
```

### Colab

Open a notebook from GitHub in Colab (`File → Open notebook → GitHub`). The first cell installs the module from
this repository when it is missing; the build takes a couple of minutes the first time.

## Figures

Every figure in `1_theory/figures/` is produced by a script in `tools/figures/`:

```bash
make figures            # all of them
python tools/make_figures.py 01     # one chapter
```

## Formatting

```bash
make format    # clang-format, Google base style, 4-space indent, 110 columns (.clang-format)
```
