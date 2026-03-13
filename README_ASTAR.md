## A* vs Dynamic Programming: How to Build and Compare

This file explains how to build the C++ code and compare the original dynamic-programming (DP) algorithm with the new A* implementation for generating born-again trees.

### 1. Build the C++ binary (recommended: Ubuntu via WSL)

The project ships with a `makefile` in `src/born_again_dp`. The most robust way to build it on Windows is to use Ubuntu via WSL.

1. Open the **Ubuntu** terminal (WSL).
2. Install build tools (one time):

```bash
sudo apt update
sudo apt install build-essential
```

3. Go to the BA-Trees source folder:

```bash
cd "/mnt/c/Users/steph/OneDrive/Documents/GitHub/BA-Trees/src/born_again_dp"
```

4. Build the binary:

```bash
make
```

This should produce an executable called `bornAgain` in the same directory.

> If you prefer to use MSYS2/MinGW instead of WSL, you can open a MinGW64 terminal, install `gcc` and `make` via `pacman`, then run the same `make` command in `src/born_again_dp`.

### 2. Run DP (baseline) vs A* (new implementation)

Once `bornAgain` is built, you can compare the original DP algorithm (objective 1) with the new A* search (objective 5). The examples below use the FICO dataset; you can swap in any other forest file from `src/resources/forests`.

From `src/born_again_dp`:

#### 2.1. DP run (objective: minimize number of leaves, `-obj 1`)

```bash
./bornAgain ../resources/forests/FICO/FICO.RF1.txt fico_dp -obj 1
```

This will generate:

- `fico_dp.out` — CSV-style statistics for the run.
- `fico_dp.tree` — the resulting born-again tree.

#### 2.2. A* run (objective: minimize number of leaves, `-obj 5`)

```bash
./bornAgain ../resources/forests/FICO/FICO.RF1.txt fico_astar -obj 5
```

This will generate:

- `fico_astar.out` — statistics for the A* run.
- `fico_astar.tree` — the resulting born-again tree built directly from the A* search.

You can repeat the same pattern for other datasets, for example:

```bash
./bornAgain ../resources/forests/COMPAS-ProPublica/COMPAS-ProPublica.RF1.txt compas_dp -obj 1
./bornAgain ../resources/forests/COMPAS-ProPublica/COMPAS-ProPublica.RF1.txt compas_astar -obj 5
```

### 3. What to compare in the `.out` files

Open the two `.out` files for a given dataset, e.g. `fico_dp.out` and `fico_astar.out`, and compare:

- **Tree quality**
  - BA tree depth
  - Number of splits
  - Number of leaves
  - These should match between DP and A* if A* is exact.

- **Performance**
  - `CPU TIME(s)` — total runtime of the algorithm.
  - `NB SUBPROBLEMS` — for DP: number of DP subproblems; for A*: number of generated A* states.
  - `NB RECURSIVE CALLS` — for DP: number of recursive DP calls; for A*: number of expanded A* states.

These statistics let you quantify both correctness (matching tree structure/size) and efficiency (time and search effort) of the A* implementation compared to the original dynamic programming approach.

