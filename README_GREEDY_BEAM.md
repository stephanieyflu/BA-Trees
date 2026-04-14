## Greedy and Beam-Search Builders for Born-Again Trees

This document describes an extension of the BA-Trees codebase with **alternative tree builders** that trade optimality for speed and simplicity. The goal is to compare:

- The **exact dynamic-programming (DP)** solver (objectives 0–2 and 5 in the original code),
- Against **approximate but fast builders**:
  - `GreedyExactCells` (objective **6**),
  - `BeamSearchExactCells` (objective **7**).

These builders are useful when DP becomes too slow on larger instances and as a way to study how much tree size and faithfulness can be sacrificed for speed.

---

### 1. New objectives

We introduce two new objective codes:

- **6 = GreedyExactCells**
  - Top-down recursive builder.
  - At each node (region), evaluates all possible splits exactly on the **filtered cell space** (`fspaceFinal`) and chooses the split that best reduces impurity (e.g., Gini).
  - Produces a faithful tree on the discretized cell grid but is not guaranteed to be globally optimal in size.

- **7 = BeamSearchExactCells**
  - Beam search over partial trees with beam width \(B\) (CLI: **`-beam B`**, stored in `Params::beamWidth`, default \(B=5\)).
  - States are ranked by a score combining **total weighted Gini impurity** over pending regions and a **penalty on the number of splits** (scale depends on dataset size via `fspaceOriginal.nbCells`). Optional modes **`-bh 1`** and **`-bh 2`** change region prioritization and (for `-bh 2`) split ranking; see §5 below.
  - For \(B\le 1\) the implementation **delegates to `buildGreedyExact()`** so `-obj 7 -beam 1` matches greedy construction (see sanity check).

These are added to:

- `Params::objectiveFunction` comments,
- `Commandline` comments and CLI usage,
- `BornAgainDecisionTree::displayRunStatistics()` objective names.

---

### 2. Shared feature-space representation

All builders (DP, greedy, beam) use the same **filtered feature space**:

```cpp
fspaceOriginal.initializeCells(randomForest->getHyperplanes(), false);
fspaceFinal.initializeCells(fspaceOriginal.exportUsefulHyperplanes(), true);
```

- `fspaceFinal` defines:
  - `orderedHyperplaneLevels[k]` — hyperplane levels per feature,
  - `codeBook[k]` — strides to convert per-feature indices into a linear cell index,
  - `cells[idx]` — class label of each cell,
  - `nbCells` — total number of cells.
- A **region** is represented, as in the DP, by a pair of corner indices:
  - `(indexBottom, indexTop)` corresponding to the “bottom-left” and “top-right” cells in that region.

This ensures all builders operate on the same discretization of the forest’s decision space.

---

### 3. Exact cell-based impurity evaluation

Both greedy and beam search use **exact impurity calculations over cells**.

For a region `(indexBottom, indexTop)`:

1. For each feature `k`, compute its index range:

   ```cpp
   int low_k = fspaceFinal.keyToCell(indexBottom, k);
   int up_k  = fspaceFinal.keyToCell(indexTop, k);
   ```

2. Enumerate all combinations of feature indices within `[low_k, up_k]` using a recursive helper:

   ```cpp
   // Pseudo-code
   void enumerateRegion(int k, int keyPrefix) {
       if (k == nbFeatures) {
           int cls = fspaceFinal.cells[keyPrefix];
           counts[cls] += 1;
           return;
       }
       for (int i = low_k; i <= up_k; ++i) {
           int nextKey = keyPrefix + i * fspaceFinal.codeBook[k];
           enumerateRegion(k + 1, nextKey);
       }
   }
   ```

   - Start with `k = 0` and `keyPrefix = 0`.
   - At the base case (`k == nbFeatures`), each reachable `keyPrefix` is a cell index in the region.

3. Use the class counts to compute impurity. **Greedy and beam builders use Gini** in the C++ implementation. (The legacy **`-obj 4`** heuristic path uses **entropy** on manufactured samples, not the code in §3–§5 here.)

For a candidate split `(feature k, level l)`, we:

- Define the left and right child regions using the same corner index formulas as the DP:

  ```cpp
  int codeBookValue   = fspaceFinal.codeBook[k];
  int rangeLow        = fspaceFinal.keyToCell(indexBottom, k);
  int rangeUp         = fspaceFinal.keyToCell(indexTop, k);
  int indexTopLeft    = indexTop     + codeBookValue * (l - rangeUp);
  int indexBottomRight= indexBottom  + codeBookValue * (l + 1 - rangeLow);
  ```

- Enumerate cells in each child region to obtain left/right class counts.
- Compute weighted impurity:

  \[
  G_{\text{split}} = \frac{N_L}{N} G_L + \frac{N_R}{N} G_R
  \]

and pick the split that minimizes \(G_{\text{split}}\).

---

### 4. GreedyExactCells (objective 6)

Implementation outline in `BornAgainDecisionTree`:

- New public method:

  ```cpp
  void buildGreedyExact();
  ```

- Private helpers:

  ```cpp
  void computeClassCountsRegion(int indexBottom, int indexTop, std::vector<int> &counts);
  int  greedyBuildRegion(int indexBottom, int indexTop, unsigned int currentDepth);
  ```

Algorithm for `greedyBuildRegion`:

1. **Compute class counts** in the region using `computeClassCountsRegion`.
2. If all cells belong to a single class:
   - Create a **leaf** node with that class.
3. Else:
   - For each feature `k` and each feasible split level `l` (between `rangeLow` and `rangeUp - 1`):
     - Compute left and right child regions via the DP formulas.
     - Use `computeClassCountsRegion` on each child to get class counts.
     - Compute weighted impurity and keep track of the best `(k, l)`.
   - If no valid split improves impurity (or no split is available), create a **leaf with majority class**.
   - Otherwise:
     - Create an **internal node** with `splitFeature = k`, `splitValue = orderedHyperplaneLevels[k][l]`.
     - Recursively call `greedyBuildRegion` on the left and right child regions, attach children, and update depth and counters.

Entry point `buildGreedyExact()`:

1. Initialize counters (`finalSplits`, `finalLeaves`, `finalDepth`) and `rebornTree`.
2. Initialize `fspaceOriginal` / `fspaceFinal` as in `buildOptimal`.
3. Call `greedyBuildRegion(0, nbCells - 1, 0)`.
4. Export statistics and tree with the existing methods.

This yields a **single greedy BA-tree** that is faithful on the discretized cells but not guaranteed to be size-optimal.

---

### 5. BeamSearchExactCells (objective 7)

Beam search runs over **partial trees** with beam width **`Params::beamWidth`** (CLI **`-beam`**, default 5).

- A **state** (`BeamState` in `BornAgainDecisionTree.h`) contains pending regions, node IDs, a local `std::vector<Node>`, split count, and score.
- **Regional impurity** is **cell-count weighted Gini** (total cells in the region times standard Gini), summed over pending regions. The **state score** adds a **split penalty** scaled by `0.05 * fspaceOriginal.nbCells` (and different formulas when **`-bh`** is non-zero—see implementation in `buildBeamExact()`).

Algorithm (high level):

1. Initialize the beam with one state: a root leaf for the full space (majority class).
2. Until all regions are pure or a max-iteration cap is reached:
   - For each beam state, pick a **pending region to expand** (highest priority score; depends on **`-bh`**).
   - Enumerate admissible splits; rank by **weighted Gini** of children (with an extra imbalance penalty when **`-bh 2`**).
   - Expand the **top 3** candidate splits per state (not every split).
   - Collect child states, sort by score, retain the best **`beamWidth`** states.
3. Copy the lowest-score state’s tree into `rebornTree` and recompute depth / leaf / split statistics.

**`-bh` (beam heuristic)** is parsed in `Commandline` and stored in `Params::beamHeuristic`:

| `-bh` | Role |
|-------|------|
| `0` | Default: expand region with largest impurity; score uses impurity sum + scaled split penalty. |
| `1` | Region choice weights impurity by **number of distinct classes**; score adds a term involving a **lower bound on remaining splits**. |
| `2` | Region choice favors **shallower** pending nodes (`imp / (1 + depth)`); candidate splits are penalized when **child cell counts are imbalanced**. |

`-obj 7` calls `buildBeamExact()` in `main.cpp`.

---

### 6. How to run and compare (without overwriting files)

Assuming the binary `bornAgain` is built as in the main README, it is convenient to keep results organized in **subfolders** so you do not overwrite previous runs.

From `src/born_again_dp`:

```bash
mkdir -p results/dp
mkdir -p results/greedy
mkdir -p results/beam
```

- **DP (optimal, objective 1)**:

  ```bash
  ./bornAgain ../resources/forests/FICO/FICO.RF1.txt results/dp/fico_dp -obj 1
  ```

  This creates:
  - `results/dp/fico_dp.out`
  - `results/dp/fico_dp.tree`

- **GreedyExactCells (objective 6)**:

  ```bash
  ./bornAgain ../resources/forests/FICO/FICO.RF1.txt results/greedy/fico_greedy -obj 6
  ```

  This creates:
  - `results/greedy/fico_greedy.out`
  - `results/greedy/fico_greedy.tree`

- **BeamSearchExactCells (objective 7)**:

  ```bash
  ./bornAgain ../resources/forests/FICO/FICO.RF1.txt results/beam/fico_beam -obj 7 -beam 5
  ```

  This creates:
  - `results/beam/fico_beam.out`
  - `results/beam/fico_beam.tree`

For each pair of results (e.g., `results/dp/fico_dp.out` vs `results/greedy/fico_greedy.out`), compare:

- **Tree size/shape**:
  - BA tree depth,
  - Number of splits / leaves.
- **Performance**:
  - `CPU TIME(s)`,
  - `NB SUBPROBLEMS` / `NB RECURSIVE CALLS` (if instrumented).
- Optionally, **agreement with the random forest**:
  - Use the Python helpers to compare predictions of the forest and each BA-tree on the train/test data.

This yields clear, quantitative comparisons between optimal DP and approximate greedy/beam-search tree builders.

---

### 7. Beam width, `-bh`, and sanity check (`sanity_check/`)

- **`-beam <W>`** (optional, default `5`) sets the beam size for objective **7** only. Parsed in `Commandline.h` and copied to `Params::beamWidth` in `main.cpp`.
- **`-bh <h>`** (optional, default `0`) sets `Params::beamHeuristic` for objective **7** as in the table above. Ignored for other objectives.
- **Regression:** For `beamWidth <= 1`, `buildBeamExact()` **delegates to** `buildGreedyExact()` (a naive beam of width 1 would still differ from greedy because of region expansion order and multi-candidate expansion; delegation makes `-obj 7 -beam 1` and `-obj 6` produce the same tree).

**Recommended check** (uses Python so it works with `bornAgain.exe` on Windows, normalizes CRLF, and avoids PowerShell exit-code quirks):

```bash
cd src/born_again_dp
make          # Linux / WSL
# OR on Git Bash / MinGW64 native Windows:
mingw32-make  # produces bornAgain.exe (or a PE `bornAgain`); do not use a Linux ELF `bornAgain` on Windows
python sanity_check/run_sanity_check.py
```

The script **skips Linux ELF** files named `bornAgain` when you are on native Windows (they cause `WinError 193`). You must compile locally with **MinGW** so a Windows PE executable exists.

This writes `sanity_check/results/sanity_result.txt` and compares `greedy.tree` vs `beam_w1.tree`. If the script reports failure, ensure the solver was **rebuilt** after pulling changes to `Commandline.h` / `Params.h` / `BornAgainDecisionTree.cpp`. Git Bash users without Python should run `python sanity_check/run_sanity_check.py` from WSL or install Python; the bare `run_sanity_check.sh` fallback requires a Unix `bornAgain` binary (not always present on native Windows builds).

