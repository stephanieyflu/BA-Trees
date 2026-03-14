## Greedy and Beam-Search Builders for Born-Again Trees

This document describes an extension of the BA-Trees codebase with **alternative tree builders** that trade optimality for speed and simplicity. The goal is to compare:

- The **exact dynamic-programming (DP)** solver (objectives 0–2),
- Against **approximate but fast builders**:
  - `GreedyExactCells` (objective 6),
  - `BeamSearchExactCells` (objective 7, planned / optional).

These builders are useful when DP becomes too slow on larger instances and as a way to study how much tree size and faithfulness can be sacrificed for speed.

---

### 1. New objectives

We introduce two new objective codes:

- **6 = GreedyExactCells**
  - Top-down recursive builder.
  - At each node (region), evaluates all possible splits exactly on the **filtered cell space** (`fspaceFinal`) and chooses the split that best reduces impurity (e.g., Gini).
  - Produces a faithful tree on the discretized cell grid but is not guaranteed to be globally optimal in size.

- **7 = BeamSearchExactCells**
  - Beam search over partial trees with a fixed beam width \(B\).
  - Keeps the best \(B\) partial trees at each depth according to a score (e.g., impurity + splits used).
  - Interpolates between greedy (beam size 1) and exhaustive search.

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

3. Use the class counts to compute impurity:
   - **Gini**:

     \[
     G = 1 - \sum_c \left(\frac{n_c}{N}\right)^2
     \]

   - Or **entropy**, if desired.

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

Beam search is implemented as a search over **partial trees** with a fixed beam width \(B\) (currently `BEAM_WIDTH = 5` in `buildBeamExact`).

- A **state** (`BeamState` in `BornAgainDecisionTree.cpp`) contains:
  - A list of pending regions `(indexBottom, indexTop)`,
  - A mapping from each pending region to a node ID in the state’s local `tree`,
  - A local `std::vector<Node>` representing the partial BA-tree,
  - The number of splits so far and a heuristic score.
- The **score** is:
  - The sum of Gini impurities of all pending regions (using `computeClassCountsRegion`),
  - Plus the number of splits, so the beam prefers purer trees with fewer splits.

Algorithm:

1. Initialize the beam with a single state:
   - One root leaf covering the whole space,
   - Root classification = majority class over all cells.
2. While the beam is non-empty and a max-iteration guard is not hit:
   - If all regions in all states are pure, stop.
   - For each state in the current beam:
     - Choose the **most impure region** to expand.
     - Enumerate all candidate splits `(feature k, level l)` for that region.
     - For each candidate, compute:
       - Left/right child regions and class counts via `computeClassCountsRegion`,
       - Weighted Gini impurity of the split.
     - Keep the **top few** (currently 3) candidate splits, and for each:
       - Create a child state by:
         - Turning the parent node into an internal node with that split,
         - Creating left/right leaf children with majority labels,
         - Updating the pending region list and node-ID mapping.
       - Recompute the child’s score.
   - Merge all child states into a new beam, sort by score, and keep the best `B`.
3. After termination, pick the best state in the beam and copy its `tree` into `rebornTree`, recomputing `finalSplits`, `finalLeaves`, and `finalDepth`.

`-obj 7` in the CLI now calls `buildBeamExact`, so beam search is fully implemented and usable.

---

### 6. How to run and compare (without overwriting files)

Assuming the binary `bornAgain` is built as in the main README, it is convenient to keep results organized in **subfolders** so you do not overwrite previous runs.

From `src/born_again_dp`:

```bash
mkdir -p results/dp
mkdir -p results/greedy
mkdir -p results/beam   # once beam search is implemented
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

- **BeamSearchExactCells (objective 7, once implemented)**:

  ```bash
  ./bornAgain ../resources/forests/FICO/FICO.RF1.txt results/beam/fico_beam -obj 7
  ```

  This would create:
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

