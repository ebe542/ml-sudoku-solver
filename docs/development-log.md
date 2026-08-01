# Development Log

This document records the development process of the project. The project is developed as a self-study machine learning project.

---
## Commit 1 — Project Initialization

**Commit:** `chore: initialize project structure`

### Objective

Create the initial repository structure for a maintainable machine learning project.

### Implemented

- Created the project directory structure.
- Added Python package structure.
- Added project configuration.
- Added README documentation.
- Added GitHub configuration.
- Added development documentation and licensing.

### Result

The repository provides a clean foundation for further development.

---
## Commit 2 — Sudoku Grid Representation

**Commit:** `feat: add sudoku grid representation`

### Objective

Create a Python representation of a 9×9 Sudoku grid.

### Implemented

- Added `SudokuGrid`.
- Added validation of grid dimensions.
- Added validation of allowed values.
- Added detection of empty cells.
- Added detection of complete grids.

### Testing

The implementation was tested with `pytest`.

Result:

`6 passed`

---
## Commit 3 — Sudoku Grid Validation

**Commit:** `feat: add sudoku grid validation`

### Objective

Ensure that Sudoku grids do not contain structural conflicts.

### Implemented

- Row validation.
- Column validation.
- 3×3 block validation.
- Complete grid validation.

### Testing

Result:

`10 passed`

---
## Commit 4 — Synthetic Sudoku Dataset

**Commit:** `feat: add synthetic sudoku dataset generator`

### Objective

Create reproducible training data without relying on an external dataset.

### Approach

A solved Sudoku grid is used as the ground truth. Random cells are removed to create incomplete puzzles.

```text
Solved Sudoku
      ↓
Remove cells
      ↓
Incomplete Sudoku
```

---
## Commit 5 — ML Feature Preprocessing

**Commit:** `feat: add sudoku feature preprocessing`

### Objective

Transform incomplete Sudoku puzzles into a representation that can be used as input for a machine learning model.

### Approach

Each empty cell in a Sudoku puzzle is treated as an individual machine learning sample.

For every empty cell:

- The complete 9×9 Sudoku grid is flattened into 81 features.
- The position of the empty cell is added as an additional feature.
- The corresponding digit from the solved Sudoku is used as the target.

```text
Incomplete Sudoku
        │
        ▼
Find empty cells
        │
        ▼
Create one sample per empty cell
        │
        ├── 81 grid features
        ├── 1 cell position feature
        │
        ▼
Target: correct Sudoku digit
```

---
## Commit 6 — Documentation & Architecture

**Commit:** `docs: document project architecture and development history`

### Objective

Document the development process and establish a clear overview of the project's current architecture.

### Implemented

- Added the development log.
- Added the project architecture documentation.
- Documented the purpose and responsibilities of the main components.
- Documented the development history and design decisions.

### Documentation

The project documentation is separated into two main areas:

- `docs/development-log.md` — development history, implementation steps, tests, and design decisions.
- `docs/architecture.md` — current technical architecture and component responsibilities.

### Result

The repository now provides both a historical development record and a technical overview of the project architecture.


---
## Commit 7 — Diverse Sudoku Training Data

**Commit:** `feat: generate diverse sudoku training data`

### Objective

Improve the synthetic dataset generation by creating training examples from multiple independent solved Sudoku grids.

The previous dataset generator created multiple incomplete puzzles from a single solved Sudoku grid. While this was useful for testing the data pipeline, it would not provide sufficient diversity for machine learning.

### Approach

A randomized backtracking algorithm is used to generate complete, valid Sudoku grids.

Each generated solution is then transformed into an incomplete puzzle by randomly removing cells.

```text
Random Sudoku seed
        │
        ▼
Generate solved Sudoku
        │
        ▼
Remove random cells
        │
        ▼
Incomplete Sudoku puzzle
```

---
## Commit 8 — Train/Test Data Preparation

**Commit:** `feat: add train test data preparation`

### Objective

Prepare the generated Sudoku data for machine learning by creating separate training and test datasets.

### Approach

The data is split at the level of complete Sudoku solutions rather than individual cells.

This prevents puzzles derived from the same solution from being distributed across both training and test data.

```text
Generated Sudoku solutions
            │
            ▼
      Solution-level split
            │
       ┌────┴────┐
       ▼         ▼
    Training    Test
       │         │
       ▼         ▼
    Puzzles   Puzzles
       │         │
       ▼         ▼
   Features   Features
       │         │
       ▼         ▼
   X_train    X_test
   y_train    y_test
```

**Implemented**

+ Added `MLDataSplit`.
+ Added train/test data preparation.
+ Added configurable test set size.
+ Added reproducible splitting using a random seed.
+ Added validation of split parameters.
+ Reused the existing diverse Sudoku dataset generator.
+ Reused the existing feature preprocessing pipeline.

### Data Representation

The resulting data is separated into:

```
X_train → training features
y_train → training targets

X_test  → test features
y_test  → test targets
```

Each feature vector contains:

```
81 Sudoku cell values
+
1 target cell position
```

### Data Leakage Consideration

The split is performed before converting the puzzles into cell-level machine learning samples.

This is important because one Sudoku solution can produce multiple incomplete puzzles. Splitting individual cell samples could allow related puzzles to appear in both the training and test sets.

By splitting complete solutions first, the test set contains Sudoku structures that were not used to create the training data.

**Reproducibility**

The split accepts a random seed. Using the same seed produces the same training and test sets.

Testing

All tests passed.

```
33 passed
```

---
## Commit 9 — Baseline Random Forest Model

**Commit:** `feat: add baseline random forest model`

### Objective

Introduce the first machine learning model for the Sudoku prediction pipeline.

The goal is to establish a baseline that can predict the correct digit for a selected empty Sudoku cell based on the current puzzle state.

### Model

A `RandomForestClassifier` from scikit-learn is used as the first baseline model.

The model is wrapped in `SudokuRandomForest` to separate the model implementation from the rest of the application.

### Pipeline

```text
Training Data
      │
      ▼
X_train / y_train
      │
      ▼
Random Forest Classifier
      │
      ▼
     fit()
      │
      ▼
Trained Model
      │
      ▼
    X_test
      │
      ▼
 Predictions
      │
      ▼
   Accuracy
```

### Implemented

+ Added the `SudokuRandomForest` model wrapper.
+ Added Random Forest training.
+ Added digit prediction.
+ Added model evaluation using accuracy.
+ Added configurable number of estimators.
+ Added reproducible model training through a random seed.
+ Added model tests for training and prediction.
+ Added validation that predictions are valid Sudoku digits.


### Baseline Experiment

The initial baseline was evaluated using:

```text
Number of solved Sudoku solutions: 100
Training/test split: 80% / 20%
Training samples: 3,200
Test samples: 800
Random Forest estimators: 100
Random seed: 42
```

### Result

The baseline achieved:

```text
Test accuracy: 11.75%
```

There are nine possible Sudoku digits (1–9), so a uniform random classifier would achieve approximately 11.11% (1 / 9) accuracy.

The baseline therefore performs only slightly better than random prediction.

### Interpretation

The result shows that the current feature representation does not provide enough information for the Random Forest to reliably infer the correct Sudoku digit.

The model currently receives the Sudoku grid and the target cell position, but the Sudoku constraints are not explicitly represented as features.

This provides a useful baseline for future feature engineering and model improvements.

### Important Limitation

This accuracy measures cell-level digit prediction.

It does not represent the accuracy of a complete Sudoku solver.

---
## Commit 10 — Sudoku Constraint Features

**Commit:** `feat: add sudoku constraint features`

### Objective

Improve the feature representation by explicitly providing Sudoku-specific constraint information to the machine learning model.

The baseline model only received the Sudoku grid and the target cell position. The new features additionally describe which digits are valid candidates for the target cell.

### Implemented

- Added Sudoku candidate calculation based on:
  - row constraints
  - column constraints
  - 3×3 block constraints
- Added nine binary candidate features for each target cell.
- Extended the feature vector from 82 to 91 features.
- Added tests for Sudoku candidate calculation.
- Added tests for the new candidate features.

### Feature Representation

Each sample now contains:

```text
81 grid values
+ 1 target cell position
+ 9 candidate indicators
= 91 features
```

The nine candidate indicators represent digits 1–9:

```text
1 → candidate is possible
0 → candidate is not possible
```

The candidate information is calculated exclusively from the incomplete Sudoku puzzle. The solution is only used to generate the target value.

### Baseline Comparison

The same Random Forest configuration was evaluated before and after adding the constraint features.

| Configuration | Features | Test Accuracy |
| ------------- | -------- | ------------- |
| Baseline | 82 | 11.75% |
| Constraint features | 91 | 50.25% |


### Result

Adding the Sudoku constraint features increased test accuracy from 11.75% to 50.25%.

This is a substantial improvement without changing the model, training data size, or Random Forest configuration.

### Interpretation

The result demonstrates the importance of domain-specific feature engineering.

The baseline model had to learn Sudoku constraints implicitly from the grid representation. By explicitly representing possible candidate digits, important Sudoku structure becomes directly available to the model.

The experiment therefore provides evidence that the feature representation has a major impact on model performance.

### Current Limitation

The 50.25% accuracy measures cell-level digit prediction.

It does not represent the ability to solve a complete Sudoku puzzle. The model still predicts individual target cells rather than producing a complete solved grid.

---
## Commit 11 — Error Analysis — Candidate Count

**Commit:** `feat: error analysis`

The prediction accuracy strongly depends on the number of valid candidate digits for the target cell.

| Candidates | Accuracy | Samples |
|---:|---:|---:|
| 1 | 100.0% | 182 |
| 2 | 44.9% | 274 |
| 3 | 30.2% | 235 |
| 4 | 26.1% | 92 |
| 5 | 13.3% | 15 |
| 6 | 0.0% | 2 |

The model performs perfectly when only one candidate is possible. However, performance decreases significantly as the number of possible candidates increases.

This indicates that the current model can exploit local Sudoku constraints but has difficulty resolving situations where multiple candidate digits remain possible.

This is an important limitation of the current cell-level prediction approach because Sudoku decisions can depend on relationships between multiple cells.

---
## Commit 12 — Candidate Interaction Features

**Commit:** `feat: add candidate interaction features`

### Objective

Improve the feature representation by incorporating information about the candidate distributions in neighbouring cells.

### Motivation

The previous error analysis showed that prediction accuracy decreases rapidly as the number of valid candidate digits increases.

This indicates that local candidate information alone is insufficient for distinguishing between multiple valid candidates.

### Implemented

- Added candidate interaction features.
- Counted candidate occurrences in
  - the same row,
  - the same column,
  - the same 3×3 block.
- Added 27 interaction features.
- Increased the feature vector size from 91 to 118 features.
- Added unit tests for candidate interaction calculation.

### Evaluation

The Random Forest configuration remained unchanged.

| Feature representation | Accuracy |
|------------------------|---------:|
| Baseline (82 features) | 11.75 % |
| Constraint features (91 features) | 50.25 % |
| Interaction features (118 features) | 66.50 % |

### Result

The interaction features improved the prediction accuracy from 50.25 % to 66.50 %.

This demonstrates that relationships between neighbouring candidate sets provide valuable information for distinguishing between multiple valid Sudoku candidates.

### Conclusion

The experiment confirms that representing local interactions between candidate sets significantly improves model performance without changing the underlying learning algorithm.

---
## Commit 13 — Feature Importance Analysis

**Commit:** `feat: add feature importance analysis`

### Objective

Improve the interpretability of the Random Forest model by analyzing which input features contribute most to the prediction.

The goal is not to improve prediction accuracy but to better understand how the model makes its decisions.

### Implemented

- Added grouped feature importance analysis.
- Added readable names for all 118 features.
- Added ranking of the most important individual features.
- Added unit tests for feature-name mapping.
- Moved reusable analysis code into the `analysis` package.

### Usage

Run the analysis from the project root:

```bash
python -m sudoku_ml.analysis.feature_importance
```

The script trains the current Random Forest model using the default configuration and reports:

- feature importance by feature group,
- the most important individual features,
- a ranked feature importance summary.

### Feature Groups

The feature vector now consists of six logical groups:

| Feature Group | Features |
|--------------|---------:|
| Grid values | 81 |
| Cell position | 1 |
| Candidate indicators | 9 |
| Row interactions | 9 |
| Column interactions | 9 |
| Block interactions | 9 |

### Feature Contribution by Group

| Feature Group | Importance |
|--------------|-----------:|
| Grid values | 30.97 % |
| Candidate indicators | 28.42 % |
| Row interactions | 13.18 % |
| Column interactions | 13.04 % |
| Block interactions | 10.95 % |
| Cell position | 3.43 % |

### Key Findings

The original Sudoku grid remains the single most important source of information.

The candidate indicators contribute almost as much information as the entire grid representation.

The interaction features account for approximately **37.17%** of the total feature importance:

- Row interactions: **13.18%**
- Column interactions: **13.04%**
- Block interactions: **10.95%**

This explains the significant improvement from **50.25%** to **66.50%** prediction accuracy after introducing candidate interaction features.

### Most Important Individual Features

The highest-ranked individual features are:

1. `candidate_3`
2. `target_cell_index`
3. `candidate_4`
4. `candidate_6`
5. `candidate_9`

The complete ranking is available through the analysis script.

### Discussion

One unexpected observation is the relatively high importance of the target cell position.

Although the position contributes only **3.43%** of the overall feature importance, it is the second most important individual feature. This may indicate that the Random Forest learns position-dependent patterns from the generated training data.

This observation should be investigated in future experiments.

### Conclusion

The analysis confirms that the improvements achieved in previous commits are supported by meaningful feature usage rather than random variation.

Both candidate indicators and candidate interaction features provide substantial information to the Random Forest and are responsible for a large part of the overall prediction performance.

---
## Commit 14 — Grouped Cross-Validation Evaluation

**Commit:** `feat: add grouped cross validation evaluation`

### Objective

Evaluate the stability and generalization performance of the Random Forest model across multiple train/test partitions.

The previous evaluation used a single solution-level train/test split. Grouped cross-validation provides a more reliable estimate by evaluating the model across five independent folds.

### Why GroupKFold?

Each generated Sudoku puzzle produces multiple cell-level machine learning samples.

A standard `KFold` split could place cells originating from the same Sudoku puzzle in both the training and validation sets. This would cause data leakage because the model would be evaluated on data closely related to its training samples.

`GroupKFold` keeps all samples originating from the same Sudoku puzzle in the same fold.

```text
Sudoku solution
      │
      ├── Cell sample 1
      ├── Cell sample 2
      ├── ...
      └── Cell sample 40
             │
             ▼
     One shared group ID
```

### Implemented

- Added group identifiers for cell-level samples.
- Added grouped feature and target generation.
- Added reusable grouped cross-validation logic.
- Added five-fold `GroupKFold` evaluation.
- Added summary statistics for:
  - mean accuracy,
  - standard deviation,
  - minimum accuracy,
  - maximum accuracy.
- Extended the Random Forest wrapper with array-based training.
- Added validation and reproducibility tests.

### Usage

Run the grouped cross-validation evaluation from the project root:

```bash
python scripts/evaluate_group_cross_validation.py
```

### Evaluation Configuration

```text
Sudoku solutions: 100
Cross-validation folds: 5
Removal rate: 0.5
Random Forest estimators: 100
Random seed: 42
Feature count: 118
```

### Results

| Fold | Accuracy |
|---:|---:|
| 1 | 67.00 % |
| 2 | 63.38 % |
| 3 | 67.37 % |
| 4 | 64.38 % |
| 5 | 67.00 % |

### Summary

| Metric | Accuracy |
|---|---:|
| Mean | 65.83 % |
| Standard deviation | 1.63 percentage points |
| Minimum | 63.38 % |
| Maximum | 67.37 % |

### Interpretation

The mean grouped cross-validation accuracy of **65.83%** is close to the previous single-split result of **66.50%**.

This indicates that the previous result was representative rather than the consequence of an unusually favorable train/test split.

The relatively small standard deviation shows that the model performs consistently across different groups of previously unseen Sudoku solutions.

### Conclusion

Grouped cross-validation confirms that the current Random Forest and feature representation provide stable cell-level prediction performance while avoiding leakage between samples derived from the same Sudoku.

---
## Commit 15 — Hybrid ML Sudoku Solver

**Commit:** `feat: add hybrid ML-guided Sudoku solver`

### Objective

Extend the cell-level classifier into a system capable of solving complete Sudoku puzzles without allowing invalid model predictions.

### Approach

The solver combines deterministic constraints with ML-guided backtracking:

```text
Incomplete Sudoku
       │
       ▼
Select cell with fewest candidates
       │
       ├── one candidate ──→ deterministic placement
       │
       └── multiple candidates
                   │
                   ▼
       Rank by Random Forest probability
                   │
                   ▼
       Try candidates with backtracking
```

### Implemented

- Added `HybridSudokuSolver`.
- Added minimum-remaining-values cell selection.
- Added candidate ranking based on Random Forest probabilities.
- Added recursive backtracking for globally consistent solutions.
- Preserved the original puzzle during solving.
- Rejected structurally invalid input puzzles.
- Exposed reusable single-cell feature generation.
- Extended the Random Forest wrapper with probability and class access.

### Design Decision

The model only determines the order in which valid candidates are attempted. Sudoku constraints filter all candidates, while backtracking ensures that a locally plausible prediction cannot make the final solution invalid.

### Testing

The test suite verifies complete and valid solutions, preservation of given digits, input immutability, and rejection of conflicting puzzles.

Result:

`71 passed`

---
## Commit 16 - End-to-End Solver Evaluation

**Commit:** `feat: add end-to-end solver evaluation`

### Objective

Evaluate the complete hybrid system rather than measuring only cell-level digit prediction accuracy.

### Implemented

- Added `SolverStats` for deterministic steps, ML decisions, and backtracks.
- Reset solver statistics for every solving attempt.
- Added `SolverEvaluationResult` with aggregated metrics.
- Added solution-rate, validity-rate, runtime, and backtracking summaries.
- Verified that completed solutions preserve all original clues.
- Added an executable end-to-end evaluation script.
- Added unit and integration tests for statistics and evaluation metrics.

### Usage

Run the evaluation from the project root:

```bash
python scripts/evaluate_solver.py
```

### Evaluation Configuration

```text
Training solutions: 100
Training/test split: 80% / 20%
Evaluation puzzles: 20
Removal rate: 0.5
Random Forest estimators: 100
Training random seed: 42
Evaluation random seed: 123
```

The separate evaluation seed produces an independently generated collection of puzzles that is not reused for model training.

### Results

| Metric | Result |
|---|---:|
| Puzzles evaluated | 20 |
| Puzzles solved | 20 |
| Valid solutions | 20 |
| Solution rate | 100.00% |
| Valid solution rate | 100.00% |
| Average runtime | 20.64 ms |
| Deterministic steps | 774 |
| ML decisions | 26 |
| Backtracks | 0 |
| Average backtracks | 0.00 |

### Interpretation

The solver placed 800 digits across the 20 evaluation puzzles. Of these, **96.75%** were deterministic single-candidate steps and **3.25%** required ML-based candidate ranking.

No candidate choice led to a dead end, so backtracking was not required in this experiment. This shows that puzzles generated with a removal rate of 0.5 are largely solved through deterministic constraint propagation.

The 100% solution rate demonstrates that the integrated solver produces valid complete grids for this evaluation set. It does not mean that the underlying cell classifier has 100% prediction accuracy, because Sudoku constraints and backtracking protect the final solving process from invalid choices.

### Testing

Result:

`75 passed`

### Next Step

Evaluate multiple removal rates and compare the ML-guided solver with a classical non-ML candidate-ordering baseline.

---
## Commit 17 - ML-Guided and Classical Solver Comparison

**Commit:** `feat: compare ML-guided and classical solver`

### Objective

Measure whether Random Forest candidate ranking improves the complete solving process compared with a classical non-ML ordering strategy.

### Experimental Design

Both solvers use the same:

- Sudoku constraints,
- minimum-remaining-values cell selection,
- recursive backtracking implementation,
- evaluation puzzles,
- solution-validity checks.

The only intentional difference is candidate ordering:

```text
Hybrid solver     -> descending Random Forest probability
Classical solver  -> ascending numerical digit order
```

### Implemented

- Added `ClassicalSudokuSolver` without a model dependency.
- Reused the hybrid solver's validation and search implementation.
- Added `SolverComparisonResult`.
- Added a reusable function for evaluation on identical puzzle sets.
- Added a command-line comparison experiment.
- Added unit and integration tests for the classical solver and comparison.

### Usage

Run the comparison from the project root:

```bash
python scripts/compare_solvers.py
```

### Evaluation Configuration

```text
Training solutions: 100
Training/test split: 80% / 20%
Evaluation puzzles: 20
Removal rate: 0.65
Random Forest estimators: 100
Training random seed: 42
Evaluation random seed: 123
```

### Results

| Metric | Hybrid | Classical |
|---|---:|---:|
| Solution rate | 100.00% | 100.00% |
| Valid solution rate | 100.00% | 100.00% |
| Average runtime | 144.88 ms | 4.39 ms |
| Deterministic steps | 1,306 | 1,867 |
| Backtracks | 494 | 1,145 |
| Average backtracks | 24.70 | 57.25 |
| ML decisions | 186 | 0 |

### Interpretation

The hybrid solver required **651 fewer backtracks**, a reduction of approximately **56.9%** compared with ascending numerical candidate ordering. This demonstrates that the model provides useful information for choosing promising search branches.

Despite the smaller search tree, the hybrid solver averaged **144.88 ms** per puzzle, compared with **4.39 ms** for the classical solver. The hybrid solver was therefore approximately **33 times slower**.

The runtime cost of repeatedly generating 118 features and executing Random Forest inference is greater than the cost of the additional classical backtracking for this experiment. The result therefore shows a trade-off:

```text
ML guidance -> better search order, fewer backtracks
Classical   -> more backtracks, lower total runtime
```

Both strategies achieved a 100% valid solution rate because deterministic constraint filtering and backtracking guarantee that invalid local choices do not become final solutions.

The deterministic-step totals include steps performed on search paths that were later reversed. They measure computational work rather than unique solved cells.

### Testing

Result:

`77 passed`

### Conclusion

The Random Forest improves candidate ordering, but it does not currently improve end-to-end runtime. A broader evaluation across multiple removal rates is needed to determine how this trade-off changes with puzzle difficulty.

---
## Commit 18 - Solver Evaluation Across Removal Rates

**Commit:** `feat: evaluate solver across removal rates`

### Objective

Measure how the trade-off between ML-guided and classical candidate ordering changes as more cells are removed from the generated Sudoku puzzles.

### Terminology

Removal rate is used as a reproducible proxy for difficulty. It is not a formal Sudoku difficulty rating: puzzles with the same number of empty cells can require very different solving techniques and search effort.

### Implemented

- Added `RemovalRateResult` for one solver comparison and removal rate.
- Added `DifficultyEvaluationResult` for the complete experiment.
- Added reusable evaluation across multiple removal rates.
- Trained a separate Random Forest for each evaluated removal rate.
- Used identical puzzle sets for hybrid and classical comparisons at each rate.
- Added valid-solution rates, runtime ratios, and backtrack reduction.
- Added average ML decisions per puzzle.
- Added parameter validation and integration tests.
- Added a command-line experiment with two readable result tables.

### Usage

Run the experiment from the project root:

```bash
python scripts/evaluate_difficulty_levels.py
```

### Evaluation Configuration

```text
Removal rates: 0.50, 0.60, 0.65, 0.70
Training solutions per rate: 100
Training/test split: 80% / 20%
Evaluation puzzles per rate: 20
Random Forest estimators: 100
Training random seed: 42
Evaluation random seed: 123
```

### Solution Quality and Runtime

| Removal rate | Hybrid valid | Classical valid | Hybrid runtime | Classical runtime | Runtime ratio |
|---:|---:|---:|---:|---:|---:|
| 50% | 100.00% | 100.00% | 19.62 ms | 0.96 ms | 20.35x |
| 60% | 100.00% | 100.00% | 72.21 ms | 2.06 ms | 34.98x |
| 65% | 100.00% | 100.00% | 138.21 ms | 4.28 ms | 32.27x |
| 70% | 100.00% | 100.00% | 376.30 ms | 8.93 ms | 42.12x |

### Search Effort

Backtracks and ML decisions are reported as averages per puzzle.

| Removal rate | Hybrid backtracks | Classical backtracks | Reduction | Hybrid ML decisions |
|---:|---:|---:|---:|---:|
| 50% | 0.00 | 0.00 | 0.00% | 1.30 |
| 60% | 6.65 | 6.90 | 3.62% | 4.80 |
| 65% | 24.70 | 57.25 | 56.86% | 9.30 |
| 70% | 138.40 | 196.55 | 29.59% | 25.50 |

### Interpretation

Both solvers produced valid solutions for every evaluated puzzle. The number of ML decisions and the search effort increased substantially as more cells were removed.

The largest relative backtrack reduction occurred at a removal rate of 0.65, where ML guidance reduced backtracking by **56.86%**. At 0.70, the hybrid solver still reduced backtracking by **29.59%**, but the relative benefit did not increase monotonically with the removal rate.

The hybrid solver was between approximately **20 and 42 times slower** across the experiment. Feature generation and Random Forest inference therefore cost more time than the additional search performed by the classical solver.

The experiment strengthens the conclusion from Commit 17:

```text
ML guidance -> fewer failed search paths at some removal rates
Classical   -> consistently lower end-to-end runtime
```

### Testing

Result:

`85 passed`

### Limitations

- Only 20 puzzles were evaluated per removal rate.
- Results use one training seed and one evaluation seed.
- Removal rate is not a formal measure of Sudoku difficulty.
- Generated puzzles are not checked for a unique solution.

### Next Step

Repeat the experiment across multiple seeds and report mean values and variability, or add formal puzzle-generation and uniqueness constraints before making stronger difficulty claims.

---
## Commit 19 - Model Persistence

**Commit:** `feat: add model persistence`

### Objective

Save a trained Random Forest and restore it in a later process so that solver tools do not need to retrain the model on every invocation.

### Approach

The fitted scikit-learn `RandomForestClassifier` is serialized with joblib. Loading creates a new `SudokuRandomForest` wrapper around the restored classifier.

```text
Generate training data
        |
        v
Train Random Forest
        |
        v
Save joblib artifact
        |
        v
Load model later
        |
        v
Predict without retraining
```

### Implemented

- Added joblib as an explicit project dependency.
- Added `SudokuRandomForest.save()`.
- Added `SudokuRandomForest.load()`.
- Created missing parent directories when saving a model.
- Validated that loaded files contain a `RandomForestClassifier`.
- Added a training script that reports hold-out accuracy and saves the model.
- Excluded generated `.joblib` artifacts and Python package metadata from Git.
- Added round-trip and invalid-object tests.
- Added a narrowly scoped pytest filter for a third-party joblib deprecation warning with NumPy 2.5.

### Usage

Train and save the default model:

```bash
python scripts/train_model.py
```

The generated artifact is stored at:

```text
models/sudoku_random_forest.joblib
```

Load the trained model:

```python
from sudoku_ml.model.random_forest import SudokuRandomForest

model = SudokuRandomForest.load(
    "models/sudoku_random_forest.joblib"
)
```

### Verification

The persistence test compares predictions before saving with predictions after loading. It also verifies that the learned classes remain unchanged:

```text
[1 2 3 4 5 6 7 8 9]
```

Result:

`87 passed`

### Artifact Management

Trained model files are local generated artifacts and are excluded through `.gitignore`:

```gitignore
# Trained model artifacts
models/*.joblib
```

Editable-install metadata is excluded separately:

```gitignore
# Python packaging metadata
*.egg-info/
```

### Security

Joblib files must only be loaded from trusted sources. Joblib relies on pickle-compatible deserialization, which can execute arbitrary code contained in a malicious file.

### Conclusion

The trained Random Forest can now be reused across processes without repeating dataset generation and model training. This provides the foundation for a command-line interface that solves user-provided Sudoku grids with a saved model.

---
## Commit 20 - Command-Line Sudoku Solver

**Commit:** `feat: add command-line Sudoku solver`

### Objective

Provide a user-facing command-line interface that loads a saved model and solves a structured Sudoku string without requiring changes to Python source code.

### Input Format

The CLI accepts exactly 81 Sudoku cells after whitespace normalization:

```text
Digits 1-9 -> given Sudoku values
0          -> empty cell
.          -> empty cell
Whitespace -> ignored
```

Example input:

```text
530070000600195000098000060800060003400803001700020006060000280000419005000080079
```

### Pipeline

```text
Command-line arguments
        |
        v
Parse 81-cell grid
        |
        v
Load joblib model
        |
        v
Run hybrid solver
        |
        v
Print solution and statistics
```

### Implemented

- Added parsing for compact and whitespace-separated Sudoku input.
- Accepted both `0` and `.` as empty-cell markers.
- Added validation for input length and allowed characters.
- Added terminal-friendly 3x3 block formatting.
- Added a default saved-model path.
- Added an optional `--model` argument.
- Connected model loading to `HybridSudokuSolver`.
- Printed deterministic steps, ML decisions, and backtracks.
- Reported invalid puzzles and missing models as command-line errors.
- Added parser, formatter, successful-execution, and error-path tests.

### Usage

Train the default model if it does not exist yet:

```bash
python scripts/train_model.py
```

Solve a Sudoku:

```bash
python -m sudoku_ml.cli "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

Specify a different model path:

```bash
python -m sudoku_ml.cli --model models/sudoku_random_forest.joblib "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

### Example Result

The example puzzle was solved as:

```text
5 3 4 | 6 7 8 | 9 1 2
6 7 2 | 1 9 5 | 3 4 8
1 9 8 | 3 4 2 | 5 6 7
------+-------+------
8 5 9 | 7 6 1 | 4 2 3
4 2 6 | 8 5 3 | 7 9 1
7 1 3 | 9 2 4 | 8 5 6
------+-------+------
9 6 1 | 5 3 7 | 2 8 4
2 8 7 | 4 1 9 | 6 3 5
3 4 5 | 2 8 6 | 1 7 9
```

Solver statistics:

```text
Deterministic steps: 51
ML decisions:        0
Branching decisions: 0
Backtracks:          0
```

This puzzle is solved entirely through single-candidate constraint propagation, so the saved model is loaded but does not need to rank an ambiguous cell.

### Testing

The CLI tests cover:

- compact input with zeros,
- dots as empty cells,
- ignored whitespace,
- invalid length,
- invalid characters,
- readable grid formatting,
- successful solving with a saved model,
- missing model files,
- command-line exit codes and error output.

Result:

`96 passed`

### Conclusion

The project now provides a complete user-facing path from a saved machine-learning model and an 81-cell Sudoku string to a valid formatted solution. The CLI also exposes solver statistics, making it suitable for both demonstration and further experiments.

---
## Commit 21 - Unique-Solution Puzzle Generation

**Commit:** `feat: add unique-solution puzzle generation`

### Objective

Generate Sudoku puzzles with exactly one valid solution so that evaluation targets are unambiguous and solver results can be compared with a unique ground truth.

### Motivation

The original synthetic dataset generator removes randomly selected cells from complete Sudoku solutions. This preserves validity of the clues but does not guarantee that the incomplete puzzle has only one solution.

A solver can therefore produce a valid completion that differs from the stored source solution. Unique-solution generation removes this ambiguity and creates a stronger foundation for future solver evaluation and difficulty analysis.

### Solution Counting

The solution counter uses deterministic backtracking with minimum-remaining-values cell selection:

```text
Select the empty cell with fewest candidates
                  |
                  v
          Try valid candidates
                  |
                  v
          Count completed grids
                  |
                  v
        Stop when limit is reached
```

For uniqueness checks, the limit is two:

```text
0 solutions -> unsolvable
1 solution  -> unique
2 solutions -> non-unique, stop searching
```

Stopping after the second solution avoids enumerating every possible completion of an underconstrained puzzle.

### Unique Puzzle Generation

The generator starts with a complete valid Sudoku and considers cells in a seeded random order. For each cell:

1. Remove the clue temporarily.
2. Count solutions up to two.
3. Keep the removal if exactly one solution remains.
4. Restore the clue if multiple solutions appear.
5. Stop when the requested number of removals is reached.

If the requested removal rate cannot be reached while preserving uniqueness, the generator raises an error rather than silently returning fewer empty cells.

### Implemented

- Added bounded Sudoku solution counting.
- Added `has_unique_solution()`.
- Added minimum-remaining-values selection for the counting search.
- Added early termination at a configurable solution limit.
- Preserved input grids during solution counting.
- Added reproducible uniqueness-preserving clue removal.
- Added correct handling for a zero removal rate.
- Added diverse unique-dataset generation.
- Preserved complete solutions as ground truth.
- Added validation for source grids, removal rates, solution limits, and sample counts.

### Usage

Generate one unique puzzle from a known solution:

```python
from sudoku_ml.dataset.unique_generator import create_unique_puzzle
from sudoku_ml.sudoku_generator import generate_solved_grid

solution = generate_solved_grid(random_seed=42)
puzzle = create_unique_puzzle(
    solution,
    removal_rate=0.5,
    random_seed=42,
)
```

Generate a diverse dataset:

```python
from sudoku_ml.dataset.unique_generator import create_unique_dataset

dataset = create_unique_dataset(
    num_samples=100,
    removal_rate=0.5,
    random_seed=42,
)
```

### Testing

The tests cover:

- a known puzzle with one solution,
- early termination for a puzzle with multiple solutions,
- invalid grids with zero solutions,
- input immutability,
- invalid solution limits,
- exact requested removal counts,
- uniqueness after clue removal,
- zero and invalid removal rates,
- reproducible puzzle and dataset generation,
- source-grid immutability,
- invalid source grids and sample counts,
- valid complete ground-truth solutions.

Result:

`112 passed`

### Current Scope

The unique generator is available as an alternative dataset source. Existing baseline and removal-rate evaluation functions still use the original faster random-removal generator. Replacing their dataset source should be performed as a separate experiment because uniqueness checks add substantial generation cost and change the evaluated data distribution.

### Conclusion

The project can now distinguish unsolvable, uniquely solvable, and non-unique Sudoku puzzles. It can also generate reproducible datasets whose puzzles have exactly one solution, enabling stronger end-to-end evaluation in the next commit.

---
## Commit 22 - Evaluation on Unique-Solution Puzzles

**Commit:** `feat: evaluate solvers on unique-solution puzzles`

### Objective

Evaluate the hybrid and classical solvers on puzzles with exactly one solution and verify that each produced grid matches the unique stored ground truth.

### Motivation

Previous evaluations counted a completed grid as successful when it was valid and preserved all clues. For a puzzle with multiple solutions, that grid could differ from the source solution while still being correct.

Unique-solution evaluation strengthens the success criterion:

```text
Valid completion
      +
Preserves all clues
      +
Exactly matches unique ground truth
      =
Verified solution
```

### Evaluation Extension

`evaluate_solver()` now accepts optional expected solution grids. When supplied, it records:

- the number of exact solution matches,
- the exact matching-solution rate.

The argument is optional, so existing baseline, difficulty, and comparison evaluations remain compatible. A length check prevents puzzles and expected solutions from being paired incorrectly.

`compare_solvers()` forwards the same optional ground truth to both solver evaluations, preserving a fair comparison.

### Unique Evaluation Pipeline

For every removal rate:

1. Generate the existing random-removal training split.
2. Train a Random Forest for that removal rate.
3. Generate a separate unique-solution evaluation dataset.
4. Solve the same puzzles with hybrid and classical ordering.
5. Compare each result with the unique ground truth.
6. Report runtime, search effort, and exact match rates.

Training remains on the original faster random-removal dataset. Only evaluation uses uniqueness-preserving generation in this commit. This isolates the effect of a stronger evaluation set without simultaneously changing the training distribution.

### Implemented

- Added optional expected solutions to the reusable solver evaluation.
- Added exact solution-match counts and rates.
- Added validation for mismatched puzzle and solution collections.
- Extended solver comparison to forward shared ground truth.
- Added evaluation across multiple unique-puzzle removal rates.
- Added a command-line experiment for unique-solution evaluation.
- Added integration and parameter-validation tests.

### Usage

Run the experiment from the project root:

```bash
python scripts/evaluate_unique_solvers.py
```

### Evaluation Configuration

```text
Removal rates: 0.50, 0.60, 0.65
Training solutions per rate: 100
Evaluation puzzles per rate: 10
Random Forest estimators: 100
Training random seed: 42
Evaluation random seed: 123
Training data: random-removal puzzles
Evaluation data: unique-solution puzzles
```

### Solution Quality and Runtime

| Removal rate | Hybrid match | Classical match | Hybrid runtime | Classical runtime | Runtime ratio |
|---:|---:|---:|---:|---:|---:|
| 50% | 100.00% | 100.00% | 0.66 ms | 0.68 ms | 0.97x |
| 60% | 100.00% | 100.00% | 12.32 ms | 1.64 ms | 7.51x |
| 65% | 100.00% | 100.00% | 52.14 ms | 3.54 ms | 14.74x |

### Search Effort

Backtracks and ML decisions are averages per puzzle.

| Removal rate | Hybrid backtracks | Classical backtracks | Reduction | Hybrid ML decisions |
|---:|---:|---:|---:|---:|
| 50% | 0.00 | 0.00 | 0.00% | 0.00 |
| 60% | 7.10 | 6.60 | -7.58% | 0.70 |
| 65% | 27.00 | 52.70 | 48.77% | 3.30 |

### Interpretation

Both solvers exactly matched the unique ground truth for every evaluated puzzle. This confirms that the previous distinction between a merely valid completion and the stored solution has been removed for this evaluation set.

At a removal rate of 0.50, all puzzles were solved deterministically and the model was not used. At 0.60, ML guidance slightly increased average backtracking from 6.60 to 7.10, corresponding to a negative reduction of 7.58%. At 0.65, ML guidance reduced average backtracking from 52.70 to 27.00, an improvement of 48.77%.

The result reinforces that ML candidate ordering is not uniformly beneficial. Its value depends on the puzzle distribution and search state. The classical solver remains faster at the rates where ML inference is used.

The near-equal sub-millisecond runtimes at 0.50 should not be interpreted as a meaningful hybrid speed advantage because no ML decisions occur and such short measurements are sensitive to timing noise.

### Testing

The additional tests cover:

- exact ground-truth matching,
- backward-compatible evaluation without ground truth,
- mismatched puzzle and solution lengths,
- unique evaluation integration,
- empty and invalid removal-rate collections,
- invalid evaluation puzzle counts.

Result:

`120 passed`

### Limitations

- Only 10 unique puzzles are evaluated per removal rate.
- Results use one training seed and one evaluation seed.
- Training data is not uniqueness-preserving.
- Removal rate remains a proxy rather than a formal difficulty measure.
- Runtime measurements are single-run values.

### Conclusion

The evaluation pipeline can now verify exact solver output against an unambiguous ground truth. Both solvers achieved a 100% exact match rate in the current experiment, while the search-effort results show that ML guidance can help substantially at some removal rates but can also be neutral or slightly harmful at others.

---
## Commit 23 - Repeated Evaluation Across Seeds

**Commit:** `feat: add repeated solver evaluation across seeds`

### Objective

Measure how strongly solver results depend on the randomly generated evaluation puzzle set instead of drawing conclusions from a single evaluation seed.

### Experimental Design

For each removal rate, the experiment:

1. Creates one deterministic training split.
2. Trains one Random Forest.
3. Generates three independent unique-puzzle evaluation sets.
4. Evaluates both solvers on each set.
5. Aggregates metrics across runs.

The trained model remains fixed across evaluation seeds. This isolates evaluation-sample variability and avoids conflating it with variation from model training.

### Summary Statistics

`MetricSummary` records:

- arithmetic mean,
- population standard deviation,
- minimum,
- maximum.

Population standard deviation describes the spread of the selected runs. It is not a confidence interval and should not be interpreted as inferential statistical evidence.

### Implemented

- Added reusable metric summaries.
- Added repeated result structures by removal rate.
- Added repeated evaluation across configurable seeds.
- Reused one trained model per removal rate.
- Generated independent unique-solution datasets for every run.
- Aggregated exact match rate, runtime, runtime ratio, backtracks, backtrack reduction, and ML decisions.
- Added validation for rates, seeds, metric values, and puzzle counts.
- Added a terminal report using mean and population standard deviation.
- Added unit and integration tests.

### Usage

Run the repeated experiment from the project root:

```bash
python scripts/evaluate_repeated_solvers.py
```

### Evaluation Configuration

```text
Removal rates: 0.50, 0.60, 0.65
Evaluation seeds: 101, 123, 202
Runs per removal rate: 3
Training solutions per rate: 100
Evaluation puzzles per run: 10
Random Forest estimators: 100
Training random seed: 42
Evaluation data: unique-solution puzzles
```

### Solution Match and Runtime

Values are mean plus or minus population standard deviation.

| Removal rate | Hybrid match | Classical match | Hybrid runtime | Classical runtime | Runtime ratio |
|---:|---:|---:|---:|---:|---:|
| 50% | 100.00% +/- 0.00 | 100.00% +/- 0.00 | 0.65 +/- 0.04 ms | 0.63 +/- 0.02 ms | 1.03 +/- 0.02 |
| 60% | 100.00% +/- 0.00 | 100.00% +/- 0.00 | 20.30 +/- 8.13 ms | 1.76 +/- 0.26 ms | 11.22 +/- 3.14 |
| 65% | 100.00% +/- 0.00 | 100.00% +/- 0.00 | 57.75 +/- 3.70 ms | 3.94 +/- 0.43 ms | 14.77 +/- 1.40 |

### Search Effort

| Removal rate | Hybrid backtracks | Classical backtracks | Reduction | ML decisions |
|---:|---:|---:|---:|---:|
| 50% | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00% +/- 0.00 | 0.00 +/- 0.00 |
| 60% | 10.73 +/- 7.29 | 11.87 +/- 6.42 | 13.52% +/- 24.68 | 1.27 +/- 0.54 |
| 65% | 29.20 +/- 6.56 | 58.33 +/- 13.15 | 45.39% +/- 22.08 | 3.60 +/- 0.24 |

### Interpretation

Both solvers maintained a 100% exact match rate across all repeated runs.

At 50%, no ambiguous candidate decisions were required. Runtime differences at this scale are measurement noise rather than meaningful solver performance differences.

At 60%, the mean backtrack reduction was positive at 13.52%, but its standard deviation was 24.68 percentage points. This high variability explains why the single-seed experiment in Commit 22 showed a negative result. Candidate-ordering benefit at this rate depends strongly on the sampled puzzles.

At 65%, the hybrid solver reduced backtracking by 45.39% on average. The 22.08-point standard deviation remains substantial, but the aggregate result is more consistently favorable than at 60%.

The classical solver remained substantially faster whenever ML inference was used. Reduced search effort does not offset repeated feature generation and Random Forest prediction costs.

### Testing

The new tests cover:

- exact summary-statistic calculation,
- rejection of empty metric collections,
- repeated integration across two seeds,
- result and seed preservation,
- empty and invalid removal-rate collections,
- empty evaluation-seed collections,
- invalid evaluation puzzle counts.

Result:

`130 passed`

### Limitations

- Only three evaluation seeds are used.
- Each run contains only 10 puzzles per removal rate.
- Population standard deviation is descriptive and not a confidence interval.
- The training seed and trained model remain fixed.
- Removal rate is not a formal Sudoku difficulty measure.
- Runtime is sensitive to local system load.

### Conclusion

Repeated evaluation reveals that single-seed backtracking results can be unstable, especially at a removal rate of 0.60. The average 0.65 result still supports a useful ML search-ordering effect, while runtime measurements consistently favor the classical solver.

---
## Commit 24 - Heuristic Puzzle Difficulty Analysis

**Commit:** `feat: add heuristic puzzle difficulty analysis`

### Objective

Describe Sudoku difficulty with reproducible structural and search-effort metrics instead of treating removal rate as the difficulty value itself.

### Motivation

Removal rate determines the number of empty cells, but puzzles with the same number of clues can require very different amounts of constraint propagation and backtracking. A useful project-level analysis should therefore observe the actual solving process.

The analyzer deliberately uses the classical solver. ML guidance would change the explored search path and make the difficulty measurement dependent on the trained model.

### Structural Metrics

The analysis records the puzzle state before solving:

- number of given cells,
- number of empty cells,
- number of cells with one initial candidate,
- average initial candidate count across empty cells.

### Search Metrics

`SolverStats` now distinguishes generic branching decisions from ML decisions:

```text
deterministic step -> exactly one candidate
branching decision -> multiple valid candidates
ML decision        -> model ranks a branching decision
backtrack          -> attempted value is reversed
```

The hybrid solver increments both `branching_decisions` and `ml_decisions` when it ranks an ambiguous cell. The classical solver increments only `branching_decisions`.

### Heuristic Score

The project-specific score is:

```text
difficulty score = branching decisions + backtracks
```

Levels are assigned using transparent thresholds:

| Level | Score |
|---|---:|
| Easy | 0 |
| Medium | 1-10 |
| Hard | 11-100 |
| Expert | above 100 |

These levels are not official Sudoku ratings. They approximate computational search effort and do not model the named logical techniques used by human solvers.

### Implemented

- Added generic branching-decision statistics to both solvers.
- Added branching output to the command-line solver.
- Added `DifficultyLevel`.
- Added per-puzzle structural and search analysis.
- Added project-specific score classification.
- Required valid uniquely solvable puzzles for analysis.
- Preserved input puzzles during analysis.
- Added dataset-level averages and level distributions.
- Added a command-line analysis experiment.
- Added classification, validation, immutability, and aggregation tests.

### Usage

Run the analysis from the project root:

```bash
python scripts/analyze_puzzle_difficulty.py
```

### Evaluation Configuration

```text
Removal rates: 0.50, 0.60, 0.65
Unique puzzles per rate: 10
Puzzle random seed: 123
Solver: classical MRV backtracking
```

### Average Difficulty Metrics

| Removal rate | Clues | Initial singles | Avg candidates | Branches | Backtracks | Score |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 41.00 | 8.20 | 2.40 | 0.00 | 0.00 | 0.00 |
| 60% | 33.00 | 3.90 | 2.99 | 0.70 | 6.60 | 7.30 |
| 65% | 29.00 | 2.50 | 3.37 | 5.80 | 52.70 | 58.50 |

### Heuristic Level Distribution

| Removal rate | Easy | Medium | Hard | Expert |
|---:|---:|---:|---:|---:|
| 50% | 10 | 0 | 0 | 0 |
| 60% | 6 | 1 | 3 | 0 |
| 65% | 2 | 3 | 2 | 3 |

### Interpretation

All 50% puzzles were solved without branching or backtracking and were classified as easy. At 60%, the same removal count produced a mixture of easy, medium, and hard puzzles. At 65%, all four heuristic levels appeared, including three expert puzzles.

The average initial candidate count rose from 2.40 to 3.37 as clues were removed, while the number of initial single candidates decreased from 8.20 to 2.50. This connects structural ambiguity with increasing classical search effort.

The mixed level distributions demonstrate why removal rate should not be presented as a formal difficulty rating. It influences difficulty but does not determine it.

### Testing

The new tests cover:

- all heuristic threshold boundaries,
- rejection of negative metrics,
- exact metrics for a one-cell puzzle,
- puzzle immutability,
- rejection of non-unique puzzles,
- dataset summary averages,
- level counts,
- rejection of empty puzzle collections,
- CLI branching-statistics output.

Result:

`142 passed`

### Limitations

- Thresholds are project-defined and empirical.
- Difficulty is measured through one classical search strategy.
- Named human-solving techniques are not detected.
- Search order affects branching and backtrack counts.
- Dataset results use only 10 puzzles per removal rate.

### Conclusion

The project now separates clue removal from measured search difficulty. Structural candidate metrics and classical solver effort provide a more informative, reproducible profile while remaining explicit about the limits of the heuristic rating.

---
## Commit 25 - Installable Command-Line Interface

**Commit:** `feat: add installable command-line interface`

### Objective

Turn the existing module-based CLI into a convenient installed command and improve input, solver-selection, version, and error-handling behavior while preserving backward compatibility.

### Package Entry Point

The project metadata now registers:

```toml
[project.scripts]
sudoku-ml = "sudoku_ml.cli:main"
```

After an editable install, the command is available inside the active virtual environment:

```bash
python -m pip install -e ".[dev]"
sudoku-ml --version
```

Verified version output:

```text
ml-sudoku-solver 0.1.0
```

### Input Sources

The CLI supports two mutually exclusive input methods:

```text
Direct argument -> sudoku-ml "81-cell puzzle"
Text file       -> sudoku-ml --input-file puzzle.txt
```

Text files are read as UTF-8. Existing normalization continues to ignore whitespace and accepts both `0` and `.` for empty cells.

Argparse requires exactly one input source. Missing input and conflicting direct/file input produce exit code 2 with usage information.

### Solver Modes

The default mode loads the saved Random Forest and creates `HybridSudokuSolver`:

```bash
sudoku-ml "81-cell puzzle"
```

The classical mode does not require a model file:

```bash
sudoku-ml --classical "81-cell puzzle"
```

The selected solver name is included in terminal statistics so that output remains self-describing.

### Version Handling

`--version` reads the installed distribution version through `importlib.metadata`. If package metadata is unavailable, the CLI reports `development` instead of failing.

The printed name is fixed to `ml-sudoku-solver`, making version output consistent across the console command, module execution, and direct function tests.

### Implemented

- Added the `sudoku-ml` project entry point.
- Preserved `python -m sudoku_ml.cli` compatibility.
- Added UTF-8 puzzle-file input.
- Made direct and file input mutually exclusive and required.
- Added classical solving without a model dependency.
- Added solver-mode output.
- Added installed-package version reporting.
- Expanded file and argument error handling.
- Removed duplicate CLI test imports.
- Documented editable installation for Git Bash on Windows.

### Usage

Hybrid solver with the default model:

```bash
sudoku-ml "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

Classical solver:

```bash
sudoku-ml --classical "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

File input:

```bash
sudoku-ml --classical --input-file puzzle.txt
```

Help and version:

```bash
sudoku-ml --help
sudoku-ml --version
```

### Verification

The installed entry point was verified for:

- version output,
- generated help text,
- a complete classical solve invocation,
- formatted solution output,
- solver statistics.

### Testing

The new tests cover:

- classical mode without a model file,
- successful UTF-8 file input,
- missing input files,
- required input selection,
- conflicting direct and file inputs,
- stable version output,
- selected solver output.

Result:

`148 passed`

### Compatibility

Existing direct string input, `--model`, parser behavior, formatted output, and `python -m sudoku_ml.cli` remain supported.

### Conclusion

The project now behaves like an installable Python application rather than only a source-tree module. Users can solve puzzles through a stable `sudoku-ml` command, choose ML-guided or classical solving, read puzzles from files, and inspect package version information.

---
## Commit 26 - Greedy ML Solver Evaluation

**Commit:** `feat: add greedy ML solver evaluation`

### Objective

Measure how often the current Random Forest can guide a complete Sudoku solve when incorrect model decisions cannot be repaired by backtracking.

### Motivation

The Hybrid solver reaches a 100% solution rate because it combines model ranking with deterministic constraints and recursive backtracking. This makes the complete system reliable but does not reveal whether the model can produce a fully correct sequence of decisions.

Cell-level accuracy is also insufficient for answering that question. A single wrong decision can invalidate an otherwise correct sequence:

```text
Correct first decisions
        |
        v
One wrong ML choice
        |
        v
Later contradiction
        |
        v
Entire Greedy solve fails
```

### Greedy Search Strategy

`GreedyMLSudokuSolver` inherits the Hybrid solver's:

- puzzle validation,
- minimum-remaining-values cell selection,
- valid-candidate filtering,
- 118-feature representation,
- Random Forest probability ranking,
- solver statistics.

It overrides the recursive search loop. When several candidates are possible, only the highest-ranked valid candidate is placed. The decision is permanent. A later cell with no valid candidate ends the solve and returns no solution.

The Greedy solver is constraint-aware rather than fully model-only. Sudoku constraints still prevent immediately illegal digits, but no search mechanism corrects a locally plausible model mistake.

### Comparison Design

Greedy and Hybrid ML use the same:

- trained Random Forest,
- unique-solution evaluation puzzles,
- expected ground-truth grids,
- cell-selection rule,
- candidate constraints,
- feature calculation and probability ranking.

The only experimental difference is backtracking.

`recovered_puzzles` counts exact Hybrid matches minus exact Greedy matches. `greedy_failure_rate` is one minus the Greedy exact matching-solution rate.

### Implemented

- Added `GreedyMLSudokuSolver` without backtracking.
- Preserved deterministic single-candidate solving.
- Added reproducible tests for deterministic success and unrecoverable ML failure.
- Added reusable Greedy-vs-Hybrid comparison results.
- Added exact match and recovery metrics.
- Added evaluation across multiple removal rates.
- Added parameter and ground-truth validation tests.
- Added a command-line experiment.

### Usage

Run the experiment from the project root:

```bash
python scripts/evaluate_greedy_solver.py
```

### Evaluation Configuration

```text
Removal rates: 0.50, 0.60, 0.65
Training solutions per rate: 100
Evaluation puzzles per rate: 20
Random Forest estimators: 100
Training random seed: 42
Evaluation random seed: 123
Evaluation data: unique-solution puzzles
```

### Exact Solution Performance

| Removal rate | Greedy exact match | Hybrid exact match | Greedy failure | Recovered by Hybrid |
|---:|---:|---:|---:|---:|
| 50% | 100.00% | 100.00% | 0.00% | 0 |
| 60% | 55.00% | 100.00% | 45.00% | 9 |
| 65% | 25.00% | 100.00% | 75.00% | 15 |

### Runtime and Search Effort

Values are averages per puzzle.

| Removal rate | Greedy runtime | Hybrid runtime | Greedy ML decisions | Hybrid ML decisions | Hybrid backtracks |
|---:|---:|---:|---:|---:|---:|
| 50% | 0.72 ms | 0.71 ms | 0.00 | 0.00 | 0.00 |
| 60% | 14.39 ms | 17.56 ms | 0.90 | 1.10 | 11.80 |
| 65% | 34.07 ms | 60.24 ms | 2.15 | 3.95 | 38.85 |

### Interpretation

At 50% removal, all puzzles were solved deterministically. Neither solver made an ML decision, so the 100% Greedy rate is not evidence of model capability.

At 60%, Greedy ML solved 11 of 20 puzzles exactly. Hybrid backtracking recovered the remaining nine and reached 100%.

At 65%, Greedy ML solved only five of 20 puzzles. Fifteen decision chains reached a contradiction and required Hybrid backtracking for recovery.

Greedy runtime is lower because failed paths are abandoned rather than explored and corrected. The speed difference therefore represents less work, not superior solving quality.

The results answer the model-focused research question:

> The Random Forest provides useful local ranking information but cannot yet produce a reliable complete decision sequence for ambiguous Sudoku puzzles.

### Testing

The new tests cover:

- deterministic Greedy success,
- a reproducible wrong-choice failure,
- zero Greedy backtracks,
- successful Hybrid recovery,
- exact match and failure-rate calculations,
- multiple removal-rate evaluation,
- invalid rates and puzzle counts,
- mismatched ground-truth collections.

Result:

`159 passed`

### Limitations

- Only 20 puzzles are evaluated per removal rate.
- One training seed and one evaluation seed are used.
- Greedy remains constraint-aware and is not a pure end-to-end model.
- Average ML-decision counts are lower for Greedy because failed solves stop early.
- No top-k, ranking, or calibration metrics are reported yet.

### Next Step

Analyze the full probability ranking with top-k accuracy, mean reciprocal rank, confidence, and calibration metrics. This will show whether failed Greedy choices often contain the correct digit near the top even when top-1 is wrong.

### Conclusion

Backtracking is currently essential for reliable complete solving. The Greedy experiment isolates the limitation of the current Random Forest and establishes a stronger baseline for the next model-analysis phase.

---
## Commit 27 - Model Probability-Ranking Analysis

**Commit:** `feat: add model probability ranking analysis`

### Objective

Analyze the Random Forest's complete probability distribution instead of evaluating only its Top-1 prediction.

### Motivation

The Greedy solver showed that one incorrect model decision can invalidate a complete solve, while Hybrid backtracking can recover by trying alternatives. Top-1 accuracy alone does not show whether the correct digit was a close second choice or ranked near the bottom.

Probability-ranking metrics make this distinction visible:

```text
Predicted digit probabilities
            |
            v
Sort digits by probability
            |
            v
Find rank of correct digit
            |
      +-----+-----+
      |     |     |
      v     v     v
    Top-k  MRR  Calibration
```

### Implemented

- Added `ProbabilityRankingResult` for reusable ranking and calibration metrics.
- Added Top-1, Top-2, and Top-3 accuracy.
- Added mean reciprocal rank.
- Added mean prediction confidence.
- Added expected calibration error with configurable confidence bins.
- Added multiclass log loss.
- Added candidate masking and probability renormalization.
- Added raw and candidate-constrained analysis modes.
- Added a command-line evaluation script.
- Added unit tests for metrics, validation, masking, and renormalization.

### Candidate-Constrained Probabilities

The Random Forest produces probabilities for all learned digit classes. The constrained mode uses candidate indicators from the input features to assign zero probability to digits that violate the current Sudoku row, column, or block constraints. The remaining probabilities are divided by their sum.

```text
Raw probabilities
        |
        v
Mask illegal digits
        |
        v
Renormalize valid digits
        |
        v
Constrained probabilities
```

The candidate mask is derived only from the incomplete puzzle. It does not use the target digit from the completed solution.

### Metrics

- Top-k accuracy reports whether the correct digit appears among the first `k` ranked classes.
- Mean reciprocal rank rewards predictions that place the correct digit near the beginning of the ranking.
- Mean confidence is the average probability assigned to the predicted Top-1 digit.
- Expected calibration error compares confidence with observed Top-1 accuracy across confidence bins.
- Log loss evaluates the probability assigned to the correct class and penalizes confident mistakes.

Higher values are better for Top-k accuracy and mean reciprocal rank. Lower values are better for expected calibration error and log loss.

### Usage

Run the analysis from the project root:

```bash
python scripts/evaluate_probability_ranking.py
```

### Evaluation Configuration

```text
Generated Sudoku solutions: 100
Training/test split: 80% / 20%
Removal rate: 0.50
Random Forest estimators: 100
Random seed: 42
Evaluation samples: 800
Calibration bins: 10
```

### Results

| Metric | Raw | Candidate-constrained |
|---|---:|---:|
| Top-1 accuracy | 66.88% | 66.88% |
| Top-2 accuracy | 90.00% | 90.00% |
| Top-3 accuracy | 97.50% | 97.50% |
| Mean reciprocal rank | 0.8152 | 0.8153 |
| Mean confidence | 36.28% | 59.20% |
| Expected calibration error | 0.3093 | 0.0801 |
| Log loss | 1.1396 | 0.6960 |

### Interpretation

The correct digit is ranked first for 66.88% of hold-out samples, among the first two for 90.00%, and among the first three for 97.50%. The model therefore provides considerably more useful information in its complete ranking than Top-1 accuracy alone reveals.

Candidate constraints leave Top-1 through Top-3 accuracy unchanged and improve mean reciprocal rank only minimally. This indicates that illegal digits rarely occupy the leading ranking positions in this test set.

The larger effect is probabilistic. Removing illegal digits raises mean confidence from 36.28% to 59.20%, reduces expected calibration error from 0.3093 to 0.0801, and reduces log loss from 1.1396 to 0.6960. The constrained probabilities represent the valid Sudoku decision space much better than the raw classifier output.

These results explain why Hybrid ML can benefit from the model even though Greedy ML remains unreliable. When the first choice is wrong, the correct digit is usually the second or third candidate and can be reached through backtracking.

### Testing

The tests cover:

- perfect and imperfect probability rankings,
- Top-k accuracy and mean reciprocal rank,
- expected calibration error,
- multiclass log loss,
- empty and mismatched input validation,
- targets absent from model classes,
- invalid calibration-bin counts,
- candidate-feature validation,
- candidate masking and renormalization,
- correction of an invalid raw Top-1 prediction.

Result:

`167 passed`

### Limitations

- The experiment uses one training/test split and one random seed.
- Only a removal rate of 0.50 is evaluated.
- Calibration results depend on the selected number of confidence bins.
- Candidate constraints encode domain knowledge and are not learned solely by the Random Forest.
- Cell-level ranking quality does not directly equal complete-puzzle solution quality.

### Next Step

Perform a feature-ablation experiment that trains and evaluates the model without explicit candidate features. This will quantify how much of the ranking performance comes from learned grid patterns and how much comes from encoded Sudoku constraints.

### Conclusion

The model's probability ranking is substantially more informative than its Top-1 accuracy. Sudoku candidate constraints mainly improve probability quality rather than leading rank accuracy, while the strong Top-2 and Top-3 results provide a clear explanation for the effectiveness of ML-guided backtracking.

---
## Commit 28 - Feature-Ablation Analysis

**Commit:** `feat: add feature ablation analysis`

### Objective

Measure how much each feature group contributes to the Random Forest's ranking performance and probability quality.

### Motivation

The complete 118-feature model reaches 66.88% Top-1 accuracy, but this result does not reveal whether the model learned Sudoku structure from the raw grid or mainly benefits from explicitly engineered constraint features.

A controlled ablation progressively exposes the model to more information:

```text
Grid + target position
          |
          v
Add candidate indicators
          |
          v
Add candidate interactions
```

Each configuration is trained as a separate model. This prevents removed feature groups from influencing the result and isolates their incremental contribution.

### Feature Configurations

| Configuration | Included features | Count |
|---|---|---:|
| Grid and position | Flattened grid and target cell index | 82 |
| Candidate indicators | Previous features plus nine valid-candidate flags | 91 |
| Candidate interactions | Previous features plus row, column, and block candidate counts | 118 |

The configuration names describe the newest feature group. Every later configuration includes all preceding groups.

### Implemented

- Added immutable feature-configuration definitions.
- Added validated feature-prefix selection.
- Added `FeatureAblationResult` for one trained configuration.
- Added reusable training and evaluation across all feature groups.
- Reused the probability-ranking metrics from Commit 27.
- Added a command-line experiment with ranking and probability-quality tables.
- Added unit tests for configuration, selection, validation, and evaluation.

### Experimental Design

All three Random Forest models use identical:

- training and test Sudoku solutions,
- target labels,
- solution-level split,
- removal rate,
- estimator count,
- random seed.

Only the number of feature columns differs. The evaluation uses raw model probabilities so that all configurations, including the model without candidate indicators, are compared consistently.

### Usage

Run the experiment from the project root:

```bash
python scripts/evaluate_feature_ablation.py
```

### Evaluation Configuration

```text
Generated Sudoku solutions: 100
Training samples: 3,200
Evaluation samples: 800
Training/test split: 80% / 20%
Removal rate: 0.50
Random Forest estimators: 100
Random seed: 42
```

### Ranking Results

| Feature configuration | Features | Top-1 | Top-2 | Top-3 | MRR |
|---|---:|---:|---:|---:|---:|
| Grid and position | 82 | 11.38% | 25.75% | 35.25% | 0.3206 |
| Candidate indicators | 91 | 50.38% | 83.12% | 96.25% | 0.7203 |
| Candidate interactions | 118 | 66.88% | 90.00% | 97.50% | 0.8152 |

Adding candidate indicators improves Top-1 accuracy by 39.00 percentage points and MRR by 0.3997. Adding candidate interactions provides another 16.50 percentage points of Top-1 accuracy and increases MRR by 0.0949.

The complete improvement from the 82-feature baseline to the 118-feature model is 55.50 percentage points in Top-1 accuracy.

### Probability-Quality Results

| Feature configuration | Mean confidence | ECE | Log loss |
|---|---:|---:|---:|
| Grid and position | 18.15% | 0.0677 | 2.2797 |
| Candidate indicators | 43.05% | 0.0908 | 1.0251 |
| Candidate interactions | 36.28% | 0.3093 | 1.1396 |

### Interpretation

The 82-feature model reaches 11.38% Top-1 accuracy, approximately the 11.11% random baseline for nine digits. Its 2.2797 log loss is also worse than the approximately 2.1972 loss of a uniform nine-class probability distribution. The Random Forest therefore learns little useful Sudoku-solving information from the flattened grid and target position alone.

Candidate indicators create the largest improvement. They explicitly tell the model which digits remain legal according to row, column, and block constraints. Candidate interactions then help distinguish between several valid digits by describing candidate distributions in neighbouring cells.

The full model provides the best ranking but not the best raw probability calibration. Its 66.88% Top-1 accuracy is paired with only 36.28% mean confidence, producing an ECE of 0.3093. The model is substantially underconfident even though its class ordering is useful.

The low ECE of the 82-feature model must not be interpreted as strong model quality. ECE measures agreement between confidence and observed accuracy, not predictive usefulness. A weak model can be well calibrated if it is appropriately uncertain.

### Testing

The tests cover:

- the three cumulative feature configurations,
- selection of 82, 91, and 118 feature columns,
- preservation of selected feature values,
- rejection of unsupported feature counts,
- rejection of arrays with insufficient columns,
- evaluation of every configuration,
- valid probability-ranking results for every trained model.

Result:

`176 passed`

### Limitations

- The experiment uses one training/test split and one random seed.
- Only a removal rate of 0.50 is evaluated.
- Feature groups are cumulative rather than evaluated in every possible combination.
- Random Forest probabilities are evaluated without post-training calibration.
- Cell-level ablation results do not directly measure complete-puzzle performance.

### Next Step

Repeat the feature-ablation experiment across several random seeds and removal rates. This will show whether the contribution of each feature group remains stable for harder and independently generated datasets.

### Conclusion

The Random Forest does not learn Sudoku effectively from raw grid values alone. Explicit candidate indicators provide the largest performance gain, while interaction features further improve ranking among valid digits. The experiment demonstrates that domain-specific feature engineering, rather than the classifier alone, is responsible for most of the current model quality.

---
## Commit 29 - Repeated Feature-Ablation Analysis

**Commit:** `feat: add repeated feature ablation`

### Objective

Verify whether the feature-ablation findings remain stable across independently generated datasets and higher removal rates.

### Motivation

Commit 28 used one solution-level split with a removal rate of 0.50 and random seed 42. Its clear result could still depend on that particular training and test sample. Repeating the entire experiment provides evidence about both stability and the influence of puzzle configuration.

### Experimental Design

For every removal rate and random seed, the evaluation:

1. Generates a new collection of solved and incomplete Sudokus.
2. Creates a new solution-level training/test split.
3. Trains separate 82-, 91-, and 118-feature Random Forests.
4. Calculates ranking and probability-quality metrics.
5. Aggregates every metric using mean, population standard deviation, minimum, and maximum.

```text
3 removal rates
       x
3 random seeds
       x
3 feature configurations
       =
27 trained Random Forest models
```

Unlike the earlier repeated solver experiment, the model is not held constant across seeds. The reported variation combines changes in generated training data, generated test data, and fitted models.

### Implemented

- Added result structures for repeated feature configurations and removal rates.
- Added repeated training across configurable random seeds.
- Added evaluation across configurable removal rates.
- Reused the cumulative feature configurations from Commit 28.
- Reused `MetricSummary` for consistent aggregation.
- Added mean and population-standard-deviation reporting.
- Added parameter validation and reproducibility tests.
- Added a command-line evaluation script.

### Usage

Run the experiment from the project root:

```bash
python scripts/evaluate_repeated_feature_ablation.py
```

### Evaluation Configuration

```text
Removal rates: 0.50, 0.60, 0.65
Random seeds: 101, 123, 202
Runs per removal rate: 3
Generated solutions per run: 100
Training/test split: 80% / 20%
Random Forest estimators: 100
Trained models: 27
```

### Ranking Results

| Removal rate | Features | Top-1 mean | Top-1 SD | Top-2 mean | Top-3 mean | MRR mean |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 82 | 11.17% | 1.07% | 22.88% | 33.75% | 0.3175 |
| 50% | 91 | 51.46% | 2.00% | 84.38% | 97.21% | 0.7287 |
| 50% | 118 | 64.50% | 1.27% | 89.08% | 98.00% | 0.8025 |
| 60% | 82 | 11.28% | 0.30% | 22.85% | 33.51% | 0.3157 |
| 60% | 91 | 37.95% | 1.40% | 69.27% | 88.92% | 0.6278 |
| 60% | 118 | 46.60% | 1.44% | 75.52% | 92.12% | 0.6848 |
| 65% | 82 | 12.63% | 0.40% | 23.78% | 34.81% | 0.3275 |
| 65% | 91 | 32.31% | 0.28% | 61.67% | 82.72% | 0.5798 |
| 65% | 118 | 39.81% | 0.31% | 67.69% | 86.35% | 0.6310 |

### Probability-Quality Results

| Removal rate | Features | Confidence mean | ECE mean | ECE SD | Log loss mean |
|---:|---:|---:|---:|---:|---:|
| 50% | 82 | 18.16% | 0.0700 | 0.0104 | 2.2683 |
| 50% | 91 | 42.47% | 0.1032 | 0.0123 | 1.0298 |
| 50% | 118 | 35.41% | 0.2923 | 0.0110 | 1.1586 |
| 60% | 82 | 18.23% | 0.0695 | 0.0021 | 2.2871 |
| 60% | 91 | 32.87% | 0.0572 | 0.0070 | 1.3445 |
| 60% | 118 | 28.53% | 0.1807 | 0.0190 | 1.4335 |
| 65% | 82 | 18.22% | 0.0559 | 0.0034 | 2.2672 |
| 65% | 91 | 29.29% | 0.0354 | 0.0054 | 1.4903 |
| 65% | 118 | 25.96% | 0.1408 | 0.0050 | 1.5649 |

### Interpretation

The 82-feature model remains close to the 11.11% random Top-1 baseline at every removal rate. Raw grid values and target position do not provide the Random Forest with enough structure to learn Sudoku effectively.

Candidate indicators provide the largest improvement in every experiment. Their Top-1 gain over the 82-feature model is 40.29 percentage points at 50% removal, 26.67 points at 60%, and 19.68 points at 65%.

Candidate interactions add another 13.04, 8.65, and 7.50 percentage points respectively. Their incremental value remains consistently positive but decreases as the removal rate increases and the puzzle contains less local information.

The full model's Top-1 accuracy decreases from 64.50% at 50% removal to 39.81% at 65%. Its Top-3 accuracy remains 86.35% at the hardest evaluated setting, so its ordered alternatives can still support backtracking.

The 91-feature model has better ECE and log loss than the 118-feature model at every removal rate. Interaction features therefore improve ranking but not raw probability quality. The decreasing ECE at higher removal rates does not imply better predictions; lower accuracy and lower confidence merely become more similar.

### Testing

The tests cover:

- aggregation across multiple random seeds,
- all three cumulative feature configurations,
- removal-rate grouping,
- valid summarized ranking metrics,
- empty removal-rate validation,
- empty random-seed validation,
- invalid removal rates,
- deterministic reproducibility.

Result:

`184 passed`

### Limitations

- Three seeds provide evidence of stability but remain a small sample.
- Removal rate is a difficulty proxy rather than a formal rating.
- Training and evaluation use the same removal rate within each run.
- Feature groups are cumulative and not tested in every combination.
- No probability-calibration method is applied.
- The experiment reports cell-level rather than complete-solver performance.

### Next Step

Compare probability-calibration methods for the full 118-feature model. The goal is to preserve its superior ranking while producing probabilities that better reflect observed accuracy.

### Conclusion

The feature-engineering findings are stable across independently generated datasets and removal rates. Candidate indicators provide most of the predictive improvement, interaction features consistently improve ranking, and raw grid features alone remain near random performance. The next model-focused challenge is improving probability calibration without losing ranking quality.
