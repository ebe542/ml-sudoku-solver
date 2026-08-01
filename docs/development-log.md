# Development Log

This document records the development process of the project.
The project is developed as a self-study machine learning project.

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

This is important because one Sudoku solution can produce multiple incomplete puzzles. Splitting individual cell samples could allow
related puzzles to appear in both the training and test sets. 

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

The goal is to establish a baseline that can predict the correct digit for a selected empty Sudoku cell based on the current puzzle
state.

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

The result shows that the current feature representation does not provide enough information for the Random Forest to reliably infer the
correct Sudoku digit.

The model currently receives the Sudoku grid and the target cell position, but the Sudoku constraints are not explicitly represented
as features.

This provides a useful baseline for future feature engineering and model improvements.

### Important Limitation

This accuracy measures cell-level digit prediction.

It does not represent the accuracy of a complete Sudoku solver.

---
## Commit 10 — Sudoku Constraint Features

**Commit:** `feat: add sudoku constraint features`

### Objective

Improve the feature representation by explicitly providing Sudoku-specific constraint information to the machine learning model.

The baseline model only received the Sudoku grid and the target cell position. The new features additionally describe which digits are
valid candidates for the target cell.

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

The baseline model had to learn Sudoku constraints implicitly from the grid representation. By explicitly representing possible candidate
digits, important Sudoku structure becomes directly available to the model.

The experiment therefore provides evidence that the feature representation has a major impact on model performance.

### Current Limitation

The 50.25% accuracy measures cell-level digit prediction.

It does not represent the ability to solve a complete Sudoku puzzle.
The model still predicts individual target cells rather than producing a complete solved grid.

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

The model performs perfectly when only one candidate is possible. However, performance decreases significantly as the number of
possible candidates increases.

This indicates that the current model can exploit local Sudoku constraints but has difficulty resolving situations where multiple
candidate digits remain possible.

This is an important limitation of the current cell-level prediction approach because Sudoku decisions can depend on relationships between
multiple cells.

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

This demonstrates that relationships between neighbouring candidate sets provide valuable information for distinguishing between multiple
valid Sudoku candidates.

### Conclusion

The experiment confirms that representing local interactions between candidate sets significantly improves model performance without
changing the underlying learning algorithm.

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

The relatively small standard deviation shows that the model performs consistently across different groups of previously unseen Sudoku
solutions.

### Conclusion

Grouped cross-validation confirms that the current Random Forest and feature representation provide stable cell-level prediction performance
while avoiding leakage between samples derived from the same Sudoku.

---
## Commit 15 — Hybrid ML Sudoku Solver

**Commit:** `feat: add hybrid ML-guided Sudoku solver`

### Objective

Extend the cell-level classifier into a system capable of solving complete
Sudoku puzzles without allowing invalid model predictions.

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

The model only determines the order in which valid candidates are attempted.
Sudoku constraints filter all candidates, while backtracking ensures that a
locally plausible prediction cannot make the final solution invalid.

### Testing

The test suite verifies complete and valid solutions, preservation of given
digits, input immutability, and rejection of conflicting puzzles.

Result:

`71 passed`
