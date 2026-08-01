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

**Commit:**
`feat: add sudoku grid representation`

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

**Commit:**
`feat: add sudoku grid validation`

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

