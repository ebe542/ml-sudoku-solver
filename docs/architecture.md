# Architecture

## Current Pipeline

```text
SudokuGrid
    │
    ▼
Sudoku Solution Generator
    │
    ▼
Diverse Dataset Generator
    │
    ▼
Solution-Level Data Preparation
    │
    ├───────────────────────────────┐
    │                               │
    ▼                               ▼
Hold-out Train/Test Split      Grouped Cross-Validation
    │                               │
    ▼                               ▼
Feature Preprocessing          GroupKFold by Sudoku
    │                               │
    ▼                               ▼
X_train / y_train              Training / Validation Folds
X_test / y_test                     │
    │                               │
    └───────────────┬───────────────┘
                    ▼
          Random Forest Model
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      Evaluation           Analysis
          │                   │
          ▼                   ▼
       Accuracy       Error Analysis
                      Feature Importance
```

## Data Generation

Complete and valid Sudoku solutions are generated using randomized backtracking.

The dataset generator creates incomplete puzzles by copying each solution and replacing randomly selected cells with `0`.

```text
Complete Sudoku solution
          │
          ├── preserved as ground truth
          │
          └── randomly remove cells
                        │
                        ▼
               Incomplete puzzle
```

A value of `0` represents an empty Sudoku cell.

The generated data is reproducible through configurable random seeds.

## Feature Representation

Each empty cell becomes an individual machine learning sample.

The current feature vector contains 118 values:

| Feature Group | Feature Count |
|---|---:|
| Grid values | 81 |
| Target cell position | 1 |
| Candidate indicators | 9 |
| Row candidate interactions | 9 |
| Column candidate interactions | 9 |
| Block candidate interactions | 9 |
| **Total** | **118** |

The target is the correct digit from the corresponding solved Sudoku.

### Candidate Indicators

For each digit from 1 to 9, a binary feature indicates whether the digit is a valid candidate for the target cell.

```text
1 = valid candidate
0 = invalid candidate
```

Candidates are calculated from the incomplete puzzle using row, column, and 3×3 block constraints.

### Candidate Interactions

Candidate interaction features count how often each digit appears as a candidate in other empty cells within the same:

- row,
- column,
- 3×3 block.

These features provide context about relationships between the target cell and its peer cells.

## Data Splitting Strategy

The hold-out dataset is split at the level of complete Sudoku solutions before cell-level feature extraction.

```text
Solved Sudoku A ──┐
Solved Sudoku B ──┼──→ Training solutions
Solved Sudoku C ──┘

Solved Sudoku D ──┐
Solved Sudoku E ──┼──→ Test solutions
Solved Sudoku F ──┘
```

This prevents cell samples derived from the same Sudoku from appearing in both the training and test sets.

## Grouped Cross-Validation

For cross-validation, every cell-level sample receives the identifier of its source Sudoku.

`GroupKFold` ensures that all samples from one Sudoku remain within the same fold.

```text
Sudoku 1
├── Cell sample 1
├── Cell sample 2
└── Cell sample 40
        │
        ▼
  Shared group ID
```

This avoids data leakage and provides a more reliable estimate of model generalization.

## Components

### SudokuGrid

Responsible for representing Sudoku grids and validating:

- dimensions,
- allowed values,
- row constraints,
- column constraints,
- 3×3 block constraints.

### Sudoku Solution Generator

Generates complete and valid Sudoku solutions using randomized backtracking.

### Dataset Generator

Creates incomplete Sudoku puzzles from solved grids and preserves the complete solutions as ground truth.

### Data Preparation

Creates solution-level training and test datasets.

### Feature Preprocessing

Converts incomplete puzzles into cell-level machine learning samples with grid, candidate, and interaction features.

### Random Forest Model

Provides the current baseline classification model.

The model supports:

- training from an `MLDataSplit`,
- training directly from feature and target arrays,
- prediction,
- accuracy evaluation.

### Evaluation

Contains reusable evaluation logic, including grouped cross-validation.

### Analysis

Contains tools for understanding model behavior, including:

- error analysis,
- accuracy by candidate count,
- confusion matrix evaluation,
- feature-importance analysis.

## Current Results

| Feature Representation | Test Accuracy |
|---|---:|
| Grid and cell position — 82 features | 11.75% |
| Candidate indicators — 91 features | 50.25% |
| Candidate interactions — 118 features | 66.50% |

Five-fold grouped cross-validation produced:

| Metric | Accuracy |
|---|---:|
| Mean | 65.83% |
| Standard deviation | 1.63 percentage points |
| Minimum | 63.38% |
| Maximum | 67.37% |

## Current Limitation

The model predicts the digit for an individual empty cell.

It does not yet solve an entire Sudoku puzzle or guarantee that a sequence of predictions produces a valid complete grid.
