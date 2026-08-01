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
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      Evaluation  Analysis  Hybrid Solver
                              │
                              ▼
                     Constraint-validated
                       completed Sudoku
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
- accuracy evaluation,
- persistence with joblib.

### Model Persistence

`SudokuRandomForest.save()` serializes the fitted `RandomForestClassifier` to a joblib file. `SudokuRandomForest.load()` validates the deserialized object and returns a new wrapper around the restored classifier.

```text
Training data
     |
     v
Random Forest training
     |
     v
models/sudoku_random_forest.joblib
     |
     v
Loaded SudokuRandomForest
```

The persistence test verifies that predictions and learned digit classes remain identical after a save-load round trip. Model artifacts are generated locally and excluded from version control.

Joblib uses pickle-compatible deserialization. Model files must therefore come from trusted sources.

### Evaluation

Contains reusable evaluation logic, including:

- grouped cross-validation for cell-level prediction,
- end-to-end hybrid solver evaluation,
- solution and validity rates,
- runtime measurement,
- deterministic-step, ML-decision, and backtracking statistics.
- direct comparison of ML-guided and classical candidate ordering on identical puzzle sets.
- comparison across multiple removal rates with a separately trained model for each rate.

### Analysis

Contains tools for understanding model behavior, including:

- error analysis,
- accuracy by candidate count,
- confusion matrix evaluation,
- feature-importance analysis.

### Hybrid Solver

The hybrid solver combines three mechanisms:

1. Minimum-remaining-values cell selection prioritizes the most constrained empty cell.
2. Random Forest probabilities rank valid candidates when multiple choices remain.
3. Recursive backtracking reverses choices that lead to contradictions.

Machine-learning output affects search order only. Candidate filtering and the final solution remain governed by deterministic Sudoku constraints.

### Classical Solver

`ClassicalSudokuSolver` reuses the same validation, minimum-remaining-values cell selection, and backtracking implementation as the hybrid solver. It replaces ML-based candidate ranking with ascending numerical order.

Keeping the solving engine identical isolates candidate ordering as the only experimental difference between both strategies.

### Command-Line Interface

The command-line interface connects text input, model persistence, and the hybrid solver:

```text
81-cell Sudoku string
        |
        v
Input parsing and validation
        |
        v
Load saved Random Forest
        |
        v
HybridSudokuSolver
        |
        v
Formatted solution and statistics
```

The parser accepts digits, dots, and whitespace. Both `0` and `.` represent empty cells. The normalized input must contain exactly 81 cells before it is reshaped into a `SudokuGrid`.

The default model path is `models/sudoku_random_forest.joblib`. A different trusted model file can be selected with `--model`. Invalid input, missing model files, and unexpected serialized object types are reported as command-line usage errors.

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

The first end-to-end evaluation used 20 independently generated puzzles with a removal rate of 0.5:

| Metric | Result |
|---|---:|
| Puzzles solved | 20 / 20 |
| Valid solution rate | 100.00% |
| Average runtime | 20.64 ms |
| Deterministic steps | 774 |
| ML decisions | 26 |
| Backtracks | 0 |

The 800 placements consisted of 96.75% deterministic steps and 3.25 % ML-ranked decisions. At this removal rate, constraint propagation therefore performs most of the solving work.

A direct comparison on 20 puzzles with a removal rate of 0.65 produced:

| Metric | Hybrid | Classical |
|---|---:|---:|
| Solution rate | 100.00% | 100.00% |
| Valid solution rate | 100.00% | 100.00% |
| Average runtime | 144.88 ms | 4.39 ms |
| Deterministic steps | 1,306 | 1,867 |
| Backtracks | 494 | 1,145 |
| Average backtracks | 24.70 | 57.25 |

ML-guided ordering reduced backtracking by approximately 56.9%. However, feature calculation and Random Forest inference made the hybrid solver about 33 times slower. The model improves search order but does not improve total runtime in the current implementation.

The extended evaluation compared 20 puzzles per removal rate:

| Removal rate | Hybrid valid | Classical valid | Hybrid runtime | Classical runtime | Runtime ratio |
|---:|---:|---:|---:|---:|---:|
| 50% | 100.00% | 100.00% | 19.62 ms | 0.96 ms | 20.35x |
| 60% | 100.00% | 100.00% | 72.21 ms | 2.06 ms | 34.98x |
| 65% | 100.00% | 100.00% | 138.21 ms | 4.28 ms | 32.27x |
| 70% | 100.00% | 100.00% | 376.30 ms | 8.93 ms | 42.12x |

Search effort increased with the removal rate:

| Removal rate | Hybrid backtracks | Classical backtracks | Reduction | ML decisions |
|---:|---:|---:|---:|---:|
| 50% | 0.00 | 0.00 | 0.00% | 1.30 |
| 60% | 6.65 | 6.90 | 3.62% | 4.80 |
| 65% | 24.70 | 57.25 | 56.86% | 9.30 |
| 70% | 138.40 | 196.55 | 29.59% | 25.50 |

Backtracks and ML decisions are averages per puzzle. The results show that ML guidance can reduce search effort, but its relative benefit is not monotonic and does not offset inference cost.

## Current Limitation

The current evaluation covers 20 puzzles per removal rate and one pair of random seeds. Puzzle difficulty is approximated through the number of removed cells; the generator does not assign formal difficulty ratings or guarantee unique solutions. Repeated experiments with more seeds and larger evaluation sets are required for statistically stronger conclusions.
