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
Solution-Level Train/Test Split
    │
    ├───────────────┐
    ▼               ▼
Training Data     Test Data
    │               │
    ▼               ▼
Feature           Feature
Preprocessing     Preprocessing
    │               │
    ▼               ▼
X_train/y_train  X_test/y_test
    │               │
    └───────┬───────┘
            ▼
       ML Model
```

### Data Splitting Strategy

The dataset is split at the level of complete Sudoku solutions.

This prevents multiple puzzles derived from the same solved Sudoku from being distributed across both the training and test sets.

The split therefore takes place before cell-level feature extraction.

```text
Solved Sudoku A ──┐
Solved Sudoku B ──┤
Solved Sudoku C ──┼──→ Training solutions
Solved Sudoku D ──┤
Solved Sudoku E ──┘

Solved Sudoku F ──┐
Solved Sudoku G ──┤
Solved Sudoku H ──┼──→ Test solutions
```

This approach reduces the risk of data leakage and provides a more meaningful evaluation of model generalization.

### Components

**SudokuGrid**

Responsible for representing and validating Sudoku grids.

**Sudoku Solution Generator**

Generates complete and valid Sudoku solutions using randomized
backtracking.

**Dataset Generator**

Creates incomplete Sudoku puzzles by removing randomly selected cells
from solved Sudoku grids.

**Data Split**

Separates complete Sudoku solutions into training and test sets before
cell-level feature extraction.

**Feature Preprocessing**

Converts incomplete Sudoku puzzles into machine learning samples.

**Machine Learning Model**

To be implemented.