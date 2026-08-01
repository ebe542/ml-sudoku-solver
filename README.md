# ML Sudoku Solver

> A self-directed machine learning project for analyzing and solving Sudoku grids.

## Project Status

The project includes a hybrid solver that combines deterministic Sudoku constraints, machine-learning candidate ranking, and backtracking.

## Project Type

Self-directed learning project.

## Overview

This project explores how machine learning can be applied to Sudoku solving.

The application takes a predefined 9x9 Sudoku grid as input and uses machine learning techniques to analyze possible values for empty cells.

The project does not perform image recognition or OCR. The input is a structured Sudoku grid.

## Learning Objectives

- Build a reproducible machine learning project
- Generate and prepare training data
- Perform feature engineering
- Train and evaluate a machine learning model
- Use model predictions within a larger algorithmic system
- Compare machine learning predictions with deterministic Sudoku-solving logic
- Practice software engineering and testing for ML projects

## Technology

- Python
- NumPy
- pandas
- scikit-learn
- pytest
- Jupyter

## Disclaimer

This is a self-directed educational project created for learning and portfolio purposes.

The project is experimental and is not intended to represent a production-grade Sudoku solving system.

## Analysis Tools

### Baseline Evaluation

```bash
python scripts/evaluate_baseline.py
```

### Error Analysis

```bash
python scripts/analyze_errors.py
```

### Feature Importance

```bash
python -m sudoku_ml.analysis.feature_importance
```

### Grouped Cross-Validation

```bash
python scripts/evaluate_group_cross_validation.py
```

Evaluates the model using solution-level groups to prevent related cell samples from appearing in both training and validation folds.

### End-to-End Solver Evaluation

```bash
python scripts/evaluate_solver.py
```

Trains the Random Forest and evaluates the complete hybrid solver on an independently generated set of Sudoku puzzles. The evaluation reports solution validity, runtime, deterministic steps, ML decisions, and backtracks.

### Solver Comparison

```bash
python scripts/compare_solvers.py
```

Compares ML-guided and ascending numerical candidate ordering on the same puzzles. At a removal rate of 0.65, ML guidance reduced backtracking by 56.9%, but the classical solver remained approximately 33 times faster.

## Hybrid Solver

`HybridSudokuSolver` solves complete puzzles while preserving Sudoku validity.
It selects constrained cells first, ranks ambiguous candidates with the trained Random Forest, and uses backtracking when a prediction leads to a dead end.

In the current 20-puzzle evaluation at a removal rate of 0.5, the solver reached a 100% valid solution rate with an average runtime of 20.64 ms per puzzle.
