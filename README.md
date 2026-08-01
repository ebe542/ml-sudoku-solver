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

### Removal-Rate Evaluation

```bash
python scripts/evaluate_difficulty_levels.py
```

Compares both solver strategies at removal rates from 0.50 to 0.70. The report includes valid solution rates, runtime ratios, average backtracks, backtrack reduction, and average ML decisions per puzzle. Removal rate is treated as a proxy for difficulty rather than a formal Sudoku difficulty rating.

### Train and Save a Model

```bash
python scripts/train_model.py
```

Trains the Random Forest, evaluates it on the hold-out test data, and stores the fitted estimator at `models/sudoku_random_forest.joblib`. Generated model artifacts are excluded from Git.

A saved model can be loaded without retraining:

```python
from sudoku_ml.model.random_forest import SudokuRandomForest

model = SudokuRandomForest.load(
    "models/sudoku_random_forest.joblib"
)
```

Only load joblib files from trusted sources. Deserializing an untrusted file can execute arbitrary code.

### Solve a Sudoku from the Command Line

Install the project in editable mode to register the `sudoku-ml` command:

```bash
python -m pip install -e ".[dev]"
```

Train the default model, then pass an 81-cell Sudoku string to the ML-guided solver:

```bash
sudoku-ml "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

Use `0` or `.` for empty cells. Whitespace and line breaks in the input are ignored. The CLI prints the original puzzle, the completed solution, and solver statistics.

To load a model from a different location:

```bash
sudoku-ml --model models/sudoku_random_forest.joblib "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

Use the classical solver without a model:

```bash
sudoku-ml --classical "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

Read a formatted puzzle from a UTF-8 text file:

```bash
sudoku-ml --classical --input-file puzzle.txt
```

Show help or the installed version:

```bash
sudoku-ml --help
sudoku-ml --version
```

The original module invocation remains available:

```bash
python -m sudoku_ml.cli --classical "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

### Generate Puzzles with a Unique Solution

```python
from sudoku_ml.dataset.unique_generator import create_unique_dataset

dataset = create_unique_dataset(
    num_samples=10,
    removal_rate=0.5,
    random_seed=42,
)
```

The uniqueness-preserving generator removes a clue only when the puzzle still has exactly one solution. Solution counting stops after a second solution is found because that is sufficient to classify a puzzle as non-unique.

### Evaluate on Unique-Solution Puzzles

```bash
python scripts/evaluate_unique_solvers.py
```

Compares the hybrid and classical solvers on identical uniquely solvable puzzles at removal rates of 0.50, 0.60, and 0.65. In addition to validity, the evaluation checks whether each solver output exactly matches the unique stored solution.

### Repeat Evaluation Across Seeds

```bash
python scripts/evaluate_repeated_solvers.py
```

Repeats the unique-solution comparison across several independently generated puzzle sets and reports mean, population standard deviation, minimum, and maximum through reusable metric summaries. This distinguishes stable behavior from results that depend strongly on one evaluation seed.

### Analyze Heuristic Puzzle Difficulty

```bash
python scripts/analyze_puzzle_difficulty.py
```

Analyzes uniquely solvable puzzles using clue count, initial candidate structure, deterministic steps, branching decisions, and classical backtracking effort. The resulting easy, medium, hard, and expert levels are project-specific heuristics and are not official Sudoku difficulty ratings.

### Evaluate Greedy ML Without Backtracking

```bash
python scripts/evaluate_greedy_solver.py
```

Compares a constraint-aware Greedy ML solver with the Hybrid ML solver on identical unique-solution puzzles. Greedy ML permanently accepts the model's highest-ranked valid candidate and cannot recover from a wrong decision. At removal rates of 0.60 and 0.65, its exact solution rates were 55% and 25%, compared with 100% for the backtracking-enabled hybrid solver.

### Analyze Model Probability Rankings

```bash
python scripts/evaluate_probability_ranking.py
```

Evaluates the Random Forest's complete probability ranking on the hold-out test set. The report compares raw model probabilities with candidate-constrained probabilities using Top-1, Top-2, and Top-3 accuracy, mean reciprocal rank, mean confidence, expected calibration error, and log loss.

In the current 800-sample evaluation, the correct digit appears in the first two positions in 90.00% of predictions and in the first three positions in 97.50%. Applying Sudoku candidate constraints leaves Top-k accuracy almost unchanged but reduces expected calibration error from 0.3093 to 0.0801 and log loss from 1.1396 to 0.6960.

## Hybrid Solver

`HybridSudokuSolver` solves complete puzzles while preserving Sudoku validity. It selects constrained cells first, ranks ambiguous candidates with the trained Random Forest, and uses backtracking when a prediction leads to a dead end.

In the current 20-puzzle evaluation at a removal rate of 0.5, the solver reached a 100% valid solution rate with an average runtime of 20.64 ms per puzzle.
