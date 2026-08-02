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

### Train and Save Models

```bash
python scripts/train_model.py
```

Trains the Random Forest, evaluates it on the hold-out test data, and stores the fitted estimator at `models/sudoku_random_forest.joblib`. Generated model artifacts are excluded from Git.

The stronger cell-level Histogram Gradient Boosting classifier can be trained separately:

```bash
python scripts/train_histogram_gradient_boosting.py
```

It uses the same generated training split and feature representation and stores the fitted estimator at `models/sudoku_histogram_gradient_boosting.joblib`.

A saved model can be loaded without retraining:

```python
from sudoku_ml.model.random_forest import SudokuRandomForest

model = SudokuRandomForest.load(
    "models/sudoku_random_forest.joblib"
)
```

Histogram Gradient Boosting models use the equivalent interface:

```python
from sudoku_ml.model.histogram_gradient_boosting import (
    SudokuHistogramGradientBoosting,
)

model = SudokuHistogramGradientBoosting.load(
    "models/sudoku_histogram_gradient_boosting.joblib"
)
```

Both wrappers support training, digit prediction, class probabilities, evaluation, persistence, and access to the learned digit classes. The command-line solver can load either model explicitly.

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

Random Forest is the default model type. Select the persistent Histogram Gradient Boosting model with:

```bash
sudoku-ml \
  --model-type histogram-gradient-boosting \
  "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

Without `--model`, the CLI chooses the default file for the selected type:

```text
random-forest               -> models/sudoku_random_forest.joblib
histogram-gradient-boosting -> models/sudoku_histogram_gradient_boosting.joblib
```

Use `--model` together with `--model-type` to load the selected model from another location. The CLI verifies that the serialized estimator matches the selected type.

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

The Greedy solver records every irreversible placement from its most recent attempt in `decision_trace`. Each entry contains the step, cell position, valid candidates, model ranking, selected digit, selected confidence, and whether the placement required an ML decision. Deterministic single-candidate placements have no model confidence.

```python
from sudoku_ml.solver import GreedyMLSudokuSolver

solver = GreedyMLSudokuSolver(model)
solution = solver.solve(puzzle)

for decision in solver.decision_trace:
    print(decision)
```

This is the project's current Model-only experiment: Sudoku constraints still reject illegal digits, but no search or backtracking repairs a model mistake.

### Analyze Model-only Decision Errors

```bash
python scripts/analyze_model_only_errors.py
```

Compares Random Forest and Histogram Gradient Boosting on identical unique-solution puzzles. Each Greedy decision trace is checked against ground truth to locate the first incorrect placement, count correct preceding decisions, measure confidence in the wrong digit, and determine the correct digit's rank.

Histogram Gradient Boosting improves exact match from 55% to 65% at 60% removal and from 25% to 30% at 65% removal. At every observed first error, the correct digit was ranked second on average. The Gradient Boosting errors were much more confident, however: 72.05% at 60% removal and 68.52% at 65%, compared with 31.22% and 29.60% for Random Forest.

### Analyze Model Probability Rankings

```bash
python scripts/evaluate_probability_ranking.py
```

Evaluates the Random Forest's complete probability ranking on the hold-out test set. The report compares raw model probabilities with candidate-constrained probabilities using Top-1, Top-2, and Top-3 accuracy, mean reciprocal rank, mean confidence, expected calibration error, and log loss.

In the current 800-sample evaluation, the correct digit appears in the first two positions in 90.00% of predictions and in the first three positions in 97.50%. Applying Sudoku candidate constraints leaves Top-k accuracy almost unchanged but reduces expected calibration error from 0.3093 to 0.0801 and log loss from 1.1396 to 0.6960.

### Compare Feature Groups

```bash
python scripts/evaluate_feature_ablation.py
```

Trains three separate Random Forest models on the same data while progressively adding grid and position features, candidate indicators, and candidate-interaction features. The experiment measures the contribution of each feature group to ranking and probability quality.

Without explicit Sudoku features, Top-1 accuracy is 11.38%, approximately the random baseline for nine digits. Candidate indicators raise it to 50.38%, and candidate interactions raise it further to 66.88%. This shows that domain-specific feature engineering provides most of the model's predictive ability.

### Repeat Feature Ablation

```bash
python scripts/evaluate_repeated_feature_ablation.py
```

Repeats the cumulative 82-, 91-, and 118-feature comparison across independently generated training and test splits at removal rates of 0.50, 0.60, and 0.65. The report includes mean ranking metrics, probability-quality metrics, and population standard deviations across seeds.

The repeated experiment confirms that raw grid features remain near random accuracy and that candidate features consistently provide the largest improvement. Interaction features improve ranking at every removal rate, although their incremental benefit decreases as more cells are removed.

### Evaluate Probability Calibration

```bash
python scripts/evaluate_probability_calibration.py
```

Compares the full 118-feature Random Forest with Sigmoid and Isotonic probability calibration. Model training, calibration, and final evaluation use independently generated Sudoku datasets. Each method is evaluated with raw probabilities and after Sudoku candidate constraints are applied.

In the current experiment, Sigmoid calibration reduces raw expected calibration error from 0.2963 to 0.0492 and raw log loss from 1.1521 to 0.7818. After candidate constraints, it reaches an expected calibration error of 0.0518 and a log loss of 0.7134 without materially changing the model's ranking performance.

### Repeat Probability Calibration

```bash
python scripts/evaluate_repeated_probability_calibration.py
```

Repeats Raw, Sigmoid, and Isotonic probability evaluation across three random seeds and removal rates from 0.50 to 0.65. Every run uses separate datasets for model training, calibration, and final evaluation.

Sigmoid consistently improves raw probability quality, but its benefit does not always survive candidate masking. After Sudoku constraints, the uncalibrated model has better ECE and log loss at removal rates of 0.60 and 0.65. Calibration therefore is not enabled globally in the solver.

### Compare Classifiers

```bash
python scripts/evaluate_model_comparison.py
```

Compares Logistic Regression, Random Forest, Extra Trees, and Histogram Gradient Boosting on the same solution-level split and complete 118-feature representation. Both raw and candidate-constrained probability rankings are evaluated.

In the current single-split experiment, Histogram Gradient Boosting achieves the best ranking with 75.75% Top-1 accuracy and the best raw log loss of 0.6129. This is 8.87 percentage points more Top-1 accuracy than the Random Forest, but the result still requires repeated evaluation across seeds and removal rates.

### Repeat Classifier Comparison

```bash
python scripts/evaluate_repeated_model_comparison.py
```

Repeats the four-classifier comparison across three seeds and removal rates of 0.50, 0.60, and 0.65. In addition to ranking and probability quality, it reports training time and batch probability-inference time.

Histogram Gradient Boosting consistently provides the best Top-1 accuracy, MRR, and log loss. Logistic Regression is the fastest inference model and has slightly better Top-3 accuracy at harder removal rates, while Random Forest and Extra Trees provide better candidate-constrained calibration.

### Compare Models in the Hybrid Solver

```bash
python scripts/evaluate_solver_models.py
```

Trains all four classifiers and evaluates them inside identical Hybrid solvers on the same unique-solution puzzles. The report measures exact ground-truth match, validity, end-to-end puzzle runtime, deterministic steps, ML decisions, and backtracking.

All models reach 100% exact match through backtracking. Logistic Regression is the fastest solver model, averaging 4.49 ms per puzzle at 65% removal compared with 62.65 ms for Random Forest. Histogram Gradient Boosting has the best cell-level ranking but does not consistently produce the fewest solver backtracks.

## Hybrid Solver

`HybridSudokuSolver` solves complete puzzles while preserving Sudoku validity. It selects constrained cells first, ranks ambiguous candidates with the trained Random Forest, and uses backtracking when a prediction leads to a dead end.

In the current 20-puzzle evaluation at a removal rate of 0.5, the solver reached a 100% valid solution rate with an average runtime of 20.64 ms per puzzle.
