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

### Unique-Solution Generation

The original dataset generator removes cells without checking whether the resulting puzzle still has one solution. The optional unique-solution generator performs an additional bounded search after every tentative removal:

```text
Complete Sudoku solution
          |
          v
Tentatively remove one clue
          |
          v
Count solutions up to two
          |
     +----+----+
     |         |
     v         v
One solution  Two solutions
Keep removal  Restore clue
```

`count_solutions()` uses minimum-remaining-values cell selection and recursive backtracking. It stops when the configured solution limit is reached. `has_unique_solution()` requests at most two solutions and returns true only when exactly one is found.

Starting from a complete valid grid, `create_unique_puzzle()` considers cells in a reproducibly shuffled order. A removal is permanent only if uniqueness remains. If the requested removal count cannot be reached, generation fails explicitly instead of returning a puzzle that violates the request.

`create_unique_dataset()` repeats this process across independently generated complete Sudoku solutions and preserves each complete grid as ground truth.

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
- exact ground-truth matching for uniquely solvable evaluation puzzles.

### Analysis

Contains tools for understanding model behavior, including:

- error analysis,
- accuracy by candidate count,
- confusion matrix evaluation,
- feature-importance analysis,
- Top-k probability-ranking analysis,
- mean reciprocal rank,
- confidence calibration and log loss,
- comparison of raw and candidate-constrained probabilities,
- feature-ablation analysis across the 82-, 91-, and 118-feature representations.

### Hybrid Solver

The hybrid solver combines three mechanisms:

1. Minimum-remaining-values cell selection prioritizes the most constrained empty cell.
2. Random Forest probabilities rank valid candidates when multiple choices remain.
3. Recursive backtracking reverses choices that lead to contradictions.

Machine-learning output affects search order only. Candidate filtering and the final solution remain governed by deterministic Sudoku constraints.

### Greedy ML Solver

`GreedyMLSudokuSolver` reuses the hybrid solver's input validation, minimum-remaining-values cell selection, candidate filtering, feature generation, and Random Forest ranking. It changes only the search strategy:

```text
Select most constrained cell
          |
          v
Rank valid candidates with ML
          |
          v
Permanently place top candidate
          |
     +----+----+
     |         |
     v         v
Continue   Contradiction
              |
              v
            Fail
```

The Greedy solver never tries a second candidate and never backtracks. It is therefore a constraint-aware measurement of whether the model can produce a complete correct decision sequence without algorithmic correction. It is not a completely unconstrained end-to-end neural solver because Sudoku rules still restrict the candidate set.

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

The parser accepts digits, dots, and whitespace. Both `0` and `.` represent empty cells. The normalized input must contain exactly 81 cells before it is reshaped into a `SudokuGrid`. Input can be supplied directly as a positional argument or read from a UTF-8 text file with `--input-file`; these sources are mutually exclusive.

The default hybrid mode loads `models/sudoku_random_forest.joblib`. A different trusted model file can be selected with `--model`. The `--classical` option uses deterministic numerical candidate ordering and does not load a model. Invalid input, missing files, conflicting input sources, and unexpected serialized object types are reported as command-line usage errors.

### Package Entry Point

The project metadata registers an installable console script:

```toml
[project.scripts]
sudoku-ml = "sudoku_ml.cli:main"
```

An editable development install creates the platform-specific executable while continuing to import code from the local `src/` directory:

```bash
python -m pip install -e ".[dev]"
sudoku-ml --version
```

The CLI reads package version metadata through `importlib.metadata` and falls back to `development` when distribution metadata is unavailable. Both `sudoku-ml` and `python -m sudoku_ml.cli` call the same tested `main()` function.

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

### Unique-Solution Evaluation

The general solver evaluation optionally accepts expected solution grids. When ground truth is supplied, `matching_solutions` counts exact grid matches and `matching_solution_rate` reports their proportion. Existing evaluations that do not provide expected solutions remain unchanged.

The unique-solution experiment trains the Random Forest on the existing random-removal training pipeline and evaluates both solvers on identical puzzles from `create_unique_dataset()`:

| Removal rate | Hybrid match | Classical match | Hybrid runtime | Classical runtime | Runtime ratio |
|---:|---:|---:|---:|---:|---:|
| 50% | 100.00% | 100.00% | 0.66 ms | 0.68 ms | 0.97x |
| 60% | 100.00% | 100.00% | 12.32 ms | 1.64 ms | 7.51x |
| 65% | 100.00% | 100.00% | 52.14 ms | 3.54 ms | 14.74x |

| Removal rate | Hybrid backtracks | Classical backtracks | Reduction | Hybrid ML decisions |
|---:|---:|---:|---:|---:|
| 50% | 0.00 | 0.00 | 0.00% | 0.00 |
| 60% | 7.10 | 6.60 | -7.58% | 0.70 |
| 65% | 27.00 | 52.70 | 48.77% | 3.30 |

All search-effort values are averages per puzzle. Exact match rates of 100% confirm that both solvers reproduce the unique ground truth. The hybrid ordering is not uniformly better: it increases backtracking slightly at 60% but reduces it substantially at 65%.

### Repeated Evaluation

The repeated evaluation trains one deterministic model per removal rate and evaluates it on multiple independently seeded unique-puzzle datasets. Keeping the trained model fixed isolates variation caused by the evaluation sample rather than mixing model-training and puzzle-sampling variation.

`MetricSummary` stores the mean, population standard deviation, minimum, and maximum of each metric. `RepeatedRemovalRateResult` groups summaries for exact match rate, runtime, runtime ratio, backtracking, backtrack reduction, and ML decisions.

Three runs with evaluation seeds 101, 123, and 202 produced the following mean and population standard deviation:

| Removal rate | Hybrid match | Classical match | Hybrid runtime | Classical runtime | Runtime ratio |
|---:|---:|---:|---:|---:|---:|
| 50% | 100.00% +/- 0.00 | 100.00% +/- 0.00 | 0.65 +/- 0.04 ms | 0.63 +/- 0.02 ms | 1.03 +/- 0.02 |
| 60% | 100.00% +/- 0.00 | 100.00% +/- 0.00 | 20.30 +/- 8.13 ms | 1.76 +/- 0.26 ms | 11.22 +/- 3.14 |
| 65% | 100.00% +/- 0.00 | 100.00% +/- 0.00 | 57.75 +/- 3.70 ms | 3.94 +/- 0.43 ms | 14.77 +/- 1.40 |

| Removal rate | Hybrid backtracks | Classical backtracks | Reduction | ML decisions |
|---:|---:|---:|---:|---:|
| 50% | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00% +/- 0.00 | 0.00 +/- 0.00 |
| 60% | 10.73 +/- 7.29 | 11.87 +/- 6.42 | 13.52% +/- 24.68 | 1.27 +/- 0.54 |
| 65% | 29.20 +/- 6.56 | 58.33 +/- 13.15 | 45.39% +/- 22.08 | 3.60 +/- 0.24 |

The large relative spread at 60% shows that a single evaluation seed can give a misleading impression of candidate-ordering quality. The 65% result remains positive across the aggregate but still exhibits substantial puzzle-sample variation.

### Heuristic Difficulty Analysis

Removal rate controls how many clues are hidden but does not directly measure logical or search difficulty. The heuristic difficulty analyzer therefore combines structural input features with model-free classical solver effort:

```text
Unique Sudoku puzzle
        |
        +--> given and empty cells
        +--> initial single candidates
        +--> average initial candidate count
        |
        v
Classical MRV backtracking solver
        |
        +--> deterministic steps
        +--> branching decisions
        +--> backtracks
        |
        v
Heuristic score and level
```

The difficulty score is defined as:

```text
difficulty score = branching decisions + backtracks
```

The project-specific levels are:

| Level | Search-effort rule |
|---|---:|
| Easy | score = 0 |
| Medium | score 1-10 |
| Hard | score 11-100 |
| Expert | score above 100 |

These thresholds are transparent and reproducible but empirical. They do not correspond to standardized human-solving difficulty because the analyzer does not identify named logical techniques such as hidden singles, pairs, or X-Wing.

An analysis of ten unique puzzles per removal rate produced:

| Removal rate | Clues | Initial singles | Avg candidates | Branches | Backtracks | Score |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 41.00 | 8.20 | 2.40 | 0.00 | 0.00 | 0.00 |
| 60% | 33.00 | 3.90 | 2.99 | 0.70 | 6.60 | 7.30 |
| 65% | 29.00 | 2.50 | 3.37 | 5.80 | 52.70 | 58.50 |

| Removal rate | Easy | Medium | Hard | Expert |
|---:|---:|---:|---:|---:|
| 50% | 10 | 0 | 0 | 0 |
| 60% | 6 | 1 | 3 | 0 |
| 65% | 2 | 3 | 2 | 3 |

The distribution demonstrates that removal rate and search difficulty are related but not equivalent. Puzzles with the same number of clues can fall into different heuristic levels.

### Greedy ML Evaluation

The Greedy comparison uses the same trained model, unique puzzles, expected solutions, cell-selection strategy, and candidate constraints for both Greedy and Hybrid ML. The only difference is whether a failed model choice can be reversed.

| Removal rate | Greedy exact match | Hybrid exact match | Greedy failure | Recovered by Hybrid |
|---:|---:|---:|---:|---:|
| 50% | 100.00% | 100.00% | 0.00% | 0 / 20 |
| 60% | 55.00% | 100.00% | 45.00% | 9 / 20 |
| 65% | 25.00% | 100.00% | 75.00% | 15 / 20 |

| Removal rate | Greedy runtime | Hybrid runtime | Greedy ML decisions | Hybrid ML decisions | Hybrid backtracks |
|---:|---:|---:|---:|---:|---:|
| 50% | 0.72 ms | 0.71 ms | 0.00 | 0.00 | 0.00 |
| 60% | 14.39 ms | 17.56 ms | 0.90 | 1.10 | 11.80 |
| 65% | 34.07 ms | 60.24 ms | 2.15 | 3.95 | 38.85 |

The 50% result does not measure model ability because no ambiguous decisions were required. At higher removal rates, one unrecoverable model mistake can invalidate the entire Greedy solve. Hybrid backtracking converts those failed decision chains into a 100% exact match rate.

### Probability-Ranking Analysis

The probability-ranking analysis evaluates more than the model's single most likely digit. For every hold-out sample, it determines the rank of the correct target digit and calculates Top-1, Top-2, and Top-3 accuracy together with mean reciprocal rank.

```text
Random Forest probabilities
            |
      +-----+-----+
      |           |
      v           v
Raw ranking   Candidate mask
                  |
                  v
             Renormalization
                  |
                  v
          Constrained ranking
```

Candidate-constrained analysis masks digits that are not valid Sudoku candidates for the target cell and renormalizes the remaining probabilities to sum to one. The mask is derived from the nine candidate-indicator features at indices 82 through 90. It uses only the incomplete puzzle and does not expose the target solution.

`ProbabilityRankingResult` stores the sample count, three Top-k accuracies, mean reciprocal rank, mean confidence, expected calibration error, and multiclass log loss. Expected calibration error groups predictions into confidence bins and measures the weighted difference between confidence and empirical accuracy.

The experiment trained a 100-estimator Random Forest on 100 generated solutions with a solution-level 80/20 split, a removal rate of 0.50, and random seed 42. The hold-out set contained 800 cell-level samples:

| Metric | Raw | Candidate-constrained |
|---|---:|---:|
| Top-1 accuracy | 66.88% | 66.88% |
| Top-2 accuracy | 90.00% | 90.00% |
| Top-3 accuracy | 97.50% | 97.50% |
| Mean reciprocal rank | 0.8152 | 0.8153 |
| Mean confidence | 36.28% | 59.20% |
| Expected calibration error | 0.3093 | 0.0801 |
| Log loss | 1.1396 | 0.6960 |

Candidate constraints do not change which digit is ranked first in this evaluation and have almost no effect on the ranking metrics. They substantially improve the probability distribution, however, because probability mass assigned to illegal digits is removed and redistributed across valid candidates.

The 90.00% Top-2 and 97.50% Top-3 accuracies explain why ML-guided backtracking is useful even though Top-1 accuracy is only 66.88%. When the first choice fails, the correct digit is usually near the top of the remaining search order.

### Feature-Ablation Analysis

The feature-ablation analysis trains a separate Random Forest for each cumulative feature representation. Every model uses the same training and test samples, estimator count, and random seed. Only the number of input columns changes:

```text
82 features
Grid + target position
          |
          v
91 features
+ candidate indicators
          |
          v
118 features
+ candidate interactions
```

`select_features()` selects a validated prefix of the complete 118-feature array. `evaluate_feature_ablation()` then trains and evaluates each configuration independently and stores its probability-ranking metrics in a `FeatureAblationResult`.

The controlled experiment produced:

| Feature configuration | Features | Top-1 | Top-2 | Top-3 | MRR |
|---|---:|---:|---:|---:|---:|
| Grid and position | 82 | 11.38% | 25.75% | 35.25% | 0.3206 |
| Candidate indicators | 91 | 50.38% | 83.12% | 96.25% | 0.7203 |
| Candidate interactions | 118 | 66.88% | 90.00% | 97.50% | 0.8152 |

| Feature configuration | Mean confidence | ECE | Log loss |
|---|---:|---:|---:|
| Grid and position | 18.15% | 0.0677 | 2.2797 |
| Candidate indicators | 43.05% | 0.0908 | 1.0251 |
| Candidate interactions | 36.28% | 0.3093 | 1.1396 |

Grid values and target position alone yield 11.38% Top-1 accuracy, close to the 11.11% random baseline for nine balanced classes. Adding candidate indicators produces the largest improvement, while candidate interactions further improve ranking among the valid digits.

The complete feature set has the best ranking but worse raw ECE and log loss than the 91-feature model. Ranking quality and probability calibration are different properties: the interaction features improve class ordering but produce underconfident raw probabilities. The low ECE of the weak 82-feature model does not indicate useful predictions; it mainly reflects that low confidence is consistent with low accuracy.

### Repeated Feature Ablation

The repeated ablation evaluates the same cumulative feature groups across several removal rates and random seeds. Unlike the repeated solver evaluation, every seed generates a new solution-level training/test split and trains three new Random Forest models. Its variation therefore includes training-data sampling, test-data sampling, and fitted-model variation.

```text
Removal rates 0.50 / 0.60 / 0.65
                 |
        Seeds 101 / 123 / 202
                 |
        New train/test split
                 |
       +---------+---------+
       |         |         |
       v         v         v
  82 features 91 features 118 features
       |         |         |
       +---------+---------+
                 |
                 v
       MetricSummary per model
```

`RepeatedFeatureConfigurationResult` contains summary statistics for ranking, confidence, calibration, and log loss. `RepeatedFeatureRemovalRateResult` groups all configurations for one removal rate, and `RepeatedFeatureAblationResult` records the seeds and all removal-rate results.

Mean ranking results across three seeds were:

| Removal rate | Features | Top-1 | Top-1 SD | Top-2 | Top-3 | MRR |
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

Probability-quality means were:

| Removal rate | Features | Confidence | ECE | ECE SD | Log loss |
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

The low Top-1 standard deviations show that the relative ordering of feature configurations is stable. Increasing removal rate reduces the information supplied by local constraints, so accuracy decreases for both feature-engineered models. Interaction features consistently improve ranking but consistently worsen raw ECE and log loss relative to the 91-feature model.

### Probability Calibration

Probability calibration adjusts the numeric confidence of an already trained classifier without changing its feature representation. The implementation compares Sigmoid and Isotonic calibration through scikit-learn's `CalibratedClassifierCV`.

```text
Training set
     |
     v
118-feature Random Forest
     |
     v
FrozenEstimator
     |
     +-----------------------+
     |                       |
     v                       v
Sigmoid calibrator     Isotonic calibrator
     |                       |
     +-----------+-----------+
                 |
                 v
       Independent evaluation set
                 |
        +--------+--------+
        |                 |
        v                 v
 Raw probabilities   Candidate-constrained
```

`FrozenEstimator` prevents calibration from retraining the Random Forest. A single explicit calibration split exposes every calibration sample to the calibrator while the frozen estimator's fitting operation remains inactive. `CalibratedProbabilityModel` adapts the scikit-learn classifier to the project's `predict_probabilities` and `classes` interface.

`ProbabilityCalibrationResult` stores raw and candidate-constrained `ProbabilityRankingResult` values for each method. The evaluation always includes the uncalibrated model as a baseline.

The experiment uses three independently generated datasets with removal rate 0.50:

| Purpose | Seed | Samples used |
|---|---:|---:|
| Random Forest training | 42 | 3,200 |
| Probability calibration | 123 | 800 |
| Final evaluation | 202 | 800 |

Keeping final evaluation data separate from calibration data prevents the flexible Isotonic mapping from being evaluated on the samples used to fit it.

Raw-probability results were:

| Method | Top-1 | Top-2 | Top-3 | MRR | Confidence | ECE | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 65.50% | 89.38% | 97.50% | 0.8076 | 35.87% | 0.2963 | 1.1521 |
| Sigmoid | 66.25% | 88.38% | 97.50% | 0.8095 | 63.68% | 0.0492 | 0.7818 |
| Isotonic | 66.25% | 88.62% | 97.88% | 0.8101 | 64.12% | 0.0578 | 0.9596 |

Candidate-constrained results were:

| Method | Top-1 | Top-2 | Top-3 | MRR | Confidence | ECE | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 65.50% | 89.38% | 97.50% | 0.8076 | 57.69% | 0.0781 | 0.7181 |
| Sigmoid | 66.25% | 88.38% | 97.62% | 0.8098 | 68.34% | 0.0518 | 0.7134 |
| Isotonic | 66.25% | 88.62% | 97.88% | 0.8101 | 64.53% | 0.0564 | 0.9546 |

Sigmoid calibration provides the best probability quality in this experiment. It substantially corrects the raw model's underconfidence. Candidate constraints already provide a strong improvement by removing probability mass from illegal digits, so the additional constrained log-loss improvement from Sigmoid is small.

Calibration changes ranking only slightly because multiclass calibration adjusts each class separately before normalization. Isotonic achieves similar ranking but worse log loss, which is consistent with its greater flexibility and possible overfitting on the calibration sample.

### Repeated Probability Calibration

The repeated calibration analysis tests whether the single-split calibration result generalizes across seeds and removal rates. Each run derives three independent dataset seeds from one run seed:

```text
run seed          -> Random Forest training
run seed + 10,000 -> probability calibration
run seed + 20,000 -> final evaluation
```

For each removal rate, `RepeatedCalibrationMethodResult` summarizes raw and candidate-constrained Top-k accuracy, MRR, confidence, ECE, and log loss. `RepeatedCalibrationRemovalRateResult` groups Raw, Sigmoid, and Isotonic results, while `RepeatedProbabilityCalibrationResult` stores all removal rates and run seeds.

Mean raw-probability quality across seeds was:

| Removal rate | Method | Confidence | ECE | ECE SD | Log loss |
|---:|---|---:|---:|---:|---:|
| 50% | Raw | 35.58% | 0.3122 | 0.0036 | 1.1507 |
| 50% | Sigmoid | 63.73% | 0.0538 | 0.0142 | 0.7584 |
| 50% | Isotonic | 64.18% | 0.0403 | 0.0112 | 1.0814 |
| 60% | Raw | 28.54% | 0.1942 | 0.0058 | 1.4296 |
| 60% | Sigmoid | 49.18% | 0.0402 | 0.0052 | 1.1697 |
| 60% | Isotonic | 48.41% | 0.0309 | 0.0099 | 1.3689 |
| 65% | Raw | 26.05% | 0.1360 | 0.0047 | 1.5637 |
| 65% | Sigmoid | 43.23% | 0.0462 | 0.0133 | 1.3903 |
| 65% | Isotonic | 41.59% | 0.0316 | 0.0070 | 1.5319 |

Mean candidate-constrained probability quality was:

| Removal rate | Method | Confidence | ECE | ECE SD | Log loss |
|---:|---|---:|---:|---:|---:|
| 50% | Raw | 57.47% | 0.0928 | 0.0030 | 0.7151 |
| 50% | Sigmoid | 68.38% | 0.0350 | 0.0108 | 0.6907 |
| 50% | Isotonic | 64.62% | 0.0364 | 0.0088 | 1.0761 |
| 60% | Raw | 44.74% | 0.0414 | 0.0032 | 1.0271 |
| 60% | Sigmoid | 55.50% | 0.0699 | 0.0080 | 1.0557 |
| 60% | Isotonic | 50.04% | 0.0364 | 0.0117 | 1.3443 |
| 65% | Raw | 38.84% | 0.0156 | 0.0063 | 1.2022 |
| 65% | Sigmoid | 49.65% | 0.1002 | 0.0102 | 1.2586 |
| 65% | Isotonic | 43.85% | 0.0474 | 0.0102 | 1.4893 |

Sigmoid reduces raw log loss at every removal rate, but candidate masking changes the calibrated distribution through zeroing and renormalization. At 60% and 65% removal, this makes Sigmoid overconfident and gives the uncalibrated constrained model better ECE and log loss. Calibration has only minor effects on Top-k accuracy and MRR, so it provides little direct benefit to candidate ordering.

Isotonic sometimes produces the lowest raw ECE but consistently has worse log loss than Sigmoid. This indicates rare high-penalty errors that ECE alone does not expose.

### Zero-Mass Candidate Fallback

Isotonic calibration exposed an edge case in candidate masking: every valid candidate can receive zero probability even when the Sudoku candidate mask itself is non-empty. Direct renormalization would then divide by zero and produce `NaN` values.

`apply_candidate_constraints()` detects rows with zero valid probability mass and replaces them with a uniform distribution over valid candidates before normalization:

```text
Valid candidates exist
        |
        v
Valid probability sum = 0?
        |
   +----+----+
   |         |
  no        yes
   |         |
   |    Uniform valid-candidate weights
   |         |
   +----+----+
        |
        v
Normalize to sum 1
```

The fallback expresses absence of a usable model preference without inventing an arbitrary ranking or permitting invalid probability values.

### Classifier Comparison

The classifier comparison keeps data and feature engineering fixed while replacing only the classification algorithm. Every model receives the same solution-level training/test split and all 118 features.

The compared models represent different learning strategies:

- Logistic Regression provides a standardized linear baseline.
- Random Forest is the existing independently averaged tree ensemble.
- Extra Trees increases randomization in tree construction.
- Histogram Gradient Boosting builds trees sequentially to correct earlier errors.

`ProbabilityModelAdapter` gives all scikit-learn estimators the shared `fit`, `predict_probabilities`, and `classes` interface used by the ranking analysis. Logistic Regression is wrapped in a pipeline with `StandardScaler`; the tree models consume the original feature values.

```text
Shared 118-feature training data
                |
    +-----------+-----------+-----------+
    |           |           |           |
    v           v           v           v
 Logistic    Random       Extra      Histogram
Regression   Forest       Trees      Gradient Boosting
    |           |           |           |
    +-----------+-----------+-----------+
                |
                v
Shared raw and candidate-constrained evaluation
```

`ModelComparisonResult` stores raw and candidate-constrained `ProbabilityRankingResult` values for one classifier. `evaluate_models()` trains every supplied model and applies the same evaluation pipeline.

The first comparison used 3,200 training samples, 800 evaluation samples, 118 features, removal rate 0.50, and random seed 42.

Raw ranking results were:

| Model | Top-1 | Top-2 | Top-3 | MRR |
|---|---:|---:|---:|---:|
| Logistic Regression | 68.00% | 90.00% | 97.50% | 0.8206 |
| Random Forest | 66.88% | 90.00% | 97.50% | 0.8152 |
| Extra Trees | 69.12% | 89.00% | 97.00% | 0.8246 |
| Histogram Gradient Boosting | 75.75% | 91.50% | 98.38% | 0.8630 |

Raw probability-quality results were:

| Model | Confidence | ECE | Log loss |
|---|---:|---:|---:|
| Logistic Regression | 83.97% | 0.1608 | 0.9574 |
| Random Forest | 36.28% | 0.3093 | 1.1396 |
| Extra Trees | 38.89% | 0.3048 | 1.0668 |
| Histogram Gradient Boosting | 86.56% | 0.1093 | 0.6129 |

Candidate-constrained results were:

| Model | Top-1 | Top-2 | Top-3 | MRR | Confidence | ECE | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 68.38% | 90.62% | 98.62% | 0.8250 | 85.90% | 0.1766 | 0.9279 |
| Random Forest | 66.88% | 90.00% | 97.50% | 0.8153 | 59.20% | 0.0801 | 0.6960 |
| Extra Trees | 69.12% | 89.00% | 97.00% | 0.8247 | 58.67% | 0.1045 | 0.6980 |
| Histogram Gradient Boosting | 75.75% | 91.50% | 98.38% | 0.8630 | 86.57% | 0.1095 | 0.6127 |

Histogram Gradient Boosting provides the strongest ranking and lowest log loss in this experiment. It improves Top-1 accuracy by 8.87 percentage points over Random Forest, but is overconfident: its constrained confidence is 86.57% at 75.75% accuracy.

Logistic Regression demonstrates that the engineered features expose substantial linearly usable information, although its probabilities are also overconfident. Extra Trees improves Top-1 over Random Forest but does not provide a clear overall advantage.

Candidate constraints improve Logistic Regression's ranking, while all three tree ensembles already place almost all useful probability mass on valid candidates. Calibration quality and ranking quality therefore remain separate model-selection criteria.

### Repeated Classifier and Runtime Comparison

The repeated classifier comparison evaluates model quality and computational cost across three seeds and removal rates. Every run creates a new solution-level split, instantiates all four classifiers, measures fitting and one complete test-set probability prediction, and then calculates raw and candidate-constrained metrics.

`TimedModelComparisonResult` extends the comparison output with training and inference durations measured by `perf_counter()`. Feature generation and metric calculation are outside the inference measurement. `RepeatedModelResult` aggregates runtime, ranking, confidence, ECE, and log loss through `MetricSummary`.

```text
Removal rate and seed
          |
          v
New solution-level split
          |
          v
Create four classifiers
          |
     +----+----------------+
     |                     |
     v                     v
Timed fit            Timed predict_proba
     |                     |
     +----------+----------+
                |
                v
Raw and constrained metrics
                |
                v
Aggregate across seeds
```

Mean candidate-constrained ranking results were:

| Removal | Model | Top-1 | Top-1 SD | Top-3 | MRR |
|---:|---|---:|---:|---:|---:|
| 50% | Logistic Regression | 66.83% | 1.46% | 98.46% | 0.8172 |
| 50% | Random Forest | 64.88% | 1.69% | 97.87% | 0.8048 |
| 50% | Extra Trees | 65.96% | 2.77% | 97.83% | 0.8099 |
| 50% | Histogram Gradient Boosting | 72.92% | 2.35% | 98.38% | 0.8485 |
| 60% | Logistic Regression | 51.35% | 0.52% | 94.31% | 0.7165 |
| 60% | Random Forest | 47.67% | 1.85% | 92.71% | 0.6919 |
| 60% | Extra Trees | 47.26% | 1.08% | 92.22% | 0.6885 |
| 60% | Histogram Gradient Boosting | 56.39% | 1.07% | 94.03% | 0.7445 |
| 65% | Logistic Regression | 43.17% | 0.48% | 88.24% | 0.6546 |
| 65% | Random Forest | 40.67% | 1.26% | 86.63% | 0.6372 |
| 65% | Extra Trees | 39.58% | 2.20% | 85.93% | 0.6280 |
| 65% | Histogram Gradient Boosting | 47.47% | 1.20% | 88.11% | 0.6793 |

Mean candidate-constrained probability quality was:

| Removal | Model | ECE | Log loss |
|---:|---|---:|---:|
| 50% | Logistic Regression | 0.1796 | 0.9460 |
| 50% | Random Forest | 0.0724 | 0.7172 |
| 50% | Extra Trees | 0.0847 | 0.7200 |
| 50% | Histogram Gradient Boosting | 0.1312 | 0.6668 |
| 60% | Logistic Regression | 0.2263 | 1.3073 |
| 60% | Random Forest | 0.0395 | 1.0333 |
| 60% | Extra Trees | 0.0388 | 1.0388 |
| 60% | Histogram Gradient Boosting | 0.1490 | 0.9781 |
| 65% | Logistic Regression | 0.2438 | 1.4836 |
| 65% | Random Forest | 0.0249 | 1.1990 |
| 65% | Extra Trees | 0.0236 | 1.2069 |
| 65% | Histogram Gradient Boosting | 0.1479 | 1.1834 |

Mean runtime results in milliseconds were:

| Removal | Model | Training | Inference |
|---:|---|---:|---:|
| 50% | Logistic Regression | 132.74 | 0.53 |
| 50% | Random Forest | 118.13 | 18.26 |
| 50% | Extra Trees | 93.21 | 29.63 |
| 50% | Histogram Gradient Boosting | 2,422.38 | 10.69 |
| 60% | Logistic Regression | 118.59 | 0.57 |
| 60% | Random Forest | 124.95 | 25.46 |
| 60% | Extra Trees | 101.73 | 25.86 |
| 60% | Histogram Gradient Boosting | 1,636.25 | 11.92 |
| 65% | Logistic Regression | 124.68 | 0.59 |
| 65% | Random Forest | 126.69 | 25.61 |
| 65% | Extra Trees | 106.04 | 25.83 |
| 65% | Histogram Gradient Boosting | 1,601.89 | 14.08 |

Histogram Gradient Boosting consistently leads Top-1, MRR, and log loss. Its Top-1 advantage over Random Forest is 8.04, 8.72, and 6.80 percentage points across increasing removal rates. Logistic Regression is consistently second in Top-1 and has marginally better Top-3 at every evaluated rate.

Gradient Boosting training is approximately 13 to 20 times slower than Random Forest training, but its batch inference is faster than both bagged tree ensembles. Logistic Regression is by far the fastest inference model. These timings cover one batch prediction over the test set and do not represent the solver's repeated small predictions or feature-generation cost.

Random Forest and Extra Trees remain better calibrated after candidate masking, while Gradient Boosting has the lowest log loss. Model selection therefore depends on whether ranking, probability calibration, training cost, or inference cost is prioritized.

### Probability-Model Protocol

The Hybrid solver depends on model behavior rather than a concrete classifier. `SudokuProbabilityModel` defines the structural interface required for candidate ranking:

```text
SudokuProbabilityModel
    |
    +-- classes
    |
    +-- predict_probabilities(X)
```

`NamedSudokuProbabilityModel` additionally requires a `name` attribute for comparison reports. Both are Python protocols, so compatible classes do not need explicit inheritance.

```text
SudokuRandomForest --------+
ProbabilityModelAdapter ---+--> SudokuProbabilityModel
CalibratedProbabilityModel +
                                  |
                                  v
                         HybridSudokuSolver
```

This removes the solver's type-level dependency on `SudokuRandomForest` without changing its runtime behavior. Candidate validity, feature generation, ranking, and backtracking remain unchanged.

### Classifier Comparison in the Hybrid Solver

The end-to-end comparison trains all four classifiers on identical data and evaluates a fresh `HybridSudokuSolver` for each model on identical uniquely solvable puzzles. `compare_models_in_hybrid_solver()` reuses the established `evaluate_solver()` logic, including validity and exact ground-truth checks.

```text
Shared training split
        |
        v
Train four classifiers
        |
        v
Shared unique-solution puzzles
        |
   +----+----+----+----+
   |         |         |
   v         v         v
Hybrid     Hybrid     Hybrid
with       with       with
different probability models
   |         |         |
   +----+----+----+----+
        |
        v
Exact match, runtime, decisions, backtracks
```

Every model-guided solver achieved 100% exact match and 100% validity at all evaluated removal rates. At 50% removal, all puzzles were solved through 40 deterministic placements per puzzle with no ML decisions, so model differences were not exercised.

End-to-end runtime means were:

| Removal | Logistic Regression | Random Forest | Extra Trees | Histogram Gradient Boosting |
|---:|---:|---:|---:|---:|
| 50% | 0.74 ms | 0.72 ms | 0.72 ms | 0.71 ms |
| 60% | 2.18 ms | 19.81 ms | 18.21 ms | 10.49 ms |
| 65% | 4.49 ms | 62.65 ms | 56.41 ms | 27.56 ms |

Search-effort means were:

| Removal | Model | Deterministic steps | ML decisions | Backtracks |
|---:|---|---:|---:|---:|
| 50% | Logistic Regression | 40.00 | 0.00 | 0.00 |
| 50% | Random Forest | 40.00 | 0.00 | 0.00 |
| 50% | Extra Trees | 40.00 | 0.00 | 0.00 |
| 50% | Histogram Gradient Boosting | 40.00 | 0.00 | 0.00 |
| 60% | Logistic Regression | 57.70 | 1.10 | 11.55 |
| 60% | Random Forest | 58.00 | 1.10 | 11.80 |
| 60% | Extra Trees | 58.25 | 1.10 | 12.05 |
| 60% | Histogram Gradient Boosting | 56.90 | 1.20 | 10.85 |
| 65% | Logistic Regression | 82.70 | 4.50 | 38.55 |
| 65% | Random Forest | 84.15 | 3.95 | 38.85 |
| 65% | Extra Trees | 74.90 | 3.55 | 28.45 |
| 65% | Histogram Gradient Boosting | 79.90 | 3.50 | 33.55 |

Histogram Gradient Boosting has the lowest backtracking at 60%, while Extra Trees reduces backtracking most at 65%. Its superior general cell-level Top-1 accuracy does not translate into uniformly superior solver search because the solver queries only ambiguous cells selected by minimum remaining values, not every removed cell represented in the classification test set.

Logistic Regression has similar or greater search effort but is approximately nine times faster than Random Forest at 60% removal and fourteen times faster at 65%. Its inexpensive probability inference more than compensates for additional search. It is therefore the strongest current candidate when end-to-end runtime is prioritized.

Deterministic-step counts can exceed the number of initially empty cells because forced placements on failed search branches are counted again after backtracking.

## Current Limitation

The current Random Forest cannot reliably solve ambiguous puzzles without correction: Greedy exact match falls to 55% at 60% removal and 25% at 65% removal. Its 66.88% Top-1 accuracy is useful for search ordering but is insufficient for an unrecoverable decision sequence. The model remains a cell-level classifier rather than an end-to-end grid model.

The repeated ablation covers three seeds and three removal rates, but this is still a small empirical sample. The raw grid contributes little predictive ability by itself, and most performance comes from explicitly encoded Sudoku constraints. The current model therefore remains dependent on feature engineering rather than learning Sudoku rules directly.

Repeated evaluation shows that Sigmoid reliably improves raw probability quality but is not uniformly beneficial after candidate masking. A single global calibration strategy is therefore not justified for the solver.

The first end-to-end comparison shows that Logistic Regression is the fastest model-guided solver, but it uses one training seed, one evaluation seed, and 20 puzzles per removal rate. The model-dependent search-effort ordering changes with removal rate, so repeated end-to-end evaluation is still required.

`SudokuHistogramGradientBoosting` now provides the same training, prediction, probability, evaluation, and persistence operations as `SudokuRandomForest`. The CLI can select either persistent model, while Random Forest remains the default.

## Persistent Histogram Gradient Boosting Model

The classifier comparisons identified Histogram Gradient Boosting as the strongest general cell-level classifier. It is now represented by a dedicated project wrapper instead of being available only through the temporary comparison adapter.

```text
MLDataSplit
    |
    v
SudokuHistogramGradientBoosting
    |
    +-- fit(data)
    +-- fit_arrays(X, y)
    +-- predict(X)
    +-- predict_probabilities(X)
    +-- evaluate(data)
    +-- classes
    +-- save(path)
    +-- load(path)
```

The wrapper owns a scikit-learn `HistGradientBoostingClassifier`. Model files contain the fitted estimator and are serialized with Joblib. Loading validates the estimator type before exposing it through the project API.

The training script uses 100 generated solutions, a solution-level 80/20 split, 50% removal, seed 42, the complete 118-feature representation, and 100 boosting iterations. The resulting artifact is written to `models/sudoku_histogram_gradient_boosting.joblib` and remains excluded from Git.

The Random Forest remains the CLI default, but model type and model path can now be selected independently.

## CLI Model Selection

The CLI separates the logical model type from the serialized model path:

```text
--model-type
    |
    +-- random-forest
    |       |
    |       +--> SudokuRandomForest.load(...)
    |
    +-- histogram-gradient-boosting
            |
            +--> SudokuHistogramGradientBoosting.load(...)
```

If `--model` is omitted, the selected model type determines the default artifact path. An explicit path overrides only the location, not the expected estimator type. Each wrapper validates the loaded estimator, so selecting Histogram Gradient Boosting while passing a Random Forest file produces a clear command-line error.

The classical solver path bypasses model loading entirely. This preserves operation without a trained model and keeps classical candidate ordering independent of the model-selection options.

## Greedy Model Decision Trace

`GreedyMLSudokuSolver` represents the project's constrained Model-only path. It still uses Sudoku rules to select the most constrained cell and exclude illegal digits, but it permanently accepts the highest-ranked valid candidate and never backtracks.

```text
Select MRV cell
      |
      v
Determine valid candidates
      |
      +-- one candidate ----> deterministic placement
      |
      +-- several ----------> model ranking ----> Top-1 placement
                                                    |
                                                    v
                                         append GreedyDecision
                                                    |
                                                    v
                                           never reconsider
```

Every placement is stored as an immutable `GreedyDecision` containing:

- one-based step number,
- zero-based row and column,
- sorted valid candidates,
- candidates in model-ranked order,
- selected digit,
- confidence in the selected ML digit,
- deterministic or ML-decision classification.

For single-candidate placements, `confidence` is `None` because the digit is selected by Sudoku constraints rather than model inference. For ambiguous placements, confidence is the model's raw probability for the selected valid digit.

The trace is reset before every solving attempt and exposed as an immutable tuple. It contains no rollback events because the Greedy solver performs no backtracking. Comparing the trace with a unique ground-truth solution can therefore locate the first incorrect irreversible choice and determine the rank of the correct digit.

## Model-only Ground-Truth Analysis

`analyze_model_only_attempt()` validates a puzzle and expected solution, runs a fresh Greedy solver, and compares every traced placement with the digit at the same ground-truth position. Comparison stops at the first mismatch because every later state already depends on that irreversible error.

```text
Puzzle + expected solution + probability model
                      |
                      v
               Greedy solve attempt
                      |
                      v
                 decision_trace
                      |
                      v
          compare each selected digit
                      |
          +-----------+-----------+
          |                       |
       correct                 first error
          |                       |
     next decision       digit, rank, confidence,
                         candidates, position
```

`ModelOnlyPuzzleResult` distinguishes exact match, valid completion, and completion. It also reports trace length, ML-decision count, correct placements before the first error, and the optional `ModelOnlyDecisionError`.

`compare_model_only_models()` applies the same analysis to named probability models on identical puzzle and ground-truth arrays. Aggregate properties report exact solution rate, failure rate, average correct decisions before error, average first-error confidence, and average correct-digit rank.

The first experiment produced:

| Removal | Model | Exact | Failure | Correct before error | Error confidence | Correct rank |
|---:|---|---:|---:|---:|---:|---:|
| 50% | Random Forest | 100.00% | 0.00% | n/a | n/a | n/a |
| 50% | Histogram Gradient Boosting | 100.00% | 0.00% | n/a | n/a | n/a |
| 60% | Random Forest | 55.00% | 45.00% | 13.67 | 31.22% | 2.00 |
| 60% | Histogram Gradient Boosting | 65.00% | 35.00% | 14.57 | 72.05% | 2.00 |
| 65% | Random Forest | 25.00% | 75.00% | 13.60 | 29.60% | 2.00 |
| 65% | Histogram Gradient Boosting | 30.00% | 70.00% | 9.86 | 68.52% | 2.00 |

Histogram Gradient Boosting improves complete Model-only success, but its wrong first choices receive more than twice the confidence of Random Forest errors. Both models rank the correct digit second on average at the first error. This provides direct motivation for retaining more than one candidate path through Beam Search.

## Bounded Beam Search

`BeamSearchSudokuSolver` keeps a bounded collection of partial grids. Every state contains an independent grid copy and a cumulative model score. This allows a lower-ranked alternative to survive after the current Top-1 path encounters a contradiction.

```text
Active states
      |
      v
Select one MRV cell per state
      |
      +-- single candidate --> one score-neutral child
      |
      +-- ambiguous --------> one child per valid candidate
                                      |
                                      v
                           add log(probability)
                                      |
                                      v
                         sort all generated children
                                      |
                                      v
                       retain best beam_width states
```

Log-probabilities convert multiplication across a decision sequence into numerically stable addition:

```text
path score = log(p1) + log(p2) + ... + log(pn)
```

Deterministic placements do not change the score because Sudoku constraints already determine their digit. Zero model probabilities are replaced with the smallest positive floating-point value before taking the logarithm, allowing every rule-valid candidate to remain representable.

`BeamSearchStats` extends the common statistics with:

- `generated_states`,
- `pruned_states`,
- `max_active_states`.

`backtracks` remains zero because the algorithm does not recursively undo a placement. Failed states disappear from the active collection, and low-scoring states are pruned when the beam limit is exceeded.

Beam width 1 behaves as a single-path bounded search. Larger widths trade memory and inference work for a greater chance of retaining the correct alternative. The known Random Forest Greedy-failure puzzle remains unsolved at widths 2 and 3 but is solved at width 4. This single fixture demonstrates recovery capability, not general model quality.

## Search-Strategy Evaluation

The Beam comparison reuses `evaluate_solver()` for every strategy. The common result now reads optional Beam statistics through the solver's stats object while returning zero for solvers that do not expose them.

```text
Shared unique puzzles and ground truth
                  |
        +---------+---------+
        |                   |
   Classical          Probability model
                            |
                 +----------+----------+
                 |          |          |
              Greedy    Beam 2/3/4   Hybrid
                 |          |          |
                 +----------+----------+
                            |
             exact match, validity, runtime,
             search effort and Beam states
```

The classical solver is evaluated once per removal rate because it has no model dependency. Every model-guided strategy receives the same puzzle arrays and expected solutions. This prevents puzzle selection from confounding model or search-strategy differences.

Exact-match results at ambiguous removal rates are:

| Removal | Model | Greedy | Beam 2 | Beam 3 | Beam 4 | Hybrid |
|---:|---|---:|---:|---:|---:|---:|
| 60% | Random Forest | 55% | 75% | 90% | 100% | 100% |
| 60% | Histogram Gradient Boosting | 65% | 80% | 85% | 100% | 100% |
| 65% | Random Forest | 25% | 60% | 70% | 80% | 100% |
| 65% | Histogram Gradient Boosting | 30% | 75% | 80% | 90% | 100% |

Runtime means in milliseconds are:

| Removal | Model | Greedy | Beam 2 | Beam 3 | Beam 4 | Hybrid |
|---:|---|---:|---:|---:|---:|---:|
| 60% | Random Forest | 14.96 | 17.95 | 20.24 | 23.14 | 17.78 |
| 60% | Histogram Gradient Boosting | 7.83 | 9.62 | 10.93 | 11.97 | 11.96 |
| 65% | Random Forest | 33.46 | 47.63 | 59.16 | 68.58 | 61.73 |
| 65% | Histogram Gradient Boosting | 16.07 | 25.25 | 30.44 | 30.37 | 26.23 |

Beam width increases solution quality but generally raises state generation and runtime. At 65% removal, Beam 4 generates 89.05 states per Random Forest puzzle and 92.55 per Gradient Boosting puzzle. It still prunes 1.80 states per puzzle for both models, which explains why exact match remains below 100%.

Histogram Gradient Boosting provides the strongest bounded-search result: 90% exact match at 65% removal with Beam 4. Hybrid remains fully reliable and slightly faster for that model. The classical solver also remains fully reliable and is much faster because feature generation and model inference are unnecessary.

## Repeated Beam Evaluation

The repeated comparison changes both the training seed and puzzle-generation seed for every run. The evaluation seed is derived from the training seed with an offset of 10,000, ensuring that training and evaluation generation do not reuse the same random sequence.

```text
random seed
    |
    +-- training seed: seed
    |
    +-- evaluation seed: seed + 10,000
```

For each removal rate and seed, both models are retrained and all strategies receive the same newly generated unique-solution puzzles. Metrics from individual runs are summarized with mean, population standard deviation, minimum, and maximum through the existing `MetricSummary` type.

At 65% removal, the repeated exact-match results are:

| Model | Greedy | Beam 2 | Beam 4 | Hybrid |
|---|---:|---:|---:|---:|
| Random Forest | 33.33% ± 4.71% | 53.33% ± 4.71% | 76.67% ± 9.43% | 100.00% ± 0.00% |
| Histogram Gradient Boosting | 43.33% ± 20.55% | 56.67% ± 24.94% | 73.33% ± 18.86% | 100.00% ± 0.00% |

The single-run advantage of Histogram Gradient Boosting Beam 4 does not persist in the repeated 65% result. Random Forest has a slightly higher mean and substantially lower variation, although three runs are insufficient to establish a statistically reliable model advantage.

Histogram Gradient Boosting remains much faster. Its 65% Beam-4 runtime is `44.41 ms ± 9.67 ms`, compared with `88.33 ms ± 17.12 ms` for Random Forest. Its Hybrid solver reaches 100% exact match in `43.32 ms ± 17.95 ms`, making it both more reliable and slightly faster than its bounded Beam-4 configuration.

## End-to-End Full-Grid Dataset

The new modeling phase changes the prediction task from one selected cell to the complete Sudoku grid.

```text
Previous tabular task
118 features for one empty cell
             |
             v
       one digit class

End-to-end task
incomplete 9 x 9 grid
             |
             v
     complete 9 x 9 grid
```

`EndToEndDataset` stores five aligned arrays:

- `X`: incomplete grids with zero-valued empty cells,
- `y`: complete solution grids,
- `empty_mask`: Boolean positions that require prediction,
- `removal_rates`: difficulty proxy used for each sample,
- `solution_ids`: identity of the source solution family.

For every independently generated complete solution, the generator creates one unique-solution puzzle at each requested removal rate. All variants retain the same `solution_id`.

```text
Source solution 17
    |
    +-- 50% removal puzzle
    +-- 60% removal puzzle
    +-- 65% removal puzzle
```

The train/test split shuffles unique solution IDs rather than individual samples. All variants of one solution therefore remain in the same partition. This prevents the model from being evaluated on a differently masked version of a solution already seen during training.

```text
100 source solutions
        |
        +-- 80 train IDs x 3 rates = 240 samples
        |
        +-- 20 test IDs  x 3 rates =  60 samples
```

Inputs and targets are copied into independent NumPy arrays. This protects complete targets from accidental modification when puzzles are transformed during later preprocessing or iterative prediction.

The inspection reports 40, 48, and 52 removed cells for rates 50%, 60%, and 65%. Their balanced mean is 46.67 empty cells per sample. Training and test partitions contain zero shared solution IDs.

## End-to-End MLP Baseline

The first full-grid baseline uses one shared `MLPClassifier`. Scikit-learn does not provide a native 81-position, nine-class output head in the same form as a neural-network framework, so the model reformulates every hidden cell as a position-conditioned training example.

```text
Incomplete 9 x 9 grid       Target position
        81 values      +    81-value one-hot vector
                  \          /
                   \        /
                    162 inputs
                         |
                         v
                    shared MLP
                         |
                         v
                9 digit probabilities
```

Only originally empty cells become training targets. During prediction, the same incomplete grid is combined with all 81 position vectors, producing an array with shape `(samples, 81, 9)`. The highest-probability digit is selected independently at every position, and given clues are restored unchanged.

The approach uses no candidate filtering, MRV selection, iterative correction, Beam Search, or backtracking. It is therefore a direct model baseline, although clue preservation is enforced as output postprocessing.

The evaluation distinguishes local and global quality:

- accuracy on originally empty cells,
- Top-3 accuracy on originally empty cells,
- exact complete-grid match,
- valid Sudoku rate,
- clue preservation,
- incorrect empty cells per puzzle,
- violated row, column, and block units.

There are 27 Sudoku units: nine rows, nine columns, and nine blocks. A unit counts as violated when its nine predicted values do not contain nine distinct digits.

The first evaluation produces:

| Metric | Result | Random reference |
|---|---:|---:|
| Empty-cell accuracy | 11.74% | 11.11% |
| Empty-cell Top-3 accuracy | 33.91% | 33.33% |
| Exact solution rate | 0.00% | — |
| Valid solution rate | 0.00% | — |
| Clue preservation rate | 100.00% | — |
| Incorrect empty cells | 41.19 | — |
| Rule violations | 26.89 of 27 | — |

The MLP performs approximately like random digit ranking on unseen source solutions. Nearly every row, column, and block is invalid. Preserved clues result from explicit copying rather than learned behavior.

Training also reaches the configured limit of 100 iterations without convergence. More iterations may improve optimization, but the near-random test result and almost universal rule violations indicate a representational and data-efficiency problem rather than only an early stopping point.

## End-to-End MLP Learning Curve

The learning-curve evaluation keeps one test partition fixed and selects nested source-solution families from the training partition. Every point trains a fresh MLP with identical architecture, iteration limit, and random seed.

```text
Fixed train partition
        |
        +-- first 50 solution families  --> MLP 1
        +-- first 100 solution families --> MLP 2
        +-- first 200 solution families --> MLP 3
        +-- first 400 solution families --> MLP 4

Fixed test partition <---------------- evaluate every MLP
```

The result separates memorization from generalization:

| Source solutions | Train accuracy | Test accuracy | Accuracy gap | Train exact | Test exact |
|---:|---:|---:|---:|---:|---:|
| 50 | 100.00% | 11.34% | 88.66% | 100.00% | 0.00% |
| 100 | 99.46% | 11.39% | 88.06% | 80.00% | 0.00% |
| 200 | 68.46% | 11.58% | 56.88% | 0.00% | 0.00% |
| 400 | 45.13% | 11.74% | 33.38% | 0.00% | 0.00% |

Small datasets are memorized almost perfectly. The 50-solution model reconstructs every training grid as a valid Sudoku but remains at random accuracy on unseen solution families. This confirms that the solution-level split successfully prevents memorized grids from inflating test performance.

As the dataset grows, fixed model capacity and 100 optimizer iterations are no longer sufficient to memorize all training samples. Training accuracy falls and final loss rises, but test accuracy remains effectively unchanged. More examples alone therefore do not produce general Sudoku reasoning in this architecture.

All four models reach the 100-iteration limit and emit `ConvergenceWarning`. Non-convergence contributes to the larger-dataset underfitting, but it cannot explain why the fully memorized 50-solution model also performs randomly on the fixed test set.
