# ML Sudoku Solver

> A self-directed machine learning project for analyzing and solving Sudoku grids.

## Project Status

Early development.

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
python -m sudoku_ml.analysis.analyze_feature_importance
```
