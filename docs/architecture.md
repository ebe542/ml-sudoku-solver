# Architecture

## Current Pipeline


```
        ┌─────────────────────┐
        │ SudokuGrid          │
        │ Representation &    │
        │ Validation          │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Sudoku Generator    │
        │ Generate complete   │
        │ valid Sudokus       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Dataset Generator   │
        │ Remove cells /      │
        │ create puzzles      │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ SudokuDataset       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Feature             │
        │ Preprocessing       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ ML Model            │
        │ (not implemented)   │
        └─────────────────────┘
```
