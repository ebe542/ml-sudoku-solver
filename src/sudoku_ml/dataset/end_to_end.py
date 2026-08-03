from dataclasses import dataclass

import numpy as np

from sudoku_ml.dataset.unique_generator import create_unique_puzzle
from sudoku_ml.sudoku_generator import generate_solved_grid


@dataclass(frozen=True)
class EndToEndDataset:
    """Store incomplete grids and complete grid targets."""

    X: np.ndarray
    y: np.ndarray
    empty_mask: np.ndarray
    removal_rates: np.ndarray
    solution_ids: np.ndarray


@dataclass(frozen=True)
class EndToEndDataSplit:
    """Store a solution-level end-to-end train/test split."""

    train: EndToEndDataset
    test: EndToEndDataset


def _validate_removal_rates(removal_rates: tuple[float, ...]) -> None:
    if not removal_rates:
        raise ValueError("At least one removal rate is required.")

    if any(not 0.0 <= rate < 1.0 for rate in removal_rates):
        raise ValueError("Removal rates must be between 0 and 1.")


def create_end_to_end_dataset(
    num_solutions: int = 100,
    removal_rates: tuple[float, ...] = (0.50,),
    random_seed: int | None = None,
) -> EndToEndDataset:
    """Create full-grid samples from independent Sudoku solutions."""
    if num_solutions <= 0:
        raise ValueError("Number of solutions must be positive.")

    _validate_removal_rates(removal_rates)
    rng = np.random.default_rng(random_seed)
    puzzles: list[np.ndarray] = []
    solutions: list[np.ndarray] = []
    sample_rates: list[float] = []
    solution_ids: list[int] = []

    for solution_id in range(num_solutions):
        solution_seed = int(
            rng.integers(0, np.iinfo(np.int32).max)
        )
        solution = generate_solved_grid(random_seed=solution_seed)

        for removal_rate in removal_rates:
            puzzle_seed = int(
                rng.integers(0, np.iinfo(np.int32).max)
            )
            puzzle = create_unique_puzzle(
                solution,
                removal_rate=removal_rate,
                random_seed=puzzle_seed,
            )

            puzzles.append(puzzle.values.copy())
            solutions.append(solution.values.copy())
            sample_rates.append(removal_rate)
            solution_ids.append(solution_id)

    X = np.asarray(puzzles, dtype=np.int8)
    y = np.asarray(solutions, dtype=np.int8)

    return EndToEndDataset(
        X=X,
        y=y,
        empty_mask=X == 0,
        removal_rates=np.asarray(sample_rates, dtype=float),
        solution_ids=np.asarray(solution_ids, dtype=int),
    )


def _select_samples(
    dataset: EndToEndDataset,
    solution_ids: np.ndarray,
) -> EndToEndDataset:
    sample_mask = np.isin(dataset.solution_ids, solution_ids)
    X = dataset.X[sample_mask].copy()

    return EndToEndDataset(
        X=X,
        y=dataset.y[sample_mask].copy(),
        empty_mask=X == 0,
        removal_rates=dataset.removal_rates[sample_mask].copy(),
        solution_ids=dataset.solution_ids[sample_mask].copy(),
    )


def create_end_to_end_train_test_split(
    num_solutions: int = 100,
    test_size: float = 0.2,
    removal_rates: tuple[float, ...] = (0.50,),
    random_seed: int | None = None,
) -> EndToEndDataSplit:
    """Split full-grid samples without sharing source solutions."""
    if num_solutions <= 1:
        raise ValueError("At least two Sudoku solutions are required.")

    if not 0.0 < test_size < 1.0:
        raise ValueError("Test size must be between 0 and 1.")

    dataset = create_end_to_end_dataset(
        num_solutions=num_solutions,
        removal_rates=removal_rates,
        random_seed=random_seed,
    )
    rng = np.random.default_rng(random_seed)
    unique_solution_ids = np.arange(num_solutions)
    rng.shuffle(unique_solution_ids)
    test_count = max(1, int(num_solutions * test_size))
    test_solution_ids = unique_solution_ids[:test_count]
    train_solution_ids = unique_solution_ids[test_count:]

    return EndToEndDataSplit(
        train=_select_samples(dataset, train_solution_ids),
        test=_select_samples(dataset, test_solution_ids),
    )
