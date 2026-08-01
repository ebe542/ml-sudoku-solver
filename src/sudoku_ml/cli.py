import argparse
from pathlib import Path

import numpy as np

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import HybridSudokuSolver


def parse_grid(text: str) -> SudokuGrid:
    """Parse an 81-character Sudoku string."""
    normalized = "".join(text.split())

    if len(normalized) != 81:
        raise ValueError(
            "Sudoku input must contain exactly 81 cells."
        )

    if any(character not in "0123456789." for character in normalized):
        raise ValueError(
            "Sudoku input may only contain digits, dots, and whitespace."
        )

    values = np.array(
        [
            0 if character == "." else int(character)
            for character in normalized
        ],
        dtype=int,
    ).reshape(9, 9)

    return SudokuGrid(values)

def format_grid(grid: SudokuGrid) -> str:
    """Format a Sudoku grid for terminal output."""
    lines: list[str] = []

    for row_index, row in enumerate(grid.values):
        cells = [
            "." if value == 0 else str(value)
            for value in row
        ]

        lines.append(
            " ".join(cells[0:3])
            + " | "
            + " ".join(cells[3:6])
            + " | "
            + " ".join(cells[6:9])
        )

        if row_index in (2, 5):
            lines.append("------+-------+------")

    return "\n".join(lines)

DEFAULT_MODEL_PATH = Path(
    "models/sudoku_random_forest.joblib"
)

def create_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Solve a Sudoku with the hybrid ML-guided solver."
        )
    )

    parser.add_argument(
        "puzzle",
        help=(
            "Sudoku as 81 cells. Use 0 or . for empty cells."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to the saved Random Forest model "
            f"(default: {DEFAULT_MODEL_PATH})."
        ),
    )

    return parser


def main(arguments: list[str] | None = None) -> int:
    """Run the Sudoku solver command-line interface."""
    parser = create_argument_parser()
    parsed_arguments = parser.parse_args(arguments)

    try:
        puzzle = parse_grid(parsed_arguments.puzzle)
        model = SudokuRandomForest.load(
            parsed_arguments.model
        )
        solver = HybridSudokuSolver(model)
        solution = solver.solve(puzzle)
    except (ValueError, TypeError, FileNotFoundError) as error:
        parser.error(str(error))

    if solution is None:
        print("No solution found.")
        return 1

    print("Input")
    print("-----")
    print(format_grid(puzzle))
    print()

    print("Solution")
    print("--------")
    print(format_grid(solution))
    print()

    print("Solver Statistics")
    print("-----------------")
    print(
        "Deterministic steps: "
        f"{solver.stats.deterministic_steps}"
    )
    print(
        "ML decisions:        "
        f"{solver.stats.ml_decisions}"
    )
    print(
        "Backtracks:          "
        f"{solver.stats.backtracks}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
