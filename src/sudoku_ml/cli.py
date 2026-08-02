import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from sudoku_ml.grid import SudokuGrid
from sudoku_ml.model.histogram_gradient_boosting import (
    SudokuHistogramGradientBoosting,
)
from sudoku_ml.model.protocol import SudokuProbabilityModel
from sudoku_ml.model.random_forest import SudokuRandomForest
from sudoku_ml.solver import ClassicalSudokuSolver, HybridSudokuSolver


PACKAGE_NAME = "ml-sudoku-solver"
RANDOM_FOREST = "random-forest"
HISTOGRAM_GRADIENT_BOOSTING = "histogram-gradient-boosting"
MODEL_TYPES = (
    RANDOM_FOREST,
    HISTOGRAM_GRADIENT_BOOSTING,
)
DEFAULT_MODEL_PATHS = {
    RANDOM_FOREST: Path("models/sudoku_random_forest.joblib"),
    HISTOGRAM_GRADIENT_BOOSTING: Path(
        "models/sudoku_histogram_gradient_boosting.joblib"
    ),
}
MODEL_DISPLAY_NAMES = {
    RANDOM_FOREST: "Random Forest",
    HISTOGRAM_GRADIENT_BOOSTING: "Histogram Gradient Boosting",
}


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

def get_version() -> str:
    """Return the installed package version or a development fallback."""
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "development"

def create_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Solve a Sudoku with the hybrid ML-guided solver."
        )
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"{PACKAGE_NAME} {get_version()}",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "puzzle",
        nargs="?",
        help=(
            "Sudoku as 81 cells. Use 0 or . for empty cells."
        ),
    )

    input_group.add_argument(
        "-f",
        "--input-file",
        type=Path,
        help="Read the Sudoku from a text file.",
    )

    parser.add_argument(
        "--classical",
        action="store_true",
        help="Use classical candidate ordering without a saved model.",
    )

    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPES,
        default=RANDOM_FOREST,
        help=(
            "Model type used for ML-guided solving "
            f"(default: {RANDOM_FOREST})."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Path to a saved model. If omitted, the default path "
            "for the selected model type is used."
        ),
    )

    return parser


def read_puzzle_text(
    puzzle: str | None,
    input_file: Path | None,
) -> str:
    """Return puzzle text from a direct argument or input file."""
    if input_file is not None:
        return input_file.read_text(encoding="utf-8")

    if puzzle is None:
        raise ValueError("A Sudoku puzzle is required.")

    return puzzle


def load_model(
    model_type: str,
    model_path: Path | None,
) -> SudokuProbabilityModel:
    """Load the selected probability model from disk."""
    resolved_path = (
        model_path
        if model_path is not None
        else DEFAULT_MODEL_PATHS[model_type]
    )

    if model_type == RANDOM_FOREST:
        return SudokuRandomForest.load(resolved_path)

    return SudokuHistogramGradientBoosting.load(resolved_path)


def main(arguments: list[str] | None = None) -> int:
    """Run the Sudoku solver command-line interface."""
    parser = create_argument_parser()
    parsed_arguments = parser.parse_args(arguments)

    try:
        puzzle_text = read_puzzle_text(
            parsed_arguments.puzzle,
            parsed_arguments.input_file,
        )
        puzzle = parse_grid(puzzle_text)

        if parsed_arguments.classical:
            solver = ClassicalSudokuSolver()
            solver_name = "classical"
        else:
            model = load_model(
                model_type=parsed_arguments.model_type,
                model_path=parsed_arguments.model,
            )
            solver = HybridSudokuSolver(model)
            solver_name = (
                "hybrid ML-guided "
                f"({MODEL_DISPLAY_NAMES[parsed_arguments.model_type]})"
            )

        solution = solver.solve(puzzle)
    except (ValueError, TypeError, OSError) as error:
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
    print(f"Solver:              {solver_name}")
    print(
        "Deterministic steps: "
        f"{solver.stats.deterministic_steps}"
    )
    print(
        "ML decisions:        "
        f"{solver.stats.ml_decisions}"
    )
    print(
        "Branching decisions: "
        f"{solver.stats.branching_decisions}"
    )
    print(
        "Backtracks:          "
        f"{solver.stats.backtracks}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
