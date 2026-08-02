from pathlib import Path

import numpy as np
import pytest

from sudoku_ml.cli import format_grid, main, parse_grid
from sudoku_ml.data.split import create_train_test_split
from sudoku_ml.model.histogram_gradient_boosting import (
    SudokuHistogramGradientBoosting,
)
from sudoku_ml.model.random_forest import SudokuRandomForest

PUZZLE_TEXT = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)


def test_parse_grid_creates_sudoku_grid() -> None:
    grid = parse_grid(PUZZLE_TEXT)

    assert grid.values.shape == (9, 9)
    assert grid.values[0, 0] == 5
    assert grid.values[0, 2] == 0
    assert len(grid.empty_cells) == 51


def test_parse_grid_accepts_dots_for_empty_cells() -> None:
    grid_with_zeros = parse_grid(PUZZLE_TEXT)
    grid_with_dots = parse_grid(PUZZLE_TEXT.replace("0", ".")
    )

    assert np.array_equal(
        grid_with_zeros.values,
        grid_with_dots.values,
    )


def test_parse_grid_ignores_whitespace() -> None:
    rows = [
        PUZZLE_TEXT[index : index + 9]
        for index in range(0, 81, 9)
    ]
    multiline_text = "\n".join(rows)

    grid = parse_grid(multiline_text)

    assert grid.values.shape == (9, 9)
    assert len(grid.empty_cells) == 51


def test_parse_grid_rejects_incorrect_length() -> None:
    with pytest.raises(ValueError, match="exactly 81 cells"):
        parse_grid("123")


def test_parse_grid_rejects_invalid_character() -> None:
    invalid_text = "X" + PUZZLE_TEXT[1:]

    with pytest.raises(ValueError, match="digits, dots, and whitespace"):
        parse_grid(invalid_text)

def test_format_grid_creates_readable_output() -> None:
    grid = parse_grid(PUZZLE_TEXT)

    formatted = format_grid(grid)
    lines = formatted.splitlines()

    assert len(lines) == 11
    assert lines[0] == "5 3 . | . 7 . | . . ."
    assert lines[3] == "------+-------+------"
    assert lines[7] == "------+-------+------"
    assert lines[-1] == ". . . | . 8 . | . 7 9"

def test_main_solves_puzzle_with_saved_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = create_train_test_split(
        num_solutions=20,
        random_seed=42,
    )

    model = SudokuRandomForest(
        n_estimators=20,
        random_seed=42,
    )
    model.fit(data)

    model_path = tmp_path / "model.joblib"
    model.save(model_path)

    exit_code = main(
        [
            PUZZLE_TEXT,
            "--model",
            str(model_path),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Input" in output
    assert "Solution" in output
    assert (
        "Solver:              hybrid ML-guided (Random Forest)"
        in output
    )
    assert "5 3 4 | 6 7 8 | 9 1 2" in output
    assert "Deterministic steps: 51" in output
    assert "ML decisions:        0" in output
    assert "Branching decisions: 0" in output
    assert "Backtracks:          0" in output


def test_main_solves_puzzle_with_histogram_gradient_boosting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = create_train_test_split(
        num_solutions=20,
        random_seed=42,
    )

    model = SudokuHistogramGradientBoosting(
        max_iter=20,
        random_seed=42,
    )
    model.fit(data)

    model_path = tmp_path / "histogram_gradient_boosting.joblib"
    model.save(model_path)

    exit_code = main(
        [
            PUZZLE_TEXT,
            "--model-type",
            "histogram-gradient-boosting",
            "--model",
            str(model_path),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert (
        "Solver:              hybrid ML-guided "
        "(Histogram Gradient Boosting)"
        in output
    )
    assert "5 3 4 | 6 7 8 | 9 1 2" in output


def test_main_rejects_model_type_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = create_train_test_split(
        num_solutions=20,
        random_seed=42,
    )
    model = SudokuRandomForest(
        n_estimators=20,
        random_seed=42,
    )
    model.fit(data)

    model_path = tmp_path / "random_forest.joblib"
    model.save(model_path)

    with pytest.raises(SystemExit) as error:
        main(
            [
                PUZZLE_TEXT,
                "--model-type",
                "histogram-gradient-boosting",
                "--model",
                str(model_path),
            ]
        )

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "HistGradientBoostingClassifier" in captured.err


def test_main_rejects_unknown_model_type(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                PUZZLE_TEXT,
                "--model-type",
                "unknown",
            ]
        )

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "invalid choice" in captured.err

def test_main_rejects_invalid_puzzle(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["123"])

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "exactly 81 cells" in captured.err


def test_main_reports_missing_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_model = tmp_path / "missing.joblib"

    with pytest.raises(SystemExit) as error:
        main(
            [
                PUZZLE_TEXT,
                "--model",
                str(missing_model),
            ]
        )

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "missing.joblib" in captured.err


def test_main_uses_classical_solver_without_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([PUZZLE_TEXT, "--classical"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Solver:              classical" in output
    assert "5 3 4 | 6 7 8 | 9 1 2" in output


def test_main_reads_puzzle_from_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    puzzle_path = tmp_path / "puzzle.txt"
    puzzle_path.write_text(PUZZLE_TEXT, encoding="utf-8")

    exit_code = main(
        [
            "--input-file",
            str(puzzle_path),
            "--classical",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Solver:              classical" in output
    assert "Solution" in output


def test_main_reports_missing_input_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(SystemExit) as error:
        main(["--input-file", str(missing_path), "--classical"])

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "missing.txt" in captured.err


def test_main_requires_one_input_source(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main([])

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "required" in captured.err


def test_main_rejects_multiple_input_sources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    puzzle_path = tmp_path / "puzzle.txt"
    puzzle_path.write_text(PUZZLE_TEXT, encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(
            [
                PUZZLE_TEXT,
                "--input-file",
                str(puzzle_path),
                "--classical",
            ]
        )

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "not allowed with argument" in captured.err


def test_main_prints_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])

    output = capsys.readouterr().out

    assert error.value.code == 0
    assert "ml-sudoku-solver 0.1.0" in output
