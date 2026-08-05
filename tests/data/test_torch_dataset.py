import numpy as np
import pytest
import torch

from sudoku_ml.data.torch_dataset import (
    TorchSudokuDataset,
    create_torch_data_loaders,
)
from sudoku_ml.dataset.end_to_end import (
    create_end_to_end_train_test_split,
)


@pytest.fixture(scope="module")
def end_to_end_split():
    return create_end_to_end_train_test_split(
        num_solutions=5,
        test_size=0.4,
        removal_rates=(0.10, 0.20),
        random_seed=42,
    )


def test_torch_dataset_returns_expected_tensors(
    end_to_end_split,
) -> None:
    dataset = TorchSudokuDataset(end_to_end_split.train)
    inputs, targets, empty_mask = dataset[0]

    assert inputs.shape == (10, 9, 9)
    assert targets.shape == (9, 9)
    assert empty_mask.shape == (9, 9)
    assert inputs.dtype == torch.float32
    assert targets.dtype == torch.int64
    assert empty_mask.dtype == torch.bool


def test_one_hot_inputs_encode_exactly_one_value_per_cell(
    end_to_end_split,
) -> None:
    dataset = TorchSudokuDataset(end_to_end_split.train)
    inputs, _, _ = dataset[0]

    assert torch.all(inputs.sum(dim=0) == 1.0)
    decoded_grid = torch.argmax(inputs, dim=0).numpy()
    assert np.array_equal(decoded_grid, end_to_end_split.train.X[0])


def test_targets_are_shifted_to_zero_based_classes(
    end_to_end_split,
) -> None:
    dataset = TorchSudokuDataset(end_to_end_split.train)
    _, targets, _ = dataset[0]

    assert int(targets.min()) >= 0
    assert int(targets.max()) <= 8
    assert np.array_equal(
        targets.numpy() + 1,
        end_to_end_split.train.y[0],
    )


def test_empty_mask_matches_zero_input_cells(
    end_to_end_split,
) -> None:
    dataset = TorchSudokuDataset(end_to_end_split.train)
    inputs, _, empty_mask = dataset[0]
    decoded_grid = torch.argmax(inputs, dim=0)

    assert torch.equal(empty_mask, decoded_grid == 0)


def test_data_loaders_return_batched_tensors(
    end_to_end_split,
) -> None:
    loaders = create_torch_data_loaders(
        end_to_end_split,
        batch_size=3,
        random_seed=42,
    )
    inputs, targets, empty_masks = next(iter(loaders.train))

    assert inputs.shape == (3, 10, 9, 9)
    assert targets.shape == (3, 9, 9)
    assert empty_masks.shape == (3, 9, 9)
    assert len(loaders.train.dataset) == len(end_to_end_split.train.X)
    assert len(loaders.test.dataset) == len(end_to_end_split.test.X)


@pytest.mark.parametrize("batch_size", (0, -1))
def test_data_loaders_reject_invalid_batch_size(
    end_to_end_split,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        create_torch_data_loaders(
            end_to_end_split,
            batch_size=batch_size,
        )


def test_data_loaders_reject_negative_workers(end_to_end_split) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        create_torch_data_loaders(
            end_to_end_split,
            num_workers=-1,
        )
