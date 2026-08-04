from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sudoku_ml.dataset.end_to_end import (
    EndToEndDataset,
    EndToEndDataSplit,
)


class TorchSudokuDataset(Dataset):
    """Expose full-grid Sudoku samples as PyTorch tensors."""

    def __init__(self, dataset: EndToEndDataset) -> None:
        self.grids = torch.from_numpy(dataset.X).to(torch.long)
        self.targets = torch.from_numpy(dataset.y - 1).to(torch.long)
        self.empty_masks = torch.from_numpy(dataset.empty_mask).to(
            torch.bool
        )

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = self.grids[index]
        one_hot_grid = F.one_hot(
            grid,
            num_classes=10,
        ).permute(2, 0, 1).to(torch.float32)

        return (
            one_hot_grid,
            self.targets[index],
            self.empty_masks[index],
        )


@dataclass(frozen=True)
class TorchDataLoaders:
    """Store training and test DataLoaders."""

    train: DataLoader
    test: DataLoader


def create_torch_data_loaders(
    split: EndToEndDataSplit,
    batch_size: int = 32,
    random_seed: int | None = 42,
    num_workers: int = 0,
) -> TorchDataLoaders:
    """Create reproducible DataLoaders from an end-to-end split."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    if num_workers < 0:
        raise ValueError("Number of workers cannot be negative.")

    generator = torch.Generator()

    if random_seed is not None:
        generator.manual_seed(random_seed)

    return TorchDataLoaders(
        train=DataLoader(
            TorchSudokuDataset(split.train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        test=DataLoader(
            TorchSudokuDataset(split.test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )
