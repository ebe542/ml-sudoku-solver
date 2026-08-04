import torch

from sudoku_ml.training.device import (
    get_device_info,
    select_device,
)


def test_device_can_be_forced_to_cpu() -> None:
    device = select_device(prefer_cuda=False)

    assert device.type == "cpu"


def test_automatic_device_matches_cuda_availability() -> None:
    device = select_device()
    expected_type = "cuda" if torch.cuda.is_available() else "cpu"

    assert device.type == expected_type


def test_device_info_describes_selected_device() -> None:
    info = get_device_info()

    assert info.device.type in {"cpu", "cuda"}
    assert info.name
    assert info.cuda_available == torch.cuda.is_available()

    if info.device.type == "cuda":
        assert info.cuda_version is not None
        assert info.total_memory_bytes is not None
        assert info.total_memory_bytes > 0
    else:
        assert info.total_memory_bytes is None
