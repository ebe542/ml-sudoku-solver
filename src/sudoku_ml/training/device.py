from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Describe the selected PyTorch computation device."""

    device: torch.device
    name: str
    cuda_available: bool
    cuda_version: str | None
    total_memory_bytes: int | None


def select_device(prefer_cuda: bool = True) -> torch.device:
    """Select CUDA when requested and available, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_device_info(prefer_cuda: bool = True) -> DeviceInfo:
    """Return information about the selected computation device."""
    device = select_device(prefer_cuda=prefer_cuda)

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)

        return DeviceInfo(
            device=device,
            name=properties.name,
            cuda_available=True,
            cuda_version=torch.version.cuda,
            total_memory_bytes=properties.total_memory,
        )

    return DeviceInfo(
        device=device,
        name="CPU",
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        total_memory_bytes=None,
    )
