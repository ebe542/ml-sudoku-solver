import torch

from sudoku_ml.training.device import get_device_info


def main() -> None:
    """Report the PyTorch and computation-device environment."""
    info = get_device_info()

    print("PyTorch Environment")
    print("-------------------")
    print(f"PyTorch version:  {torch.__version__}")
    print(f"CUDA build:       {torch.version.cuda}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    print(f"Selected device:  {info.device}")
    print(f"Device name:      {info.name}")

    if info.total_memory_bytes is not None:
        total_memory_gib = info.total_memory_bytes / 1024**3
        print(f"GPU memory:       {total_memory_gib:.2f} GiB")

    tensor = torch.ones((2, 2), device=info.device)
    result = tensor @ tensor
    print(f"Tensor device:    {result.device}")
    print(f"Tensor result:    {result.tolist()}")


if __name__ == "__main__":
    main()
