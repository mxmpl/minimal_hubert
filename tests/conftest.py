import pytest
import torch
from torch.testing import make_tensor


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def waveforms(device: torch.device) -> torch.Tensor:
    return make_tensor((8, 16_000), dtype=torch.float32, low=0, high=10, device=device)
