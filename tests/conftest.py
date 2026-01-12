import pytest
import torch
from torch.testing import make_tensor

# To replace by hypothesis
BATCH, LENGTH = 8, 16_000
CONV_LENGTH = 49
LOW, HIGH_MINUS_LOW = 0, 10


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def waveforms(device: torch.device) -> torch.Tensor:
    return make_tensor((BATCH, LENGTH), dtype=torch.float32, low=LOW, high=HIGH_MINUS_LOW + LOW, device=device)
