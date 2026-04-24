import numpy as np
import pytest
import torch
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays
from hypothesis.strategies import DrawFn

hypothesis_settings = settings(max_examples=20, deadline=None)


@st.composite
def waveforms(draw: DrawFn) -> torch.Tensor:
    batch_size = draw(st.integers(min_value=1, max_value=4))
    length = draw(st.integers(min_value=1_000, max_value=16_000))
    elements = st.floats(-10, 10, allow_nan=False, allow_infinity=False, allow_subnormal=False)
    data = draw(np_arrays(dtype=np.float32, shape=(batch_size, length), elements=elements))
    return torch.from_numpy(data)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
