import warnings

import pytest
import torch
from hypothesis import given, settings

from minimal_hubert import HuBERT, known_huberts

from .conftest import waveforms

LS960_URLS = [url for url in known_huberts()["base"] if "ls960" in url]


@pytest.fixture(scope="session")
def ls960_models(device: torch.device) -> list[HuBERT]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="torch.load used with 'weights_only=False'")
        return [HuBERT.from_pretrained(url).eval().to(device) for url in LS960_URLS]


@given(waveforms=waveforms())
@settings(max_examples=10, deadline=None)
def test_ls960_consistency(ls960_models: list[HuBERT], device: torch.device, waveforms: torch.Tensor) -> None:
    waveforms = waveforms.to(device)
    reference = ls960_models[0](waveforms)
    for model in ls960_models[1:]:
        torch.testing.assert_close(model(waveforms), reference)
