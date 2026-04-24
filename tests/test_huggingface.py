import pytest
import torch
from hypothesis import given
from transformers import HubertConfig, HubertModel

from minimal_hubert import HuBERT

from .conftest import hypothesis_settings, waveforms


@pytest.fixture(scope="module")
def hf_hubert(device: torch.device) -> HubertModel:
    return HubertModel(HubertConfig()).eval().to(device)


@pytest.fixture(scope="module")
def my_hubert(hf_hubert: HubertModel, device: torch.device) -> HuBERT:
    model = HuBERT().eval().to(device)
    model.load_state_dict(hf_hubert.state_dict())
    return model


@given(waveforms=waveforms())
@hypothesis_settings
def test_hf_encoder_intermediate(
    my_hubert: HuBERT, hf_hubert: HubertModel, device: torch.device, waveforms: torch.Tensor
) -> None:
    waveforms = waveforms.to(device)
    x = hf_hubert(waveforms, output_hidden_states=True)
    y = my_hubert.get_intermediate_outputs(waveforms, before_residual=False)
    torch.testing.assert_close(x.last_hidden_state, y[-1])
    for xi, yi in zip(x.hidden_states[1:], y, strict=True):
        torch.testing.assert_close(xi, yi)
