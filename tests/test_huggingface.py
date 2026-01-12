import pytest
import torch
from transformers import HubertConfig, HubertModel

from minimal_hubert import HuBERT as MyHuBERT
from minimal_hubert.model import state_dict_from_torchaudio_or_huggingface


@pytest.fixture(scope="module")
def hf_hubert(device: torch.device) -> HubertModel:
    return HubertModel(HubertConfig()).eval().to(device)


@pytest.fixture(scope="module")
def my_hubert(hf_hubert: HubertModel, device: torch.device) -> MyHuBERT:
    model = MyHuBERT().eval().to(device)
    model.load_state_dict(state_dict_from_torchaudio_or_huggingface(hf_hubert.state_dict()))
    return model


@torch.no_grad
def test_hf_encoder_intermediate(my_hubert: MyHuBERT, hf_hubert: HubertModel, waveforms: torch.Tensor) -> None:
    x = hf_hubert(waveforms, output_hidden_states=True)
    y = my_hubert.get_intermediate_outputs(waveforms, before_residual=False)
    torch.testing.assert_close(x.last_hidden_state, y[-1])
    for xi, yi in zip(x.hidden_states[1:], y, strict=True):
        torch.testing.assert_close(xi, yi)
