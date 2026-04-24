import pytest
import torch
from hypothesis import given
from s3prl.upstream.hubert.hubert_model import HubertConfig, HubertModel, HubertPretrainingConfig

from minimal_hubert import HuBERTPretrain

from .conftest import hypothesis_settings, waveforms


@pytest.fixture(scope="module")
def num_classes() -> int:
    return 100


@pytest.fixture(scope="module")
def s3prl_hubert(num_classes: int, device: torch.device) -> HubertModel:
    cfg = HubertConfig(label_rate=50, final_dim=256)
    return HubertModel(cfg, HubertPretrainingConfig(), dictionaries=[range(num_classes)]).eval().to(device)


@pytest.fixture(scope="module")
def my_hubert(s3prl_hubert: HubertModel, num_classes: int, device: torch.device) -> HuBERTPretrain:
    model = HuBERTPretrain(num_classes).eval().to(device)
    model.load_state_dict(s3prl_hubert.state_dict())
    return model


@pytest.mark.filterwarnings("ignore::FutureWarning")
@given(waveforms=waveforms())
@hypothesis_settings
def test_s3prl_encoder_intermediate(
    my_hubert: HuBERTPretrain,
    s3prl_hubert: HubertModel,
    device: torch.device,
    waveforms: torch.Tensor,
) -> None:
    waveforms = waveforms.to(device)
    y = my_hubert.get_intermediate_outputs(waveforms, before_residual=False)
    for layer, yi in enumerate(y):
        xi = s3prl_hubert.extract_features(waveforms, output_layer=layer + 1)[0]
        torch.testing.assert_close(xi, yi)
