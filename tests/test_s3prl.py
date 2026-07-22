from pathlib import Path
from typing import NotRequired, TypedDict

import pytest
import torch
from hypothesis import given
from s3prl.upstream.hubert.convert import load_converted_model
from s3prl.upstream.hubert.hubert_model import HubertConfig, HubertModel, HubertPretrainingConfig

from minimal_hubert import HuBERT, HuBERTPretrain

from .conftest import hypothesis_settings, waveforms

S3PRL_CKPT_URL = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_base_ls960.pt"


class Tolerance(TypedDict):
    rtol: NotRequired[float]
    atol: NotRequired[float]


def tolerance(device: torch.device) -> Tolerance:
    return {"rtol": 1e-4, "atol": 1e-4} if device.type == "cpu" else {}


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
        torch.testing.assert_close(xi, yi, **tolerance(device))


@pytest.fixture(scope="module")
def s3prl_ckpt_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ckpts") / "hubert_base_ls960.pt"
    torch.hub.download_url_to_file(S3PRL_CKPT_URL, path.as_posix())
    return path


@pytest.fixture(scope="module")
def s3prl_pretrained_hubert(s3prl_ckpt_path: Path, device: torch.device) -> HubertModel:
    model, _ = load_converted_model(str(s3prl_ckpt_path))
    return model.eval().to(device)


@pytest.fixture(scope="module")
def my_pretrained_hubert(s3prl_ckpt_path: Path, device: torch.device) -> HuBERT:
    return HuBERT.from_pretrained(s3prl_ckpt_path).eval().to(device)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@given(waveforms=waveforms())
@hypothesis_settings
def test_s3prl_pretrained_encoder_intermediate(
    my_pretrained_hubert: HuBERT,
    s3prl_pretrained_hubert: HubertModel,
    device: torch.device,
    waveforms: torch.Tensor,
) -> None:
    waveforms = waveforms.to(device)
    y = my_pretrained_hubert.get_intermediate_outputs(waveforms, before_residual=False)
    for layer, yi in enumerate(y):
        xi = s3prl_pretrained_hubert.extract_features(waveforms, output_layer=layer + 1)[0]
        torch.testing.assert_close(xi, yi, **tolerance(device))
