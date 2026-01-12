from copy import deepcopy

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.testing import assert_close, make_tensor
from torchaudio.models import HuBERTPretrainModel as ReferenceHuBERT
from torchaudio.models import hubert_pretrain_base

from minimal_hubert import HuBERTPretrainModel as MyHuBERT
from minimal_hubert.model import state_dict_from_torchaudio

# To replace by hypothesis
BATCH, LENGTH = 8, 16_000
CONV_LENGTH = 49
LOW, HIGH_MINUS_LOW = 0, 10


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def ref_hubert(device: torch.device) -> ReferenceHuBERT:
    return hubert_pretrain_base().eval().to(device)


@pytest.fixture(scope="session")
def num_classes(ref_hubert: ReferenceHuBERT) -> int:
    return ref_hubert.logit_generator.label_embeddings.size(0)


@pytest.fixture(scope="session")
def my_hubert(ref_hubert: ReferenceHuBERT, num_classes: int, device: torch.device) -> MyHuBERT:
    model = MyHuBERT(num_classes).eval().to(device)
    model.load_state_dict(state_dict_from_torchaudio(ref_hubert.state_dict()))
    return model


@pytest.fixture
def waveforms(device: torch.device) -> Tensor:
    return make_tensor((BATCH, LENGTH), dtype=torch.float32, low=LOW, high=HIGH_MINUS_LOW + LOW, device=device)


@torch.no_grad
def test_encoder_forward(my_hubert: MyHuBERT, ref_hubert: ReferenceHuBERT, waveforms: Tensor) -> None:
    x, _ = ref_hubert.wav2vec2(waveforms)
    y = my_hubert.feature_extractor(waveforms)
    y = my_hubert.feature_projection(y)
    y = my_hubert.encoder(y)
    assert_close(x, y)


@torch.no_grad
def test_encoder_intermediate(my_hubert: MyHuBERT, ref_hubert: ReferenceHuBERT, waveforms: Tensor) -> None:
    x, _ = ref_hubert.wav2vec2.extract_features(waveforms)
    y = my_hubert.get_intermediate_outputs(waveforms, before_residual=False)
    for xi, yi in zip(x, y, strict=True):
        assert_close(xi, yi)


class DummyMaskGenerator(nn.Module):
    def __init__(self, embedding: nn.Parameter) -> None:
        super().__init__()
        self.mask_embedding = embedding

    def get_mask(self) -> Tensor:
        torch.manual_seed(0)
        proba = 0.5
        return torch.randn((BATCH, CONV_LENGTH), dtype=torch.float32) > proba

    def forward(self, x: Tensor, _: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mask = self.get_mask().to(x.device)
        x[mask] = self.mask_embedding.to(x.dtype)
        return x, mask


def torchaudio_hubert_loss(
    logit_m: Tensor,
    logit_u: Tensor | None,
    feature_penalty: Tensor,
    masked_weight: float,
    unmasked_weight: float,
    feature_weight: float,
    reduction: str,
) -> Tensor:
    """From https://github.com/pytorch/audio/blob/main/examples/hubert/loss/hubert_loss.py"""
    loss = feature_penalty * feature_weight * logit_m.shape[0]
    target_m = torch.zeros(logit_m.shape[0], dtype=torch.long, device=logit_m.device)
    loss_m = F.cross_entropy(logit_m, target_m, reduction=reduction)
    loss += loss_m * masked_weight
    if logit_u is not None:
        target_u = torch.zeros(logit_u.shape[0], dtype=torch.long, device=logit_m.device)
        loss_u = F.cross_entropy(logit_u, target_u, reduction=reduction)
        loss += loss_u * unmasked_weight
    return loss


@torch.no_grad
def test_loss(my_hubert: MyHuBERT, ref_hubert: ReferenceHuBERT, waveforms: Tensor, num_classes: int) -> None:
    ref_hubert = deepcopy(ref_hubert)
    mask_generator = DummyMaskGenerator(ref_hubert.mask_generator.mask_embedding)
    ref_hubert.mask_generator = mask_generator

    labels = torch.randint(0, num_classes, (BATCH, CONV_LENGTH), device=waveforms.device)
    mask = mask_generator.get_mask().to(waveforms.device)
    x = torchaudio_hubert_loss(
        *ref_hubert(waveforms, labels),
        masked_weight=1.0,
        unmasked_weight=0.0,
        feature_weight=1.0,
        reduction="mean",
    )
    y = my_hubert(waveforms, labels, mask=mask)[0].mean()
    assert_close(x, y)
