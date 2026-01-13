from copy import deepcopy

import pytest
import torch
from spidr.data.dataset import DEFAULT_CONV_LAYER_CONFIG, conv_length
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.models import HuBERTPretrainModel as ReferenceHuBERT
from torchaudio.models import hubert_pretrain_base

from minimal_hubert import HuBERTPretrain as MyHuBERT
from minimal_hubert.model import state_dict_from_torchaudio_or_huggingface


@pytest.fixture(scope="module")
def torchaudio_hubert(device: torch.device) -> ReferenceHuBERT:
    return hubert_pretrain_base().eval().to(device)


@pytest.fixture(scope="module")
def num_classes(torchaudio_hubert: ReferenceHuBERT) -> int:
    return torchaudio_hubert.logit_generator.label_embeddings.size(0)


@pytest.fixture(scope="module")
def my_hubert(torchaudio_hubert: ReferenceHuBERT, num_classes: int, device: torch.device) -> MyHuBERT:
    model = MyHuBERT(num_classes).eval().to(device)
    model.load_state_dict(state_dict_from_torchaudio_or_huggingface(torchaudio_hubert.state_dict()))
    return model


@torch.no_grad
def test_torchaudio_encoder_forward(
    my_hubert: MyHuBERT,
    torchaudio_hubert: ReferenceHuBERT,
    waveforms: Tensor,
) -> None:
    x, _ = torchaudio_hubert.wav2vec2(waveforms)
    y = my_hubert.feature_extractor(waveforms)
    y = my_hubert.feature_projection(y)
    y = my_hubert.encoder(y)
    torch.testing.assert_close(x, y)


@torch.no_grad
def test_torchaudio_encoder_intermediate(
    my_hubert: MyHuBERT,
    torchaudio_hubert: ReferenceHuBERT,
    waveforms: Tensor,
) -> None:
    x, _ = torchaudio_hubert.wav2vec2.extract_features(waveforms)
    y = my_hubert.get_intermediate_outputs(waveforms, before_residual=False)
    for xi, yi in zip(x, y, strict=True):
        torch.testing.assert_close(xi, yi)


class DummyMaskGenerator(nn.Module):
    def __init__(self, embedding: nn.Parameter) -> None:
        super().__init__()
        self.mask_embedding = embedding

    def get_mask(self, shape: tuple[int, ...]) -> Tensor:
        torch.manual_seed(0)
        proba = 0.5
        return torch.randn(shape, dtype=torch.float32) > proba

    def forward(self, x: Tensor, _: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mask = self.get_mask((x.size(0), x.size(1))).to(x.device)
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
def test_torchaudio_loss(
    my_hubert: MyHuBERT,
    torchaudio_hubert: ReferenceHuBERT,
    waveforms: Tensor,
    num_classes: int,
) -> None:
    torchaudio_hubert = deepcopy(torchaudio_hubert)
    mask_generator = DummyMaskGenerator(torchaudio_hubert.mask_generator.mask_embedding)
    torchaudio_hubert.mask_generator = mask_generator

    shape = (waveforms.size(0), int(conv_length(DEFAULT_CONV_LAYER_CONFIG, waveforms.size(1))))
    labels = torch.randint(0, num_classes, shape, device=waveforms.device)
    mask = mask_generator.get_mask(shape).to(waveforms.device)
    x = torchaudio_hubert_loss(
        *torchaudio_hubert(waveforms, labels),
        masked_weight=1.0,
        unmasked_weight=0.0,
        feature_weight=1.0,
        reduction="mean",
    )
    y = my_hubert(waveforms, labels, mask=mask, attention_mask=None)[0].mean()
    torch.testing.assert_close(x, y)
