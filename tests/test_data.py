import pytest
import torch
from spidr.config import DEFAULT_CONV_LAYER_CONFIG, MaskingConfig
from spidr.data.dataset import conv_length
from spidr.data.masks import MaskGenerator
from torch import Tensor

from minimal_hubert import HuBERTPretrain
from minimal_hubert.data import SpeechWithLabelsCollatorWithMasking

NUM_CLASSES = 100


def collator(*, enable_padding: bool) -> SpeechWithLabelsCollatorWithMasking:
    return SpeechWithLabelsCollatorWithMasking(
        MaskGenerator(MaskingConfig()),
        max_sample_size=250_000,
        conv_layer_config=DEFAULT_CONV_LAYER_CONFIG,
        enable_padding=enable_padding,
        rand_crop=False,
    )


def sample(num_samples: int) -> tuple[Tensor, Tensor]:
    num_labels = int(conv_length(DEFAULT_CONV_LAYER_CONFIG, torch.tensor(num_samples)))
    return torch.randn(num_samples), torch.randint(NUM_CLASSES, size=(num_labels,))


def test_collator_without_padding_has_no_attention_mask() -> None:
    """Batches are cropped to the shortest waveform, so there is nothing to mask."""
    batch = [sample(16_000), sample(20_000), sample(24_000)]
    _, _, attn_mask, _ = collator(enable_padding=False)(batch)
    assert attn_mask is None


def test_collator_with_padding_has_broadcastable_attention_mask() -> None:
    batch = [sample(16_000), sample(20_000), sample(24_000)]
    wavs, _, attn_mask, _ = collator(enable_padding=True)(batch)
    assert attn_mask is not None
    max_len = int(conv_length(DEFAULT_CONV_LAYER_CONFIG, torch.tensor(wavs.size(1))))
    assert attn_mask.shape == (len(batch), 1, 1, max_len)


def test_collator_padded_mask_marks_the_valid_frames() -> None:
    """The mask is True on the frames each waveform actually has, and False on the padding."""
    num_samples = [16_000, 20_000, 24_000]
    _, _, attn_mask, _ = collator(enable_padding=True)([sample(n) for n in num_samples])
    assert attn_mask is not None
    lengths = conv_length(DEFAULT_CONV_LAYER_CONFIG, torch.tensor(num_samples))
    max_len = attn_mask.size(-1)
    expected = torch.arange(max_len).expand(len(num_samples), max_len) < lengths[:, None]
    assert torch.equal(attn_mask[:, 0, 0], expected)


@pytest.mark.parametrize("enable_padding", [True, False])
def test_collator_labels_align_with_frames(*, enable_padding: bool) -> None:
    batch = [sample(16_000), sample(20_000), sample(24_000)]
    wavs, labels, _, mask = collator(enable_padding=enable_padding)(batch)
    max_len = int(conv_length(DEFAULT_CONV_LAYER_CONFIG, torch.tensor(wavs.size(1))))
    assert labels.shape == (len(batch), max_len)
    assert mask.shape == (len(batch), max_len)


def test_masked_frame_count_matches_loss_size(device: torch.device) -> None:
    """The training loop normalizes by mask.sum() instead of loss.size(0): they must agree."""
    batch = [sample(16_000), sample(20_000), sample(24_000)]
    wavs, labels, attn_mask, mask = collator(enable_padding=False)(batch)
    model = HuBERTPretrain(NUM_CLASSES).eval().to(device)
    with torch.no_grad():
        loss, _ = model(wavs.to(device), labels.to(device), mask=mask.to(device), attention_mask=attn_mask)
    assert loss.size(0) == int(mask.sum())


def test_model_broadcast_mask_matches_expanded_mask(device: torch.device) -> None:
    batch = [sample(16_000), sample(20_000), sample(24_000)]
    wavs, labels, attn_mask, mask = collator(enable_padding=True)(batch)
    assert attn_mask is not None
    batch_size, _, _, max_len = attn_mask.shape
    model = HuBERTPretrain(NUM_CLASSES).eval().to(device)
    wavs, labels, mask, attn_mask = wavs.to(device), labels.to(device), mask.to(device), attn_mask.to(device)
    with torch.no_grad():
        broadcast, _ = model(wavs, labels, mask=mask, attention_mask=attn_mask)
        expanded, _ = model(wavs, labels, mask=mask, attention_mask=attn_mask.expand(batch_size, 1, max_len, max_len))
    torch.testing.assert_close(broadcast, expanded)
