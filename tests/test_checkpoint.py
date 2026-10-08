from pathlib import Path

import pytest
import torch

from minimal_hubert import HuBERT, HuBERTPretrain


def test_hubert_from_pretraining_checkpoint(tmp_path: Path) -> None:
    """A checkpoint saved during pretraining loads in the encoder-only model, dropping the pretraining head."""
    pretrain = HuBERTPretrain(100).eval()
    path = tmp_path / "step_1000.pt"
    torch.save({"model": pretrain.state_dict()}, path)
    model = HuBERT.from_pretrained(path)
    waveforms = torch.randn(1, 16_000)
    expected = pretrain.encoder(pretrain.feature_projection(pretrain.feature_extractor(waveforms)))
    torch.testing.assert_close(model(waveforms), expected)


def test_missing_local_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        HuBERT.from_pretrained(tmp_path / "missing.pt")
