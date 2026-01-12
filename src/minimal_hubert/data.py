import math
from pathlib import Path

import torch
from spidr.config import DEFAULT_CONV_LAYER_CONFIG, DataConfig, MaskingConfig
from spidr.data.dataset import (
    BucketizeBatchSampler,
    DistributedBatchSampler,
    SpeechCollatorWithMasking,
    SpeechDataset,
    SpeechDatasetFromArchive,
    SpeechDatasetFromFiles,
    conv_length,
)
from spidr.data.masks import MaskGenerator
from torch import Tensor
from torch import distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class SpeechDatasetWithLabelsFromArchive(SpeechDatasetFromArchive):
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        waveform = super().__getitem__(index)
        labels = self.manifest[index, "units"]
        return waveform, torch.tensor(labels)


class SpeechDatasetWithLabelsFromFiles(SpeechDatasetFromFiles):
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        waveform = super().__getitem__(index)
        labels = self.manifest[index, "units"]
        return waveform, torch.tensor(labels)


def speech_dataset_with_labels(manifest_path: Path | str, *, normalize: bool) -> SpeechDataset:
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        columns = set(f.readline().strip().split(","))
    if {"fileid", "path", "num_samples", "archive", "byte_offset", "byte_size"}.issubset(columns):
        return SpeechDatasetWithLabelsFromArchive(manifest_path, normalize=normalize)
    return SpeechDatasetWithLabelsFromFiles(manifest_path, normalize=normalize)


def crop_audio_and_labels(
    waveform: Tensor,
    label: Tensor,
    num_samples: int,
    max_sample_size: int,
    *,
    rand_crop: bool,
) -> tuple[Tensor, Tensor, int]:
    frame_offset = 0
    length = waveform.size(0)
    num_samples = min(num_samples, max_sample_size)
    if length > num_samples and rand_crop:
        frame_offset = int(torch.randint(length - num_samples, size=(1,)))
    elif length < num_samples:
        num_samples = length
    # label_offset = max(conv_length(DEFAULT_CONV_LAYER_CONFIG, frame_offset), 0)
    # num_label = conv_length(DEFAULT_CONV_LAYER_CONFIG, num_samples)
    kernel_size, stride, sample_rate = 25, 20, 16
    label_offset = max(math.floor((frame_offset - kernel_size * sample_rate) / (stride * sample_rate)) + 1, 0)
    num_label = math.floor((num_samples - kernel_size * sample_rate) / (stride * sample_rate)) + 1
    return (
        waveform[frame_offset : frame_offset + num_samples],
        label[label_offset : label_offset + num_label],
        num_samples,
    )


class SpeechWithLabelsCollatorWithMasking(SpeechCollatorWithMasking):
    def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        num_samples = max(len(wav) for wav, _ in batch) if self.enable_padding else min(len(wav) for wav, _ in batch)
        wavs_labels_len = [
            crop_audio_and_labels(wav, label, num_samples, self.max_sample_size, rand_crop=self.rand_crop)
            for wav, label in batch
        ]
        wav_list, label_list, wav_lengths = zip(*wavs_labels_len, strict=True)
        wavs = pad_sequence(wav_list, batch_first=True)
        labels = pad_sequence(label_list, batch_first=True, padding_value=-1)
        lengths = conv_length(self.conv_layer_config, torch.tensor(wav_lengths))
        batch_size, max_len = wavs.size(0), int(lengths.max())
        padding_mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
        attn_mask = ~padding_mask[:, None, None, :].expand(batch_size, 1, max_len, max_len)
        mask_indices = self.mask_generator(padding_mask)[0]
        return wavs, labels, attn_mask, mask_indices


def build_dataloader_with_labels(data_cfg: DataConfig, mask_cfg: MaskingConfig) -> DataLoader:
    dataset = speech_dataset_with_labels(data_cfg.manifest, normalize=data_cfg.normalize)
    batch_sampler = DistributedBatchSampler(
        BucketizeBatchSampler(
            lengths=dataset.manifest["num_samples"].to_list(),
            num_buckets=data_cfg.num_buckets,
            min_len=data_cfg.min_sample_size,
            max_len=data_cfg.max_sample_size,
            max_token_count=data_cfg.max_batch_length,
            batch_size=None,
            seed=data_cfg.random_seed,
            bucket_method=data_cfg.bucket_method,
            shuffle=True,
            drop_last=data_cfg.drop_last,
        ),
        seed=data_cfg.random_seed,
        shuffle=True,
        drop_last=data_cfg.drop_last,
    )
    collate_fn = SpeechWithLabelsCollatorWithMasking(
        MaskGenerator(mask_cfg),
        max_sample_size=data_cfg.max_sample_size,
        conv_layer_config=DEFAULT_CONV_LAYER_CONFIG,
        enable_padding=data_cfg.enable_padding,
        rand_crop=data_cfg.rand_crop,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=data_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=data_cfg.prefetch_factor,
        persistent_workers=data_cfg.persistent_workers,
        generator=torch.Generator().manual_seed(
            data_cfg.random_seed + (dist.get_rank() if dist.is_initialized() else 0)
        ),
    )
