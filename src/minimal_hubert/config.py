import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spidr.config import DataConfig, MaskingConfig, OptimizerConfig, RunConfig


@dataclass(frozen=True)
class UserConfig:
    num_classes: int
    manifest: str
    workdir: str
    wandb_project: str
    wandb_name: str
    wandb_mode: Literal["online", "offline"]


def hubert_optim_config() -> OptimizerConfig:
    return OptimizerConfig(
        betas=(0.9, 0.98),
        init_lr_scale=1e-8,
        final_lr_scale=1e-8,
        warmup_steps=32_000,
        hold_steps=0,
        decay_steps=368_000,
        to_freeze=[],
    )


def hubert_data_config(manifest: str) -> DataConfig:
    return DataConfig(
        manifest,
        normalize=True,
        min_sample_size=32_000,
        max_sample_size=250_000,
        max_batch_length=2_800_000,
        num_buckets=100,
        num_workers=24,
        prefetch_factor=4,
        bucket_method="percentile",
    )


@dataclass(frozen=True)
class Config:
    """Full configuration."""

    num_classes: int
    run: RunConfig
    data: DataConfig
    optimizer: OptimizerConfig
    masking: MaskingConfig


def read_config(path: str | Path) -> Config:
    user = UserConfig(**tomllib.loads(Path(path).read_text(encoding="utf-8")))
    return Config(
        num_classes=user.num_classes,
        run=RunConfig(user.workdir, user.wandb_project, user.wandb_name, user.wandb_mode),
        data=hubert_data_config(user.manifest),
        optimizer=hubert_optim_config(),
        masking=MaskingConfig(),
    )
