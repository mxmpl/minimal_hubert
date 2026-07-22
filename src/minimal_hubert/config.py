import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spidr.config import DataConfig, MaskingConfig, OptimizerConfig, RunConfig

from .compatibility import Size


@dataclass(frozen=True)
class HuBERTConfig:
    size: Size
    extractor_mode: Literal["layer_norm", "group_norm"]
    encoder_embed_dim: int
    encoder_num_layers: int
    encoder_num_heads: int
    encoder_ff_interm_features: int
    encoder_layer_norm_first: bool
    final_dim: int
    encoder_projection_dropout: float
    encoder_attention_dropout: float
    encoder_ff_interm_dropout: float
    encoder_dropout: float
    encoder_layer_drop: float

    @classmethod
    def from_size(cls, size: Size) -> "HuBERTConfig":
        match size:
            case "base":
                return cls(
                    size="base",
                    extractor_mode="group_norm",
                    encoder_embed_dim=768,
                    encoder_num_layers=12,
                    encoder_num_heads=12,
                    encoder_ff_interm_features=3_072,
                    encoder_layer_norm_first=False,
                    final_dim=256,
                    encoder_projection_dropout=0.1,
                    encoder_attention_dropout=0.1,
                    encoder_ff_interm_dropout=0.0,
                    encoder_dropout=0.1,
                    encoder_layer_drop=0.05,
                )
            case "large":
                return cls(
                    size="large",
                    extractor_mode="layer_norm",
                    encoder_embed_dim=1_024,
                    encoder_num_layers=24,
                    encoder_num_heads=16,
                    encoder_ff_interm_features=4_096,
                    encoder_layer_norm_first=True,
                    final_dim=768,
                    encoder_projection_dropout=0.0,
                    encoder_attention_dropout=0.0,
                    encoder_ff_interm_dropout=0.0,
                    encoder_dropout=0.0,
                    encoder_layer_drop=0.0,
                )
            case "xlarge":
                return cls(
                    size="xlarge",
                    extractor_mode="layer_norm",
                    encoder_embed_dim=1_280,
                    encoder_num_layers=48,
                    encoder_num_heads=16,
                    encoder_ff_interm_features=5_120,
                    encoder_layer_norm_first=True,
                    final_dim=1_024,
                    encoder_projection_dropout=0.0,
                    encoder_attention_dropout=0.0,
                    encoder_ff_interm_dropout=0.0,
                    encoder_dropout=0.0,
                    encoder_layer_drop=0.0,
                )
        raise ValueError(f"Invalid size {size}. Must be either 'base', 'large', or 'xlarge'")


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
        pin_memory=True,
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
