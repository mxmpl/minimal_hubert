import math
from pathlib import Path

import torch
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.errors import HFValidationError
from spidr.config import DEFAULT_CONV_LAYER_CONFIG
from spidr.models.components import (
    ConvLayerBlock,
    ConvPositionalEmbedding,
    FeatureExtractor,
    FeatureProjection,
    FeedForward,
    LayerNorm,
    SelfAttention,
    Transformer,
    TransformerLayer,
)
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .compatibility import (
    _LOGIT_TEMPERATURE,
    Size,
    load_hubert_fairseq_state_dict,
    load_state_dict_pre_hook_compatibility_frameworks,
    load_state_dict_pre_hook_strip_pt_specifics,
    size_from_state_dict,
)
from .config import HuBERTConfig


class LogitGenerator(nn.Module):
    def __init__(self, num_classes: int, size: Size = "base") -> None:
        super().__init__()
        cfg = HuBERTConfig.from_size(size)
        self.label_embeddings = nn.Parameter(torch.FloatTensor(num_classes, cfg.final_dim))
        nn.init.uniform_(self.label_embeddings)
        self.final_proj = nn.Linear(cfg.encoder_embed_dim, cfg.final_dim)
        self.logit_temp = nn.Buffer(torch.tensor(_LOGIT_TEMPERATURE))

    def forward(self, x: Tensor, label: Tensor) -> Tensor:
        x = self.final_proj(x)
        if (label < 0).any():
            raise ValueError("Negative labels found: slicing when wrong")
        pos = torch.index_select(self.label_embeddings, 0, label).unsqueeze(0)
        negs = self.label_embeddings.unsqueeze(1).expand(-1, x.size(0), -1)
        targets = torch.cat([pos, negs], dim=0)
        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x) / self.logit_temp
        neg_is_pos = (pos == negs).all(-1)
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        return logits.transpose(0, 1)


def hubert_components(cfg: HuBERTConfig) -> tuple[FeatureExtractor, nn.Sequential, Transformer]:
    blocks, in_channels = nn.ModuleList(), 1
    for i, (out_channels, kernel_size, stride) in enumerate(DEFAULT_CONV_LAYER_CONFIG):
        if cfg.extractor_mode == "layer_norm":
            norm = LayerNorm(normalized_shape=out_channels, elementwise_affine=True)
        elif cfg.extractor_mode == "group_norm" and i == 0:
            norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=True)
        else:
            norm = None
        blocks.append(ConvLayerBlock(in_channels, out_channels, kernel_size, stride, norm, bias=False))
        in_channels = out_channels
    feature_extractor = FeatureExtractor(blocks)
    pos_conv = ConvPositionalEmbedding(cfg.encoder_embed_dim, kernel_size=128, groups=16, depth=1)
    pos_conv.convs[0] = weight_norm(pos_conv.convs[0], dim=2)
    pos_conv.layer_norm = nn.Identity()  # ty: ignore[invalid-assignment]
    layers = nn.ModuleList()
    for _ in range(cfg.encoder_num_layers):
        attn = SelfAttention(
            cfg.encoder_embed_dim, cfg.encoder_num_heads, qkv_bias=True, dropout=cfg.encoder_attention_dropout
        )
        ff = FeedForward(cfg.encoder_embed_dim, cfg.encoder_ff_interm_features, cfg.encoder_ff_interm_dropout)
        layers.append(TransformerLayer(attn, cfg.encoder_dropout, ff, layer_norm_first=cfg.encoder_layer_norm_first))
    encoder = Transformer(
        layers,
        pos_conv,
        cfg.encoder_dropout,
        cfg.encoder_layer_drop,
        layer_norm_first=not cfg.encoder_layer_norm_first,
    )
    feature_projection = nn.Sequential(
        FeatureProjection(DEFAULT_CONV_LAYER_CONFIG[-1][0], cfg.encoder_embed_dim),
        nn.Dropout(cfg.encoder_projection_dropout),
    )
    return feature_extractor, feature_projection, encoder


class HuBERT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, size: Size = "base") -> None:
        super().__init__()
        self.config = HuBERTConfig.from_size(size)
        self.feature_extractor, self.feature_projection, self.encoder = hubert_components(self.config)
        self.init_weights_()
        self._hook_compat = self.register_load_state_dict_pre_hook(load_state_dict_pre_hook_compatibility_frameworks)
        self._hook_strip_pt = self.register_load_state_dict_pre_hook(load_state_dict_pre_hook_strip_pt_specifics)

    def init_weights_(self) -> None:
        module = self.encoder.pos_conv_embed
        std = math.sqrt(4.0 / (module.embed_dim * module.kernel_size))
        for conv in module.convs:
            nn.init.normal_(conv.weight, mean=0.0, std=std)  # ty: ignore[invalid-argument-type]
            nn.init.constant_(conv.bias, 0.0)  # ty: ignore[invalid-argument-type]

    def get_intermediate_outputs(
        self,
        waveforms: Tensor,
        *,
        attention_mask: Tensor | None = None,
        num_layers: int | None = None,
        before_residual: bool = True,
    ) -> list[Tensor]:
        x = self.feature_extractor(waveforms)
        x = self.feature_projection(x)
        return self.encoder.get_intermediate_outputs(x, attention_mask, num_layers, before_residual=before_residual)

    def forward(self, waveforms: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        x = self.feature_extractor(waveforms)
        x = self.feature_projection(x)
        return self.encoder(x, attention_mask)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **model_kwargs: str,
    ) -> "HuBERT":
        try:
            model_kwargs.pop("strict", None)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                force_download=force_download,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                strict=True,
                **model_kwargs,
            )
        except HFValidationError:
            state_dict = load_hubert_fairseq_state_dict(pretrained_model_name_or_path, for_pretraining=False)
            model = HuBERT(size=size_from_state_dict(state_dict)).eval()
            model.load_state_dict(state_dict)
            return model


class HuBERTPretrain(HuBERT):
    def __init__(self, num_classes: int, size: Size = "base") -> None:
        super().__init__(size)
        self._hook_strip_pt.remove()
        self.num_classes = num_classes
        self.logit_generator = LogitGenerator(num_classes, size=size)
        encoder_embed_dim = self.logit_generator.final_proj.in_features
        self.mask_embedding = nn.Parameter(torch.FloatTensor(encoder_embed_dim))
        nn.init.uniform_(self.mask_embedding)

    def forward(  # ty: ignore[invalid-method-override]
        self,
        waveforms: Tensor,
        labels: Tensor,
        *,
        mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        x = self.feature_extractor(waveforms)
        features_pen = x.float().pow(2).mean()
        x = self.feature_projection(x)
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_embedding.to(x.dtype).expand_as(x), x)
        else:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        x = self.encoder(x, attention_mask)
        mask_indices = torch.nonzero(mask, as_tuple=True)
        logits = self.logit_generator(x[mask_indices], labels[mask_indices])
        features_loss = features_pen * logits.shape[0]
        logits_loss = -F.log_softmax(logits, dim=1)[:, 0]
        return features_loss + logits_loss, {"feature_loss": features_loss, "logits_loss": logits_loss}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **model_kwargs: str,
    ) -> "HuBERTPretrain":
        try:
            model_kwargs.pop("strict", None)
            return super(HuBERT, cls).from_pretrained(  # Skip HuBERT.from_pretrained
                pretrained_model_name_or_path,
                force_download=force_download,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                strict=True,
                **model_kwargs,
            )
        except HFValidationError:
            state_dict = load_hubert_fairseq_state_dict(pretrained_model_name_or_path, for_pretraining=True)
            num_classes = state_dict["logit_generator.label_embeddings"].size(0)
            model = HuBERTPretrain(num_classes, size=size_from_state_dict(state_dict)).eval()
            model.load_state_dict(state_dict)
            return model
