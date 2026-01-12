import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from spidr.config import DEFAULT_CONV_LAYER_CONFIG
from spidr.models.components import (
    ConvLayerBlock,
    ConvPositionalEmbedding,
    FeatureExtractor,
    FeatureProjection,
    FeedForward,
    SelfAttention,
    Transformer,
    TransformerLayer,
)
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

LOGIT_TEMPERATURE = 0.1
FEATURE_WEIGHT = 1.0  # Set to 1.0 instead of 10.0 to ~compensate for feature_grad_mult=0.1 in the original code


class LogitGenerator(nn.Module):
    def __init__(self, encoder_embed_dim: int, num_classes: int, final_dim: int) -> None:
        super().__init__()
        self.label_embeddings = nn.Parameter(torch.FloatTensor(num_classes, final_dim))
        nn.init.uniform_(self.label_embeddings)
        self.final_proj = nn.Linear(encoder_embed_dim, final_dim)
        self.logit_temp = nn.Buffer(torch.tensor(LOGIT_TEMPERATURE))

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


def hubert_base_components(
    encoder_projection_dropout: float,
    encoder_attention_dropout: float,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_drop: float,
) -> tuple[FeatureExtractor, nn.Sequential, Transformer, nn.Parameter]:
    blocks, in_channels = nn.ModuleList(), 1
    for i, (out_channels, kernel_size, stride) in enumerate(DEFAULT_CONV_LAYER_CONFIG):
        norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=True) if i == 0 else None
        blocks.append(ConvLayerBlock(in_channels, out_channels, kernel_size, stride, norm, bias=False))
        in_channels = out_channels
    feature_extractor = FeatureExtractor(blocks)
    pos_conv = ConvPositionalEmbedding(768, 128, 16, 1)
    pos_conv.convs[0] = weight_norm(pos_conv.convs[0], dim=2)
    pos_conv.layer_norm = nn.Identity()
    layers = nn.ModuleList()
    for _ in range(12):
        attention = SelfAttention(768, 12, qkv_bias=True, dropout=encoder_attention_dropout)
        feed_forward = FeedForward(768, 3_072, encoder_ff_interm_dropout)
        layers.append(TransformerLayer(attention, encoder_dropout, feed_forward, layer_norm_first=False))
    encoder = Transformer(layers, pos_conv, encoder_dropout, encoder_layer_drop, layer_norm_first=True)
    feature_projection = nn.Sequential(
        FeatureProjection(DEFAULT_CONV_LAYER_CONFIG[-1][0], 768),
        nn.Dropout(encoder_projection_dropout),
    )
    mask_embedding = nn.Parameter(torch.FloatTensor(768))
    return feature_extractor, feature_projection, encoder, mask_embedding


class HuBERT(nn.Module):
    def __init__(
        self,
        *,
        encoder_projection_dropout: float = 0.1,
        encoder_attention_dropout: float = 0.1,
        encoder_ff_interm_dropout: float = 0.0,
        encoder_dropout: float = 0.1,
        encoder_layer_drop: float = 0.05,
    ) -> None:
        super().__init__()
        self.feature_extractor, self.feature_projection, self.encoder, self.mask_embedding = hubert_base_components(
            encoder_projection_dropout,
            encoder_attention_dropout,
            encoder_ff_interm_dropout,
            encoder_dropout,
            encoder_layer_drop,
        )
        self.init_weights_()

    def init_weights_(self) -> None:
        nn.init.uniform_(self.mask_embedding)
        module = self.encoder.pos_conv_embed
        std = math.sqrt(4.0 / (module.embed_dim * module.kernel_size))
        for conv in module.convs:
            nn.init.normal_(conv.weight, mean=0.0, std=std)
            nn.init.constant_(conv.bias, 0.0)

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

    def forward(
        self, waveforms: Tensor, *, mask: Tensor | None = None, attention_mask: Tensor | None = None
    ) -> Tensor:
        x = self.feature_extractor(waveforms)
        x = self.feature_projection(x)
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_embedding.to(x.dtype).expand_as(x), x)
        else:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        return self.encoder(x, attention_mask)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "HuBERT":
        state_dict = torch.load(path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model = cls()
        model.load_state_dict(state_dict)
        return model.eval()


class HuBERTPretrain(HuBERT):
    def __init__(
        self,
        num_classes: int,
        *,
        encoder_projection_dropout: float = 0.1,
        encoder_attention_dropout: float = 0.1,
        encoder_ff_interm_dropout: float = 0.0,
        encoder_dropout: float = 0.1,
        encoder_layer_drop: float = 0.05,
    ) -> None:
        super().__init__(
            encoder_projection_dropout=encoder_projection_dropout,
            encoder_attention_dropout=encoder_attention_dropout,
            encoder_ff_interm_dropout=encoder_ff_interm_dropout,
            encoder_dropout=encoder_dropout,
            encoder_layer_drop=encoder_layer_drop,
        )
        self.feature_weight = nn.Buffer(torch.tensor(FEATURE_WEIGHT))
        self.logit_generator = LogitGenerator(768, num_classes, 256)

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
        features_loss = features_pen * self.feature_weight * logits.shape[0]
        logits_loss = -F.log_softmax(logits, dim=1)[:, 0]
        return features_loss + logits_loss, {"feature_loss": features_loss, "logits_loss": logits_loss}

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "HuBERTPretrain":
        state_dict = torch.load(path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        num_classes = state_dict["logit_generator.label_embeddings"].size(0)
        model = cls(num_classes)
        model.load_state_dict(state_dict)
        return model.eval()


def _fix_state_dict_key(key: str) -> str:
    if key == "masked_spec_embed":
        return "mask_embedding"
    key = re.sub(r"^wav2vec2\.", "", key)
    key = re.sub(r"^mask_generator\.", "", key)
    key = re.sub(r"^encoder\.transformer\.", "encoder.", key)
    key = re.sub(r"^feature_projection\.", "feature_projection.0.", key)
    key = re.sub(r"^encoder\.feature_projection\.", "feature_projection.0.", key)
    key = re.sub(r"\.out_proj\.", ".proj.", key)
    return re.sub(r"^encoder\.pos_conv_embed\.conv\.", "encoder.pos_conv_embed.convs.0.", key)


def state_dict_from_torchaudio_or_huggingface(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    new_state_dict = {_fix_state_dict_key(k): v for k, v in state_dict.items()}
    qkv = {"weight": defaultdict(dict), "bias": defaultdict(dict)}
    for key, tensor in new_state_dict.items():
        match = re.match(r"^(encoder\.layers\.\d+\.attention)\.(q|k|v)_proj\.(weight|bias)$", key)
        if match:
            layer, group, param = match.groups()
            qkv[param][layer][group] = tensor
    for weight in ["weight", "bias"]:
        for layer in qkv[weight]:
            q, k, v = qkv[weight][layer]["q"], qkv[weight][layer]["k"], qkv[weight][layer]["v"]
            new_state_dict[f"{layer}.qkv.{weight}"] = torch.cat((q, k, v), dim=0)
            for group in ["q", "k", "v"]:
                del new_state_dict[f"{layer}.{group}_proj.{weight}"]
    if "logit_generator.label_embeddings" in new_state_dict:
        new_state_dict["feature_weight"] = torch.tensor(FEATURE_WEIGHT)
        new_state_dict["logit_generator.logit_temp"] = torch.tensor(LOGIT_TEMPERATURE)
    return new_state_dict


def _fix_state_dict_key_s3prl(key: str) -> str:
    if key == "mask_emb":
        return "mask_embedding"
    if key == "label_embs_concat":
        return "logit_generator.label_embeddings"
    key = re.sub(r"^wav2vec2\.", "", key)
    key = re.sub(r"^mask_generator\.", "", key)
    key = re.sub(r"^encoder\.transformer\.", "encoder.", key)
    key = re.sub(r"^feature_projection\.", "feature_projection.0.", key)
    key = re.sub(r"^encoder\.feature_projection\.", "feature_projection.0.", key)
    key = re.sub(r"\.out_proj\.", ".proj.", key)
    key = re.sub(r"\.fc1\.", ".feed_forward.intermediate_dense.", key)
    key = re.sub(r"\.fc2\.", ".feed_forward.output_dense.", key)
    key = re.sub(r"\.self_attn\.", ".attention.", key)
    key = re.sub(r"\.self_attn_layer_norm\.", ".layer_norm.", key)
    key = re.sub(r"(feature_extractor\.conv_layers\.\d+)\.0\.weight", r"\1.conv.weight", key)
    key = re.sub(r"(feature_extractor\.conv_layers\.\d+)\.2", r"\1.layer_norm", key)
    key = re.sub(r"^post_extract_proj\.", "feature_projection.0.projection.", key)
    key = re.sub(r"^layer_norm\.", "feature_projection.0.layer_norm.", key)
    key = re.sub(r"^final_proj\.", "logit_generator.final_proj.", key)
    key = re.sub(r"\.weight_g", ".parametrizations.weight.original0", key)
    key = re.sub(r"\.weight_v", ".parametrizations.weight.original1", key)
    return re.sub(r"^encoder\.pos_conv\.", "encoder.pos_conv_embed.convs.", key)


def state_dict_from_s3prl(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    new_state_dict = {_fix_state_dict_key_s3prl(k): v for k, v in state_dict.items()}
    qkv = {"weight": defaultdict(dict), "bias": defaultdict(dict)}
    for key, tensor in new_state_dict.items():
        match = re.match(r"^(encoder\.layers\.\d+\.attention)\.(q|k|v)_proj\.(weight|bias)$", key)
        if match:
            layer, group, param = match.groups()
            qkv[param][layer][group] = tensor
    for weight in ["weight", "bias"]:
        for layer in qkv[weight]:
            q, k, v = qkv[weight][layer]["q"], qkv[weight][layer]["k"], qkv[weight][layer]["v"]
            new_state_dict[f"{layer}.qkv.{weight}"] = torch.cat((q, k, v), dim=0)
            for group in ["q", "k", "v"]:
                del new_state_dict[f"{layer}.{group}_proj.{weight}"]
    if "logit_generator.label_embeddings" in new_state_dict:
        new_state_dict["feature_weight"] = torch.tensor(FEATURE_WEIGHT)
        new_state_dict["logit_generator.logit_temp"] = torch.tensor(LOGIT_TEMPERATURE)
    return new_state_dict
