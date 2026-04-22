import re
import sys
import types
import warnings
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from pickle import UnpicklingError
from typing import Literal, overload

import torch
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchaudio.models.wav2vec2.utils.import_fairseq import _convert_state_dict

_LOGIT_TEMPERATURE = 0.1

type Size = Literal["base", "large", "xlarge"]


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


def _convert_from_torchaudio_or_hf(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
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
        new_state_dict["logit_generator.logit_temp"] = torch.tensor(_LOGIT_TEMPERATURE)
    if "feature_weight" in new_state_dict:
        assert new_state_dict.pop("feature_weight").item() == 1.0
    return new_state_dict


def _convert_from_s3prl(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
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
        new_state_dict["logit_generator.logit_temp"] = torch.tensor(_LOGIT_TEMPERATURE)
    return new_state_dict


def _convert_from_fairseq(state_dict: dict[str, torch.Tensor], *, for_pretraining: bool) -> dict[str, torch.Tensor]:
    new_state_dict = _convert_state_dict(state_dict)
    if for_pretraining:
        new_state_dict["mask_embedding"] = state_dict["mask_emb"]
        new_state_dict["logit_generator.final_proj.weight"] = state_dict["final_proj.weight"]
        new_state_dict["logit_generator.final_proj.bias"] = state_dict["final_proj.bias"]
        new_state_dict["logit_generator.label_embeddings"] = new_state_dict.pop("label_embs_concat")[:-4]
    else:
        del new_state_dict["label_embs_concat"]
    return new_state_dict


def size_from_state_dict(state_dict: dict[str, Tensor]) -> Size:
    prefix = "encoder.layers."
    layers = {int(k.removeprefix(prefix).split(".")[0]) for k in state_dict if k.startswith(prefix)}
    if layers == set(range(12)):
        return "base"
    if layers == set(range(24)):
        return "large"
    if layers == set(range(48)):
        return "xlarge"
    raise ValueError(f"Invalid model size configuration. We found those layers: {layers}")


@overload
def convert_hubert_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    for_pretraining: Literal[True],
) -> tuple[dict[str, torch.Tensor], int, Size]: ...


@overload
def convert_hubert_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    for_pretraining: Literal[False],
) -> tuple[dict[str, torch.Tensor], Size]: ...


def convert_hubert_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    for_pretraining: bool,
) -> tuple[dict[str, torch.Tensor], int, Size] | tuple[dict[str, torch.Tensor], Size]:
    new_state_dict = state_dict.get("model", state_dict)
    new_state_dict = _convert_from_fairseq(new_state_dict, for_pretraining=for_pretraining)
    new_state_dict = _convert_from_torchaudio_or_hf(new_state_dict)
    # new_state_dict = _convert_from_s3prl(new_state_dict)
    if not for_pretraining:
        return new_state_dict, size_from_state_dict(new_state_dict)
    num_classes = new_state_dict["logit_generator.label_embeddings"].size(0)
    return new_state_dict, num_classes, size_from_state_dict(new_state_dict)


@contextmanager
def fake_fairseq_dictionary() -> Generator[None, None, None]:
    """Trick the sys.modules to be able to load fairseq checkpoints without fairseq installed."""
    original = {name: sys.modules.get(name) for name in ("fairseq", "fairseq.data", "fairseq.data.dictionary")}
    try:
        sys.modules.setdefault("fairseq", types.ModuleType("fairseq"))
        sys.modules.setdefault("fairseq.data", types.ModuleType("fairseq.data"))
        m = types.ModuleType("fairseq.data.dictionary")
        m.Dictionary = type("Dictionary", (), {})  # ty: ignore[unresolved-attribute]
        sys.modules["fairseq.data.dictionary"] = m
        yield
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_state_dict_from_remote_or_local(path_or_url: str | Path) -> dict[str, torch.Tensor]:
    loader = torch.load if Path(path_or_url).is_file() else load_state_dict_from_url
    try:
        state_dict = loader(str(path_or_url), map_location="cpu", weights_only=True)
    except UnpicklingError:
        warnings.warn("Loaded with 'weights_only=False'. Expected only if this is a fairseq checkpoint.", stacklevel=1)
        with fake_fairseq_dictionary():
            state_dict = loader(str(path_or_url), map_location="cpu", weights_only=False)
    return state_dict


def known_huberts() -> dict[Size, list[str]]:
    return {
        "base": [
            "facebook/hubert-base-ls960",
            "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
            "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt",
            "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_mls_cv_8lang_it3.pt",
            "utter-project/mHuBERT-147",
            "utter-project/mHuBERT-147-base-1st-iter",
            "utter-project/mHuBERT-147-base-2nd-iter",
            "reazon-research/japanese-hubert-base-k2",
            "TencentGameMate/chinese-hubert-base",
            # "coml/hubert-base-vp20",
            # "coml/hubert-base-mmsulab",
            # "https://huggingface.co/espnet/espnet_cvhubert/blob/main/exp/hubert_iter2_train_ssl_torchaudiohubert_base_960h_pretrain_it2_raw/latest.pth",
        ],
        "large": [
            "facebook/hubert-large-ll60k",
            "TencentGameMate/chinese-hubert-large",
            "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt",
        ],
        "xlarge": [
            "facebook/hubert-xlarge-ll60k",
            "https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt",
        ],
    }
