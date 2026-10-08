import logging
from pathlib import Path

import polars as pl
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from spidr.config import DEFAULT_CONV_LAYER_CONFIG
from spidr.data.dataset import conv_length

from minimal_hubert.utils import conv_length_expr, merge_manifest_with_units

from .conftest import hypothesis_settings


def num_frames(num_samples: int) -> int:
    return int(conv_length(DEFAULT_CONV_LAYER_CONFIG, torch.tensor(num_samples)))


def write_manifest(path: Path, num_samples: dict[str, int]) -> Path:
    manifest = pl.DataFrame(
        {
            "fileid": list(num_samples),
            "path": [f"/data/{fileid}.wav" for fileid in num_samples],
            "num_samples": list(num_samples.values()),
        }
    )
    if path.suffix == ".csv":
        manifest.write_csv(path)
    else:
        manifest.write_ndjson(path)
    return path


def write_units(path: Path, units: dict[str, list[int]]) -> Path:
    pl.DataFrame({"fileid": list(units), "units": list(units.values())}).write_ndjson(path)
    return path


def merge(tmp_path: Path, manifest: Path, units: Path, *, from_mfcc: bool = False) -> pl.DataFrame:
    output = tmp_path / "output.jsonl"
    merge_manifest_with_units(str(manifest), str(units), str(output), from_mfcc=from_mfcc)
    return pl.read_ndjson(output)


@given(num_samples=st.integers(min_value=0, max_value=10_000_000))
@hypothesis_settings
def test_conv_length_expr_matches_spidr(num_samples: int) -> None:
    got = pl.select(conv_length_expr(pl.lit(num_samples))).item()
    assert got == num_frames(num_samples)


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_units_are_truncated_to_the_number_of_frames(tmp_path: Path, suffix: str) -> None:
    """Units are cut to the number of frames given by the CNN, and the manifest columns are kept."""
    num_samples = {"a": 16_000, "b": 32_000}
    manifest = write_manifest(tmp_path / f"manifest{suffix}", num_samples)
    units = write_units(tmp_path / "units.jsonl", {k: list(range(num_frames(n) + 1)) for k, n in num_samples.items()})
    merged = merge(tmp_path, manifest, units)
    assert merged.columns == ["fileid", "path", "num_samples", "units"]
    for row in merged.iter_rows(named=True):
        assert row["units"] == list(range(num_frames(row["num_samples"])))
        assert row["path"] == f"/data/{row['fileid']}.wav"


def test_audio_shorter_than_one_frame_has_no_units(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path / "manifest.csv", {"a": 100})
    units = write_units(tmp_path / "units.jsonl", {"a": [0, 1, 2]})
    assert merge(tmp_path, manifest, units)["units"].to_list() == [[]]


def test_mfcc_units_are_downsampled(tmp_path: Path) -> None:
    """MFCC frames are every 10ms: every other unit is kept, then units are truncated."""
    num_samples = 32_000
    manifest = write_manifest(tmp_path / "manifest.csv", {"a": num_samples})
    units = write_units(tmp_path / "units.jsonl", {"a": list(range(2 * num_frames(num_samples) + 4))})
    merged = merge(tmp_path, manifest, units, from_mfcc=True)
    assert merged["units"].to_list() == [list(range(0, 2 * num_frames(num_samples), 2))]


def test_rows_follow_the_units_order(tmp_path: Path) -> None:
    """Each file gets its own units, even when the manifest and the units file are in different orders."""
    num_samples = {"c": 48_000, "a": 16_000, "b": 32_000}
    manifest = write_manifest(tmp_path / "manifest.csv", num_samples)
    units = write_units(tmp_path / "units.jsonl", {k: [ord(k)] * num_frames(num_samples[k]) for k in ("b", "c", "a")})
    merged = merge(tmp_path, manifest, units)
    assert merged["fileid"].to_list() == ["b", "c", "a"]
    for row in merged.iter_rows(named=True):
        assert row["units"] == [ord(row["fileid"])] * num_frames(num_samples[row["fileid"]])


def test_files_without_units_are_dropped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    manifest = write_manifest(tmp_path / "manifest.csv", {"a": 16_000, "b": 16_000, "c": 16_000})
    units = write_units(tmp_path / "units.jsonl", {"a": [0] * 100, "c": [0] * 100})
    with caplog.at_level(logging.WARNING):
        merged = merge(tmp_path, manifest, units)
    assert merged["fileid"].to_list() == ["a", "c"]
    assert "1/3 files in the manifest have no units" in caplog.text


def test_units_without_manifest_entry_are_dropped(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path / "manifest.csv", {"a": 16_000})
    units = write_units(tmp_path / "units.jsonl", {"a": [0] * 100, "z": [0] * 100})
    assert merge(tmp_path, manifest, units)["fileid"].to_list() == ["a"]


def test_duplicate_fileids_in_units(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path / "manifest.csv", {"a": 16_000, "b": 16_000})
    units = tmp_path / "units.jsonl"
    units.write_text('{"fileid":"a","units":[0]}\n{"fileid":"b","units":[0]}\n{"fileid":"a","units":[1]}\n')
    with pytest.raises(ValueError, match="Duplicate fileids in the units file"):
        merge(tmp_path, manifest, units)


def test_duplicate_fileids_in_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("fileid,path,num_samples\na,/data/a.wav,16000\na,/data/other/a.wav,16000\n")
    units = write_units(tmp_path / "units.jsonl", {"a": [0] * 100})
    with pytest.raises(ValueError, match="Duplicate fileids in the manifest file"):
        merge(tmp_path, manifest, units)
