import polars as pl
from spidr.config import DEFAULT_CONV_LAYER_CONFIG
from spidr.data.dataset import conv_length


def get_fileids_from_manifest_and_units(path_manifest: str, path_units: str) -> set[str]:
    manifest = pl.scan_ndjson(path_manifest).sort("fileid").select("fileid").collect()
    fileids = set(manifest["fileid"].unique())
    units = pl.scan_ndjson(path_units).sort("fileid").select("fileid").filter(pl.col("fileid").is_in(fileids))
    assert manifest.equals(units)
    return fileids


def merge_manifest_with_units(path_manifest: str, path_units: str, *, from_mfcc: bool) -> pl.DataFrame:
    fileids = get_fileids_from_manifest_and_units(path_manifest, path_units)
    length = conv_length(DEFAULT_CONV_LAYER_CONFIG, pl.read_ndjson(path_manifest)["num_samples"].to_torch())
    return (
        pl.concat(
            (
                pl.read_ndjson(path_manifest).sort("fileid"),
                pl.scan_ndjson(path_units)
                .sort("fileid")
                .filter(pl.col("fileid").is_in(fileids))
                .drop("fileid")
                .with_columns(pl.col("units").list.gather_every(2) if from_mfcc else pl.col("units"))
                .collect(),
            ),
            how="horizontal",
        )
        .with_columns(pl.Series(name="length", values=length))
        .with_columns(pl.col("units").list.slice(offset=0, length=pl.col("length")))
        .drop("length")
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path-manifest", help="Path to manifest file")
    parser.add_argument("units", help="Path to the JSONL file with units")
    parser.add_argument("output", help="Path to the output manifest file with units")
    parser.add_argument(
        "--from-mfcc",
        type=bool,
        action="store_true",
        help="Add this flag if units are derived from MFCC (10ms instead of 20ms)",
    )
    args = parser.parse_args()
    merge_manifest_with_units(args.manifest, args.units, from_mfcc=args.from_mfcc).write_ndjson(args.output)
