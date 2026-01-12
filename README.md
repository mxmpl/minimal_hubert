# Minimal implementation of HuBERT pretraining

This package provides a minimal implementation of HuBERT pretraining recipe.
It uses core functions from the [SpidR codebase](https://github.com/facebookresearch/spidr).

With this code, you should be able to train one HuBERT iteration of 400k steps in 10 hours on 16 H100s.
We also provide distributed inference and derivation of discrete units to speed up the intermediate steps between pretraining iterations.

## Install

Install the Python package:

```bash
pip install minimal-hubert
```

Or clone this repository:

```bash
git clone https://github.com/mxmpl/minimal_hubert
cd minimal_hubert
uv sync
```

## Usage

**Load a pretrained model:**

```python
from minimal_hubert import HuBERTPretrain

model = HuBERTPretrain.from_pretrained("./path/to/checkpoint.pt")
```

This method can load a checkpoint obtained with the package, but also from torchaudio or HuggingFace.


## Pretraining HuBERT, step by step

You will find below a short tutorial on how to pretrain HuBERT from scratch.
It can be followed step by step, or adapted to your needs and constraints.
You can skip the first pretraining iteration if you don't want to start from scratch but rather from features from a
pretrained model.

This tutorial expects you to have access to a SLURM cluster.
Ideally, you should have access to at least 16 H100 or A100 GPUs at a time for pretraining.
It's also better to have access to a lot of (possibly older) GPUs in parallel for fast extraction of intermediate features.
This tutorial been tested on the Jean Zay supercomputer.

You can modify the arguments of the scripts in `slurm/` to adapt to your cluster constraints, or directly use the
corresponding entry-points.

Replace all environment variables below with the actual paths:
- Both iterations:
  - `ROOT_TRAIN_AUDIO`, `ROOT_VAL_AUDIO`: directories with audio files
  - `PATH_TRAIN_MANIFEST`, `PATH_VAL_MANIFEST`: path to the manifest files ("fileid", "num_samples", and "path")
  - `PATH_ABX_ITEM`: path to the item file for ABX discriminability
  - `ROOT_ABX_AUDIO`: directory with audio files for ABX
- Iteration specific:
  - `ROOT_TRAIN_MFCC`, `ROOT_VAL_MFCC`: directories for MFCC features (for it1)
  - `ROOT_TRAIN_FEATURES`, `ROOT_VAL_FEATURES`: directories for HuBERT features (for it2)
  - `PATH_KMEANS_IT{1,2}`: path to the K-means checkpoint
  - `UNITS_JSONL_IT{1,2}`: path to the JSONL with discrete units
  - `PATH_TRAIN_MANIFEST_WITH_UNITS_IT{1,2}`, `PATH_VAL_MANIFEST_WITH_UNITS_IT1`: path to the manifest files with units
  - `PATH_CHECKPOINTS_IT{1,2}`: directory with all checkpoints (in "workdir")
  - `VALIDATION_JSONL_IT{1,2}`: JSONL file with validation losses
  - `OUTPUT_ABX_IT{1,2}`: JSONL file with ABX error rates

### Prerequisites

First, you need to have manifest files of your training and validation datasets, with the list of audio files and
number of samples.
You can create them like this:

```bash
python -m spidr.data.write_manifest $ROOT_TRAIN_AUDIO $PATH_TRAIN_MANIFEST --ext .wav
python -m spidr.data.write_manifest $ROOT_VAL_AUDIO $PATH_VAL_MANIFEST --ext .wav
```

### First iteration

#### Extract MFCC

Extract MFCCs in parallel with this:

```bash
sbatch slurm/features.slurm $PATH_TRAIN_MANIFEST $ROOT_TRAIN_MFCC mfcc
sbatch slurm/features.slurm $PATH_VAL_MANIFEST $ROOT_VAL_MFCC mfcc
```

If you can run 128 jobs in parallel, it will be very quick.

#### Train K-means on MFCC

Then, fit a K-means with 100 clusters on the MFCC frames, using 10% of the audio files:

```bash
sbatch slurm/kmeans.slurm $ROOT_TRAIN_MFCC $PATH_KMEANS_IT1 100 10
```

This should be quite fast if you have good CPUs.

#### Transcribe MFCC in discrete units

Use the fitted K-means to derive discrete units in parallel:

```bash
sbatch slurm/transcribe.slurm $ROOT_TRAIN_MFCC $PATH_KMEANS_IT1 $UNITS_JSONL_IT1
sbatch slurm/transcribe.slurm $ROOT_VAL_MFCC $PATH_KMEANS_IT1 $UNITS_JSONL_IT1
```

Then merge your manifests with the resulting JSONL to have manifests with units, ready for pretraining:

```bash
python -m minimal_hubert.utils.merge_manifest_with_units $PATH_TRAIN_MANIFEST $UNITS_JSONL_IT1 $PATH_TRAIN_MANIFEST_WITH_UNITS_IT1 --from-mfcc
python -m minimal_hubert.utils.merge_manifest_with_units $PATH_VAL_MANIFEST $UNITS_JSONL_IT1 $PATH_VAL_MANIFEST_WITH_UNITS_IT1 --from-mfcc
```

This may use a lot of memory, so run this on a compute node.

#### First pretraining iteration

Fill the empty fields in `./configs/it1.toml` ("manifest" and "workdir").
Then launch a pretraining job:

```bash
python -m minimal_hubert ./configs/it1.toml -N 4 -G 4 -c 24 -t 1200 -C h100
```

Adapt the arguments to your specific cluster. This will take ~10 hours if you have 16 H100s.

#### Select the best checkpoint

Compute the validation loss for all intermediate checkpoints:

```bash
sbatch scripts/validate.slurm $PATH_VAL_MANIFEST_WITH_UNITS_IT1 $PATH_CHECKPOINTS_IT1 $VALIDATION_JSONL_IT1
```

Find the best checkpoint and create a symlink:

```bash
python -m minimal_hubert.utils.best_checkpoint $PATH_CHECKPOINTS_IT1 $VALIDATION_JSONL_IT1
```

### Second iteration

Same steps as before, but we now use representations derived from an intermediate of the model obtained previously
instead of MFCCs.

#### Find the best layer

We consider that the best layer is the one that maximizes the ABX discriminability of the intermediate representations.
You will need to find an "item" file with forced alignment at phoneme or triphone level to compute ABX.

```bash
sbatch scripts/abx.slurm $PATH_ABX_ITEM $ROOT_ABX_AUDIO $OUTPUT_ABX_IT1 $PATH_CHECKPOINTS_IT1/best.pt
```

Check out the error rates in `$OUTPUT_ABX`, and select the layer with the lowest ones.

#### Extract features from the best layer

```bash
sbatch scripts/features.slurm $PATH_TRAIN_MANIFEST $ROOT_TRAIN_FEATURES hubert $PATH_CHECKPOINTS_IT1/best.pt $BEST_LAYER_IT1
sbatch scripts/features.slurm $PATH_VAL_MANIFEST $ROOT_VAL_FEATURES hubert $PATH_CHECKPOINTS_IT1/best.pt $BEST_LAYER_IT1
```

#### Train K-means on features

```bash
sbatch scripts/kmeans.slurm $ROOT_TRAIN_FEATURES $PATH_KMEANS_IT2 500 10
# sbatch slurm/kmeans.slurm $ROOT_TRAIN_FEATURES $PATH_KMEANS_IT2 500 20 # If you want use 5% of files instead
```

This can run for a long time if you have a large dataset (much longer than K-means on MFCCs).
Adjust you subsampling ratio accordingly.

#### Transcribe features in discrete units

```bash
sbatch scripts/transcribe.slurm $ROOT_TRAIN_FEATURES $UNITS_JSONL_IT2 $PATH_KMEANS_IT2
sbatch scripts/transcribe.slurm $ROOT_VAL_FEATURES $UNITS_JSONL_IT2 $PATH_KMEANS_IT2
```

```bash
python -m minimal_hubert.utils.merge_manifest_with_units $PATH_TRAIN_MANIFEST $UNITS_JSONL_IT2 $PATH_TRAIN_MANIFEST_WITH_UNITS_IT2
python -m minimal_hubert.utils.merge_manifest_with_units $PATH_VAL_MANIFEST $UNITS_JSONL_IT2 $PATH_VAL_MANIFEST_WITH_UNITS_IT2
```

#### Second pretraining iteration

Fill the empty fields in `./configs/it2.toml` ("manifest" and "workdir").
Then launch a pretraining job:

```bash
python -m minimal_hubert ./configs/it2.toml -N 4 -G 4 -c 24 -t 1200 -C h100
```

#### Select the best checkpoint

```bash
sbatch scripts/validate.slurm $PATH_VAL_MANIFEST_WITH_UNITS_IT2 $PATH_CHECKPOINTS_IT2 $VALIDATION_JSONL_IT2
```

```bash
python -m minimal_hubert.utils.best_checkpoint $PATH_CHECKPOINTS_IT2 $VALIDATION_JSONL_IT2
```

## Citation

This codebase heavily borrows from [SpidR](https://github.com/facebookresearch/spidr):
```bibtex
@article{
  poli2025spidr,
  title={SpidR: Learning Fast and Stable Linguistic Units for Spoken Language Models Without Supervision},
  author={Maxime Poli and Mahi Luthra and Youssef Benchekroun and Yosuke Higuchi and Martin Gleize and Jiayi Shen and Robin Algayres and Yu-An Chung and Mido Assran and Juan Pino and Emmanuel Dupoux},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=E7XAFBpfZs},
}
```

HuBERT:
```bibtex
@article{
  9585401,
  title={HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units},
  author={Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2021},
  volume={29},
  number={},
  pages={3451-3460},
  keywords={Predictive models;Representation learning;Self-supervised learning;Self-supervised learning;BERT},
  doi={10.1109/TASLP.2021.3122291},
}
```
