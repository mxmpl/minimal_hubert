"""Entry point for launching pretraining."""

import argparse
from pathlib import Path

from spidr.slurm import launch_with_submitit, slurm_config_parse_args
from spidr.tools import init_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch HuBERT pretraining.", allow_abbrev=False)
    parser.add_argument("configs", type=Path, nargs="+", help="TOML config file(s) for the training run(s).")
    parser.add_argument("-A", "--account", type=str, required=True, help="Slurm account")
    parser.add_argument("-N", "--nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("-G", "--gpus-per-node", type=int, required=True, help="GPUs per node")
    parser.add_argument("-c", "--cpus-per-task", type=int, help="CPUs per task")
    parser.add_argument("--mem-per-gpu", type=str, help="Memory per GPU")
    parser.add_argument("-t", "--time", type=int, help="Time limit in minutes")
    parser.add_argument("-C", "--constraint", type=str, help="Slurm constraint")
    parser.add_argument("-p", "--partition", type=str, help="Slurm partition")
    parser.add_argument("-q", "--qos", type=str, help="Slurm QoS")
    parser.add_argument("--dump", type=Path, help="Submitit dump", required=True)
    args = parser.parse_args()

    from .config import read_config
    from .train import train

    init_logger()
    jobs = [(train, (read_config(cfg),)) for cfg in args.configs]
    name = jobs[0][1][0].run.wandb_name
    launch_with_submitit(name, jobs, args.dump, slurm_config_parse_args(args), copy_code=False)
