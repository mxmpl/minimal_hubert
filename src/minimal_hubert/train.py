"""Training loop."""

import logging
import os
from contextlib import ExitStack

import torch
import wandb
from spidr.checkpoint import Checkpointer
from spidr.environment import setup_training
from spidr.tools import AverageMeters, profiler_context
from spidr.train import init_wandb
from torch import GradScaler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from .config import Config, read_config
from .data import build_dataloader_with_labels
from .model import HuBERTPretrain

logger = logging.getLogger()


class LinearDecayLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        max_updates: int,
    ) -> None:
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        if self._step_count <= self.warmup_updates:
            return [self._step_count / self.warmup_updates * base_lr for base_lr in self.base_lrs]
        if self._step_count >= self.max_updates:
            return [0.0 for _ in self.base_lrs]
        pct_remaining = (self.max_updates - self._step_count) / (self.max_updates - self.warmup_updates)
        return [base_lr * pct_remaining for base_lr in self.base_lrs]


def train(cfg: Config) -> None:
    with ExitStack() as stack:
        logger.info("Starting job")
        setup_training(cfg.run.random_seed, use_deterministic=cfg.run.use_deterministic)
        stack.callback(dist.destroy_process_group)
        global_rank, world_size = dist.get_rank(), dist.get_world_size()
        is_main = global_rank == 0
        if is_main:
            init_wandb(cfg)
            stack.callback(wandb.finish)

        logger.info("Building model, optimizer, and dataloaders")
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.optimizer.dtype]
        model = HuBERTPretrain(cfg.num_classes).to(device).train()
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            fused=True,
        )
        scaler = GradScaler("cuda", enabled=cfg.optimizer.mixed_precision)
        scheduler = LinearDecayLRScheduler(
            optimizer,
            cfg.optimizer.warmup_steps,
            cfg.optimizer.warmup_steps + cfg.optimizer.decay_steps,
        )
        loader = build_dataloader_with_labels(cfg.data, cfg.masking)
        dist.barrier(device_ids=[device.index])
        ckpt = Checkpointer(cfg.run.dir, cfg.run.save_interval, cfg.run.keep_latest)
        ckpt.init_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        ckpt.load_existing_run()
        step, epoch = int(ckpt.step), int(ckpt.epoch)
        stack.callback(lambda: ckpt.save(step, epoch))
        model.compile(dynamic=True)
        ddp_model = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)

        logger.info("Starting training loop")
        meters = AverageMeters(["loss", "grad_norm", "batch_size", "feature_loss"], device=device)
        profiler = stack.enter_context(profiler_context(cfg.run.dir / "trace.html" if is_main else None))
        pbar = stack.enter_context(tqdm(total=cfg.optimizer.max_steps, initial=step, disable=not is_main))
        while step < cfg.optimizer.max_steps:
            epoch += 1
            loader.batch_sampler.set_epoch(epoch)
            logger.info("Starting epoch %s", epoch)
            for waveforms, labels, attn_mask, mask in loader:
                if step >= cfg.optimizer.max_steps:
                    break
                with torch.autocast("cuda", dtype, cfg.optimizer.mixed_precision):
                    loss, outputs = ddp_model(
                        waveforms.to(device),
                        labels.to(device),
                        mask=mask.to(device),
                        attention_mask=attn_mask.to(device),
                    )
                num_frames = torch.tensor(loss.size(0), dtype=torch.long, device=device)
                dist.all_reduce(num_frames)
                loss = loss.sum() * world_size / num_frames
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(ddp_model.parameters(), cfg.optimizer.max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr = scheduler.get_last_lr()[0]
                scheduler.step()
                step += 1

                meters.update(
                    loss=loss.detach(),
                    batch_size=waveforms.size(0),
                    grad_norm=grad_norm,
                    feature_loss=outputs["feature_loss"],
                )
                pbar.update()
                if is_main and step % cfg.run.log_interval == 0:
                    infos = meters.pop() | {"lr": lr, "step": step, "epoch": epoch}
                    wandb.log({f"train/{key}": value for key, value in infos.items()})
                    pbar.set_postfix(loss=infos["loss"], feature_loss=infos["feature_loss"])
                    ckpt.save(step, epoch)
                profiler.step()

        if is_main:
            ckpt.save_final(step, epoch)
        logger.info("Training finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    cfg = read_config(parser.parse_args().config)
    train(cfg)
