from dataclasses import dataclass
import gc
from pathlib import Path
from collections import defaultdict

import os
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map
from omegaconf import OmegaConf
from safetensors.torch import load_file
from accel_hydra.trainer import Trainer
from accel_hydra.trainer.base import LRSchedulerInterval
from accel_hydra.utils.logging import LoggingLogger
from accel_hydra.utils.accelerate import AcceleratorSaveTrainableParams

from accelerate import DistributedDataParallelKwargs
@dataclass(kw_only=True)
class CASTTrainer(Trainer):

    logging_file: str | Path
    pretrained_weights: Path | None = None
    def setup_accelerator(self) -> None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        tracker = None
        if self.logging_config is not None:
            assert self.logging_config.report_to in (
                "wandb", "swanlab", "tensorboard"
            ), (
                f"Unsupported logger: {self.logging_config.report_to}. "
                "Supported loggers are 'wandb', 'swanlab', and 'tensorboard'."
            )
            if self.logging_config.report_to == "swanlab":
                from swanlab.integration.accelerate import SwanLabTracker
                tracker = SwanLabTracker(
                    run_name=self.logging_config.project,
                    experiment_name=self.logging_config.name,
                    logdir=self.logging_config.save_dir,
                    workspace=self.logging_config.workspace,
                    resume=True,
                    id=self.logging_config.resume_id
                )
            else:
                tracker = self.logging_config.report_to

        self.accelerator = AcceleratorSaveTrainableParams(
            log_with=tracker,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            project_dir=self.project_dir,
            step_scheduler_with_optimizer=(
                self.lr_scheduler_interval == LRSchedulerInterval.STEP
            ),
            kwargs_handlers=[ddp_kwargs]
        )

        train_batch_sampler = self.train_dataloader.batch_sampler
        if not hasattr(train_batch_sampler, "batch_size"):
            assert self.even_batches is False, "even_batches must be False when batch_sampler does not have batch_size"

            # due to this line: https://github.com/huggingface/accelerate/blob/main/src/accelerate/data_loader.py#L246
            assert getattr(
                train_batch_sampler, "drop_last", False
            ) is True, "drop_last must be True when batch_sampler does not have batch_size"

        self.accelerator.even_batches = self.even_batches
        # TODO when `loss_fn` does not have named_parameters/buffers, loading will raise error
        (
            self.train_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        self.accelerator.register_for_checkpointing(self)
        for checkpoint_object in self.checkpoint_objects:
            self.accelerator.register_for_checkpointing(checkpoint_object)
        if self.pretrained_weights is not None:
            self.accelerator.print(
                f"start from checkpoint: {self.pretrained_weights}"
            )
            ckpt_path = self.pretrained_weights
            if os.path.isdir(ckpt_path):
                ckpt_path = os.path.join(ckpt_path, "model.safetensors")
            state_dict = load_file(ckpt_path)
            self.model.module.load_pretrained(state_dict)

        if self.resume_from_checkpoint is not None:
            self.accelerator.print(
                f"resume from checkpoint: {self.resume_from_checkpoint}"
            )
            self.accelerator.load_state(
                self.resume_from_checkpoint, strict=False
            )
            
    def on_train_start(self):
        super().on_train_start()
        self.train_loss = 0
        self.train_batch_num = 0
        if self.accelerator.is_main_process:
            self.logger = LoggingLogger(self.logging_file).create_instance()
            with open(self.project_dir / "config.yaml", "w") as writer:
                OmegaConf.save(self.config_dict, writer)

        if isinstance(self.model, DistributedDataParallel):
            num_params, trainable_params = self.model.module.count_params()
        else:
            num_params, trainable_params = self.model.count_params()

        if self.accelerator.is_main_process:
            self.logger.info(
                f"parameter number: {num_params}, trainable parameter number: {trainable_params}"
            )

    def on_train_epoch_start(self):
        self.train_loss = 0
        self.train_batch_num = 0
        loaders = self.train_dataloader
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        for dl in loaders:
            bs = getattr(dl, "batch_sampler", None)
            if hasattr(bs, "set_epoch"):
                bs.set_epoch(self.epoch)
                print("set_epoch", flush=True)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.log({"train/lr": lr}, step=self.step)
        loss_dict = self.loss_fn(output)
        log_dict = {}
        for loss_name in loss_dict:
            if loss_name != "loss":
                log_dict[f"train/{loss_name}"] = loss_dict[loss_name].item()
        self.accelerator.log(
            log_dict,
            step=self.step,
        )
        loss = loss_dict["loss"]
        if torch.isfinite(loss):
            self.train_loss += loss.item()
        else:
            print(f"Loss is not finite (nan or inf) at batch_idx {batch_idx}!")
            for name, param in self.model.named_parameters():
                has_nan = torch.isnan(param).any()
                has_inf = torch.isinf(param).any()
                if has_nan or has_inf:
                    nan_count = torch.isnan(param).sum().item()
                    inf_count = torch.isinf(param).sum().item()
                    print(f"param {name}: nan num {nan_count}, inf num {inf_count}, shape={tuple(param.shape)}", flush=True)
        self.train_batch_num += 1
        return loss

    def on_validation_start(self):
        self.val_loss_dict = defaultdict(float)
        self.val_batch_num = 0
        self.val_time_aligned_batch_num = 0

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        output = self.accelerator.gather_for_metrics(output)
        output = tree_map(lambda x: x.mean(), output)
        loss_dict = self.loss_fn(output)
        for loss_name in loss_dict:
            self.val_loss_dict[loss_name] += loss_dict[loss_name].item()
        self.val_batch_num += 1

    def get_val_metrics(self):
        metric_name = self.metric_monitor.metric_name
        assert metric_name in self.val_loss_dict, \
            f"{metric_name} not found in validation loss dict"
        return {
            metric_name: self.val_loss_dict[metric_name] / self.val_batch_num
        }

    def on_train_epoch_end(self):
        train_loss = self.train_loss / self.train_batch_num
        val_loss = self.val_loss_dict["loss"] / self.val_batch_num
        log_dict = {}
        for loss_name in self.val_loss_dict:
            log_dict[f"val/{loss_name}"
                    ] = self.val_loss_dict[loss_name] / self.val_batch_num
        self.accelerator.log(
            log_dict,
            step=self.step,
        )
        logging_msg = f"epoch[{self.epoch}], train loss: {train_loss:.3f}, val loss: {val_loss:.3f}"
        self.accelerator.print(logging_msg)
        if self.accelerator.is_main_process:
            self.logger.info(logging_msg)

        torch.cuda.empty_cache()
        gc.collect()
