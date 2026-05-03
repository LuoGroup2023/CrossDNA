# -*- coding: utf-8 -*-
import os
import sys
import copy
import random
import time
from functools import wraps
from typing import Any, Callable, List, Sequence

# =============================================================================
# Early environment hooks (must be before importing torch)
# =============================================================================
# 1) Disable torch.compile / Dynamo / Inductor (make @torch.compile no-op)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# if "CUDA_VISIBLE_DEVICES" not in os.environ:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" # 或者干脆不设置，由 shell 决定

# 2) Prefer conda libs, avoid mixing with /usr/local/cuda-*
_CONDA = os.environ.get("CONDA_PREFIX")
if _CONDA:
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    torchlib = os.path.join(_CONDA, f"lib/python{pyver}/site-packages/torch/lib")
    old = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in old.split(":") if p and "/usr/local/cuda" not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join([os.path.join(_CONDA, "lib"), torchlib] + parts)

# NCCL / Hydra
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

# Project root for imports (stable even if Hydra chdir)
if os.getenv("PROJECT_ROOT", None) is None:
    os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# Imports
# =============================================================================
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig as _DictConfig
from omegaconf.nodes import AnyNode
from omegaconf.base import ContainerMetadata, Metadata
from collections import defaultdict

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.strategies.ddp import DDPStrategy

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks

log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up training substantially)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# OmegaConf resolvers
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

# Allow ListConfig/DictConfig to be deserialized by torch.load
torch.serialization.add_safe_globals(
    [ListConfig, _DictConfig, ContainerMetadata, Any, Metadata, list, defaultdict, dict, int, AnyNode]
)

# =============================================================================
# WandB retry logger
# =============================================================================
class DummyExperiment:
    """Dummy experiment for non-rank0 processes."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx):
        return self

    def __setitem__(self, *args, **kwargs):
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns real experiment on rank 0 and DummyExperiment otherwise."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):
    """WandbLogger that insists on wandb.init() and retries on failure."""

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)

            if wandb.run is not None:
                rank_zero_warn(
                    "A wandb run is already in progress; this WandbLogger will reuse it. "
                    "Call wandb.finish() first if this is not desired."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                self._experiment = wandb._attach(attach_id)
            else:
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        print("wandb.init Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds before retry...")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


# =============================================================================
# LightningModule
# =============================================================================
class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        # Reduce memory usage / speed up
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()

        # Save full config into self.hparams (but don't let PL logger auto-log it here)
        self.save_hyperparameters(config, logger=False)

        # Dataset
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Config sanity
        self._check_config()

        # Hook guard (avoid duplicate setup under DDP)
        self._has_setup = False

        # Internal state for TBPTT/BPTT etc.
        self._initialize_state()

        # Setup model/task/enc/dec
        self.setup()

    def setup(self, stage: str | None = None):
        if not self.hparams.train.disable_dataset:
            self.dataset.setup()

        if self._has_setup:
            return
        self._has_setup = True

        # Combine encoder/decoder configs
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate backbone model
        self.model = utils.instantiate(registry.model, self.hparams.model)

        # Optional post-init hook
        if (name := self.hparams.train.post_init_hook.get("_name_", None)) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            kwargs.pop("_name_", None)
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # Instantiate task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Encoder/decoder modules
        encoder = encoders.instantiate(encoder_cfg, dataset=self.dataset, model=self.model)
        decoder = decoders.instantiate(decoder_cfg, model=self.model, dataset=self.dataset)

        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)

        self.loss = self.task.loss
        self.loss_val = getattr(self.task, "loss_val", self.task.loss)
        self.metrics = self.task.metrics

        # Torchmetrics holders
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics

    # -------------------------
    # Config checks / state helpers
    # -------------------------
    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        for key in ["n_context", "n_context_eval"]:
            n = self.hparams.train.state.get(key)
            assert n is None or (isinstance(n, int) and n >= 0)

    def _initialize_state(self):
        self._state = None
        self._memory_chunks = []
    
    def _set_module_trainable(self, module, requires_grad: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = requires_grad

    # GB、NT微调set
    def _apply_freeze_unfreeze_policy(self):
        freeze_epochs = int(self.hparams.train.get("freeze_backbone_epochs", 0) or 0)

        # 默认 0，不影响预训练，也不影响没有开启该选项的任务
        if freeze_epochs <= 0:
            return

        if not hasattr(self, "_freeze_policy_state"):
            self._freeze_policy_state = None

        # 前 freeze_epochs 个 epoch：冻结 backbone，只训练 encoder/decoder/head
        if self.current_epoch < freeze_epochs:
            if self._freeze_policy_state != "frozen":
                self._set_module_trainable(self.model, False)
                self._set_module_trainable(self.encoder, True)
                self._set_module_trainable(self.decoder, True)
                self._freeze_policy_state = "frozen"
                print(f"[FreezePolicy] Epoch {self.current_epoch}: backbone frozen, head trainable.")

        # 之后：全部解冻
        else:
            if self._freeze_policy_state != "unfrozen":
                self._set_module_trainable(self.model, True)
                self._set_module_trainable(self.encoder, True)
                self._set_module_trainable(self.decoder, True)
                self._freeze_policy_state = "unfrozen"
                print(f"[FreezePolicy] Epoch {self.current_epoch}: backbone unfrozen, full finetuning.")

    def _reset_state(self, batch, device=None):
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        if isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        if isinstance(state, list):
            return [self._detach_state(s) for s in state]
        if isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        if state is None:
            return None
        raise NotImplementedError

    def _process_state(self, batch, batch_idx, train=True):
        key = "n_context" if train else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        if n_context == 0 and self.hparams.train.state.mode not in ["tbptt"]:
            self._initialize_state()
            return

        mode = self.hparams.train.state.mode

        if mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        elif mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif mode == "tbptt":
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, batch):
        return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state)

    def step(self, x_t):
        x_t, *_ = self.encoder(x_t)
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    # -------------------------
    # Shared step
    # -------------------------
    def _shared_step(self, batch, batch_idx, prefix="train"):
        self._process_state(batch, batch_idx, train=(prefix == "train"))
        x, y, w = self.forward(batch)

        loss = self.loss(x, y, **w) if prefix == "train" else self.loss_val(x, y, **w)

        metrics = self.metrics(x, y, **w)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        log_on_step = ("eval" in self.hparams) and self.hparams.eval.get("log_on_step", False) and (prefix == "train")

        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    # -------------------------
    # Lightning hooks (Lightning 2.x compatible)
    # -------------------------
    # def on_train_epoch_start(self):
    #     if hasattr(self.task, "_reset_torchmetrics"):
    #         self.task._reset_torchmetrics("train")

    def on_train_epoch_start(self):
        # 兼顾微调和预训练
        self._apply_freeze_unfreeze_policy()

        if hasattr(self.task, "_reset_torchmetrics"):
            self.task._reset_torchmetrics("train")

    def on_validation_epoch_start(self):
        if hasattr(self.task, "_reset_torchmetrics"):
            for name in getattr(self, "val_loader_names", []):
                self.task._reset_torchmetrics(name)

    def on_test_epoch_start(self):
        if hasattr(self.task, "_reset_torchmetrics"):
            for name in getattr(self, "test_loader_names", []):
                self.task._reset_torchmetrics(name)

    # -------------------------
    # Steps
    # -------------------------
    def training_step(self, batch, batch_idx, dataloader_idx=0):

        # 只打印一次
        if batch_idx == 0 and not hasattr(self, "_printed_batch"):
            self._printed_batch = True
            b = batch
            import torch
            print("batch type:", type(b), "len:", len(b))
            print("inputs type:", type(b[0]), "len:", len(b[0]))
            print("masked_seq:", b[0][0].shape, b[0][0].dtype)
            print("mask:", b[0][1].shape, b[0][1].dtype)
            print("mlm_labels:", b[0][2].shape, b[0][2].dtype,
                "num(-100)=", (b[0][2] == -100).sum().item())
            print("target:", b[1].shape, b[1].dtype)

        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Explicit step-wise trainer loss for WandB
        self.log_dict(
            {"trainer/loss": loss, "trainer/epoch": self.current_epoch},
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Any extra module metrics
        extra = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                extra.update(module.metrics)

        if extra:
            self.log_dict(
                extra,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                add_dataloader_idx=False,
                sync_dist=True,
            )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # EMA loader support (kept)
        ema = (
            self.val_loader_names[dataloader_idx].endswith("/ema")
            and getattr(self.optimizers().optimizer, "stepped", False)
        )
        if ema:
            self.optimizers().swap_ema()

        loss = self._shared_step(batch, batch_idx, prefix=self.val_loader_names[dataloader_idx])

        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Standard: log metrics like validation/test
        # Here we use "test/<idx>" naming via dataloader names prepared in test_dataloader()
        prefix = self.test_loader_names[dataloader_idx] if hasattr(self, "test_loader_names") else "test"
        return self._shared_step(batch, batch_idx, prefix=prefix)

    # -------------------------
    # Optimizer / scheduler
    # -------------------------
    def configure_optimizers(self):
        # optional param grouping hooks
        if "optimizer_param_grouping" in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        # remove _name_ for add_param_group below (avoid leaking)
        if hasattr(self.hparams.optimizer, "_name_"):
            del self.hparams.optimizer._name_

        # add special param groups
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]

        for hp in hps:
            group_params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group({"params": group_params, **self.hparams.optimizer, **hp})

        # layer decay
        if self.hparams.train.layer_decay.get("_name_", None) is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay["_name_"],
                partial=True,
            )

            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                layer_id = get_num_layer(name)
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        "params": [],
                        "lr": None,
                        "weight_decay": self.hparams.optimizer.weight_decay,
                    }
                layer_wise_groups[layer_id]["params"].append(p)
                num_max_layers = max(num_max_layers, layer_id)

            for layer_id, group in layer_wise_groups.items():
                group["lr"] = self.hparams.optimizer.lr * (
                    self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id)
                )

            optimizer.param_groups = []
            for _, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        keys = set([k for hp in hps for k in hp.keys()])
        utils.train.log_optimizer(log, optimizer, keys)

        if "scheduler" not in self.hparams:
            return optimizer

        lr_scheduler = utils.instantiate(registry.scheduler, self.hparams.scheduler, optimizer)
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler]

    # -------------------------
    # Dataloaders
    # -------------------------
    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix: str):
        if utils.is_dict(loaders):
            return [f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()], list(loaders.values())
        if utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        return [prefix], [loaders]

    def _eval_dataloaders(self):
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)

        val_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_names, test_loaders = self._eval_dataloaders_names(test_loaders, "test")

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_names += [name + "/ema" for name in val_names]
            val_loaders = val_loaders + val_loaders
            test_names += [name + "/ema" for name in test_names]
            test_loaders = test_loaders + test_loaders

        if self.hparams.train.get("remove_test_loader_in_eval", False):
            return val_names, val_loaders
        if self.hparams.train.get("remove_val_loader_in_eval", False):
            return test_names, test_loaders

        return val_names + test_names, val_loaders + test_loaders

    def val_dataloader(self):
        names, loaders = self._eval_dataloaders()
        self.val_loader_names = names
        return loaders

    def test_dataloader(self):
        # Use test loaders only (like original naming final/test/...)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        test_names, test_loaders = self._eval_dataloaders_names(test_loaders, "test")
        self.test_loader_names = ["final/" + name for name in test_names]
        return test_loaders


# =============================================================================
# Trainer creation
# =============================================================================
def create_trainer(config: DictConfig, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB logger
    if config.get("wandb") is not None:
        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Callbacks via registry
    if config.get("callbacks") is not None:
        for _name_, cb_cfg in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            if _name_ not in registry.callbacks:
                raise KeyError(f"Callback '{_name_}' not found in registry.callbacks. Available: {list(registry.callbacks.keys())}")
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            cb_cfg._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, cb_cfg))

    # ProgressiveResizing info (optional)
    if config.get("callbacks") is not None and config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        print(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Auto-ddp strategy if multi-device and strategy not set
    n_devices = config.trainer.get("devices", 1)
    if isinstance(n_devices, Sequence) and not isinstance(n_devices, (str, bytes)):
        n_devices = len(n_devices)

    if n_devices > 1 and config.trainer.get("strategy", None) is None:
        config.trainer.strategy = dict(
            _target_="pytorch_lightning.strategies.DDPStrategy",
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    # Instantiate trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")

    # Special processing for seqlen warmup reload (optional)
    if config.get("callbacks") is not None and config.callbacks.get("seqlen_warmup_reload", None) is not None:
        trainer_config_dict = dict(config.trainer)
        epochs_cume = 0
        accumulate_grad_schedule = {}

        for stage in config.callbacks.seqlen_warmup_reload.stage_params:
            batch_size = stage["batch_size"]
            grad_accum_factor = config.train.global_batch_size // batch_size
            accumulate_grad_schedule[epochs_cume] = grad_accum_factor
            epochs_cume += stage["epochs"]

        trainer_config_dict["accumulate_grad_batches"] = accumulate_grad_schedule
        trainer_config_dict.pop("_target_", None)

        # strategy must be an object for pl.Trainer(**dict)
        if "strategy" in trainer_config_dict:
            trainer_config_dict.pop("strategy", None)

        trainer_config_dict["strategy"] = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        trainer = pl.Trainer(**trainer_config_dict, callbacks=callbacks, logger=logger)
    else:
        trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    return trainer


# =============================================================================
# Train entry
# =============================================================================
def train(config: DictConfig):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)

    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    # Load pretrained model if specified
    if config.train.get("pretrained_model_path", None) is not None:
        model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )

    # Optional validation at start
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(model)

    # Fit / resume
    if config.train.ckpt is not None:
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)

    if config.train.test:
        trainer.test(model)


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: DictConfig):
    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)
    train(config)


if __name__ == "__main__":
    main()