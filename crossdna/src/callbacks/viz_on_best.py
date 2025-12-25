# src/callbacks/viz_on_best.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import torch
import pytorch_lightning as pl

from src.tasks.decoders import SequenceDecoder, VizDumpCfg  

# class VizOnBestCallback(pl.Callback):
#     """
#     在 val 上 monitor 指标刷新最佳时，跑一遍可视化导出：
#       1) 只在 global_zero 上运行；
#       2) 用当前数据集的 val_dataloader 做前向（no_grad, eval 模式）；
#       3) 逐 batch 给 SequenceDecoder 喂 labels/masks 上下文，正常 forward 即可；
#       4) 写出 .npz 文件到 out_dir。
#     """

#     def __init__(self,
#                  monitor: str = "val/mcc",
#                  mode: str = "max",
#                  out_dir: str = "viz",
#                  filename_pattern: str = "tokens_epoch{epoch:03d}_step{step}.npz",
#                  limit_batches: Optional[int] = None,
#                  stride: int = 4,
#                  max_per_class: int = 40000,
#                  project_to_common_dim: Optional[int] = None):
#         super().__init__()
#         self.monitor = monitor
#         self.mode = mode
#         self.out_dir = out_dir
#         self.filename_pattern = filename_pattern
#         self.limit_batches = limit_batches
#         self.cfg = VizDumpCfg(stride=stride, max_per_class=max_per_class,
#                               project_to_common_dim=project_to_common_dim)

#         self._best = float("-inf") if mode == "max" else float("+inf")

#     def _is_better(self, cur):
#         if cur is None: return False
#         if self.mode == "max": return cur > self._best
#         else: return cur < self._best

#     def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
#         # 只在主进程
#         if not trainer.is_global_zero:
#             return

#         # 取监控指标（例如 'val/mcc'）
#         monitor_key = getattr(pl_module.hparams.train, "monitor", self.monitor)
#         cur = trainer.callback_metrics.get(monitor_key, None)
#         if cur is None:
#             return
#         if isinstance(cur, torch.Tensor):
#             cur = cur.item()

#         if not self._is_better(cur):
#             return

#         # 更新最佳
#         self._best = cur

#         # 找到真正的 SequenceDecoder 实例（pl_module.decoder 可能是 PassthroughSequential）
#         seq_dec = None
#         for m in pl_module.decoder.modules():
#             if isinstance(m, SequenceDecoder):
#                 seq_dec = m
#                 break
#         if seq_dec is None:
#             print("[VizOnBest] No SequenceDecoder found; skip.")
#             return

#         # 输出路径
#         out_dir = Path(self.out_dir)
#         out_dir.mkdir(parents=True, exist_ok=True)
#         out_path = out_dir / self.filename_pattern.format(
#             epoch=trainer.current_epoch, step=trainer.global_step
#         )

#         # 开启记录
#         seq_dec.viz_begin(str(out_path), cfg=self.cfg)

#         # 用一个“可视化评估 loader”
#         # 这里就用当前数据集的 val_dataloader（和 eval 相同），避免泄漏训练集
#         viz_loaders = pl_module.dataset.val_dataloader(**pl_module.hparams.loader)
#         # 允许 dataset 返回 dict/list；统一封装成列表
#         if not isinstance(viz_loaders, (list, tuple)):
#             viz_loaders = [viz_loaders]
#         # 只用第一个 val loader（非 ema）
#         viz_loader = viz_loaders[0]

#         was_training = pl_module.training
#         pl_module.eval()
#         device = pl_module.device

#         # 逐 batch 前向，设置上下文
#         with torch.no_grad():
#             for bidx, batch in enumerate(viz_loader):
#                 if (self.limit_batches is not None) and (bidx >= self.limit_batches):
#                     break

#                 # 尝试从 batch 里拿 labels / attention_mask / special_tokens_mask
#                 labels, attn_mask, sp_mask = self._extract_ctx(batch, device)

#                 # 把上下文交给 decoder
#                 seq_dec.set_viz_context(labels=labels, attention_mask=attn_mask, special_tokens_mask=sp_mask)

#                 # 正常 forward 一次（会触发 SequenceDecoder 内部记录）
#                 _ = pl_module.forward(batch)

#         # 写文件
#         seq_dec.viz_finalize(model_name=pl_module.__class__.__name__,
#                              dataset_name=getattr(pl_module.hparams.dataset, "_name_", "dataset"),
#                              epoch=trainer.current_epoch)

#         if was_training:
#             pl_module.train()

#         print(f"[VizOnBest] dumped: {out_path} (monitor={monitor_key}, best={self._best:.6f})")

#     @staticmethod
#     def _extract_ctx(batch, device):
#         """
#         尽量通用地从 batch 里拿：
#           labels: [B]
#           attention_mask: [B,L]
#           special_tokens_mask: [B,L] or None
#         适配你现在 SequenceDataset 的常见三元组 (x, y, zdict)。
#         """
#         labels = None
#         attn = None
#         sp = None

#         if isinstance(batch, (list, tuple)) and len(batch) >= 2:
#             x = batch[0]
#             y = batch[1]
#             z = batch[2] if (len(batch) >= 3 and isinstance(batch[2], dict)) else {}

#             # labels
#             labels = (y if isinstance(y, torch.Tensor) else torch.as_tensor(y))
#             if labels.dim() > 1:  # [B,1] -> [B]
#                 labels = labels.view(labels.size(0))
#             labels = labels.to(device)

#             # attention mask
#             if isinstance(z, dict):
#                 if "attention_mask" in z:
#                     attn = z["attention_mask"]
#                 elif "mask" in z:
#                     attn = z["mask"]
#             if attn is None:
#                 # fallback: 全 1（需要 L），从 x 推断
#                 if isinstance(x, torch.Tensor):
#                     B, L = x.shape[0], x.shape[1]
#                 elif isinstance(x, (list, tuple)):
#                     B, L = x[0].shape[0], x[0].shape[1]
#                 else:
#                     raise RuntimeError("Cannot infer sequence length for attention_mask.")
#                 attn = torch.ones((B, L), dtype=torch.long)
#             attn = (attn if isinstance(attn, torch.Tensor) else torch.as_tensor(attn)).to(device)

#             # special_tokens_mask
#             if isinstance(z, dict) and "special_tokens_mask" in z:
#                 sp = z["special_tokens_mask"]
#                 sp = (sp if isinstance(sp, torch.Tensor) else torch.as_tensor(sp)).to(device)
#             else:
#                 sp = None

#         elif isinstance(batch, dict):
#             # 如果你的 DataLoader 直接返回 dict
#             labels = (batch.get("labels"))
#             labels = (labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)).to(device)
#             attn = batch.get("attention_mask")
#             if attn is None:
#                 x = batch.get("input_ids")
#                 B, L = x.shape[:2]
#                 attn = torch.ones((B, L), dtype=torch.long, device=device)
#             else:
#                 attn = (attn if isinstance(attn, torch.Tensor) else torch.as_tensor(attn)).to(device)
#             sp = batch.get("special_tokens_mask", None)
#             if sp is not None and not isinstance(sp, torch.Tensor):
#                 sp = torch.as_tensor(sp).to(device)
#         else:
#             raise RuntimeError("Unsupported batch format for visualization context.")

#         return labels, attn, sp


class VizOnBestCallback(pl.Callback):
    def __init__(self,
                 monitor: str = "val/mcc",
                 mode: str = "max",
                 out_dir: str = "viz",
                 filename_pattern: str = "tokens_epoch{epoch:03d}_step{step}.npz",
                 limit_batches: Optional[int] = None,
                 stride: int = 4,
                 max_per_class: int = 40000,
                 project_to_common_dim: Optional[int] = None):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.out_dir = out_dir
        self.filename_pattern = filename_pattern
        self.limit_batches = limit_batches
        self.cfg = VizDumpCfg(stride=stride, max_per_class=max_per_class,
                              project_to_common_dim=project_to_common_dim)
        self._best = float("-inf") if mode == "max" else float("+inf")

    def _is_better(self, cur):
        if cur is None: return False
        return (cur > self._best) if self.mode == "max" else (cur < self._best)

    # --- 新增：递归搬运 batch 到设备 ---
    def _to_device(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        elif isinstance(obj, dict):
            return {k: self._to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(x, device) for x in obj)
        else:
            return obj

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # --- 新增：sanity check 阶段直接跳过 ---
        if trainer.sanity_checking:
            return

        if not trainer.is_global_zero:
            return

        monitor_key = getattr(pl_module.hparams.train, "monitor", self.monitor)
        cur = trainer.callback_metrics.get(monitor_key, None)
        if isinstance(cur, torch.Tensor):
            cur = cur.item()
        if not self._is_better(cur):
            return
        self._best = cur

        # 找到 SequenceDecoder
        seq_dec = None
        for m in pl_module.decoder.modules():
            if isinstance(m, SequenceDecoder):
                seq_dec = m
                break
        if seq_dec is None:
            print("[VizOnBest] No SequenceDecoder found; skip.")
            return

        out_dir = Path(self.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.filename_pattern.format(
            epoch=trainer.current_epoch, step=trainer.global_step
        )

        seq_dec.viz_begin(str(out_path), cfg=self.cfg)

        viz_loaders = pl_module.dataset.val_dataloader(**pl_module.hparams.loader)
        if not isinstance(viz_loaders, (list, tuple)):
            viz_loaders = [viz_loaders]
        viz_loader = viz_loaders[0]

        was_training = pl_module.training
        pl_module.eval()
        device = pl_module.device

        with torch.no_grad():
            for bidx, batch in enumerate(viz_loader):
                if (self.limit_batches is not None) and (bidx >= self.limit_batches):
                    break

                # --- 关键：把整个 batch 搬到与模型相同的 device ---
                batch = self._to_device(batch, device)

                # 从 batch 提取上下文（labels/masks 已经在 device 上）
                labels, attn_mask, sp_mask = self._extract_ctx(batch, device)
                seq_dec.set_viz_context(labels=labels, attention_mask=attn_mask, special_tokens_mask=sp_mask)

                # 正常 forward（内部会记录 token 特征）
                _ = pl_module.forward(batch)

        seq_dec.viz_finalize(model_name=pl_module.__class__.__name__,
                             dataset_name=getattr(pl_module.hparams.dataset, "_name_", "dataset"),
                             epoch=trainer.current_epoch)
        if was_training:
            pl_module.train()
        print(f"[VizOnBest] dumped: {out_path} (monitor={monitor_key}, best={self._best:.6f})")

    @staticmethod
    def _extract_ctx(batch, device):
        labels = None; attn = None; sp = None

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0]; y = batch[1]
            z = batch[2] if (len(batch) >= 3 and isinstance(batch[2], dict)) else {}
            labels = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, device=device)
            if labels.dim() > 1:
                labels = labels.view(labels.size(0))
            # 注意：这里的 attn/sp 已经在 _to_device 里搬过了
            attn = z.get("attention_mask", z.get("mask", None))
            if attn is None:
                B, L = (x.shape[0], x.shape[1]) if isinstance(x, torch.Tensor) else (x[0].shape[0], x[0].shape[1])
                attn = torch.ones((B, L), dtype=torch.long, device=device)
            sp = z.get("special_tokens_mask", None)

        elif isinstance(batch, dict):
            labels = batch["labels"]
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, device=device)
            attn = batch.get("attention_mask", None)
            if attn is None:
                x = batch["input_ids"]
                B, L = x.shape[:2]
                attn = torch.ones((B, L), dtype=torch.long, device=device)
            sp = batch.get("special_tokens_mask", None)
        else:
            raise RuntimeError("Unsupported batch format for visualization context.")

        return labels, attn, sp