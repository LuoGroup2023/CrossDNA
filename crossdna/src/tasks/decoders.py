from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

import src.models.nn.utils as U
import src.utils as utils
import src.utils.config
import src.utils.train

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
log = src.utils.train.get_logger(__name__)

@dataclass
class VizDumpCfg:
    stride: int = 4
    max_per_class: int = 40000
    save_fp16: bool = True
    seed: int = 42
    project_to_common_dim: Optional[int] = None
    project_seed: int = 12345


class Decoder(nn.Module):
    """This class doesn't do much but just signals the interface that Decoders are expected to adhere to
    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset

        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)


# class SequenceDecoder(Decoder):
#     def __init__(
#         self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last"
#     ):
#         super().__init__()

#         # print('d_model:', d_model)
#         # print('d_output:', d_output)

#         self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

#         if l_output is None:
#             self.l_output = None
#             self.squeeze = False
#         elif l_output == 0:
#             # Equivalent to getting an output of length 1 and then squeezing
#             self.l_output = 1
#             self.squeeze = True
#         else:
#             assert l_output > 0
#             self.l_output = l_output
#             self.squeeze = False

#         self.use_lengths = use_lengths
#         self.mode = mode

#         if mode == 'ragged':
#             assert not use_lengths

#     def forward(self, x, state=None, lengths=None, l_output=None, mask=None):
#         """
#         x: (n_batch, l_seq, d_model)
#         Returns: (n_batch, l_output, d_output)
#         """

#         if self.l_output is None:
#             if l_output is not None:
#                 assert isinstance(l_output, int)  # Override by pass in
#             else:
#                 # Grab entire output
#                 l_output = x.size(-2)
#             squeeze = False
#         else:
#             l_output = self.l_output
#             squeeze = self.squeeze

#         if self.mode == "last":
#             restrict = lambda x: x[..., -l_output:, :]
#         elif self.mode == "first":
#             restrict = lambda x: x[..., :l_output, :]
#         elif self.mode == "pool":
#             if mask is None:
#                 restrict = lambda x: (
#                     torch.cumsum(x, dim=-2)
#                     / torch.arange(
#                         1, 1 + x.size(-2), device=x.device, dtype=x.dtype
#                     ).unsqueeze(-1)
#                 )[..., -l_output:, :]           
#             else:
#                 # sum masks
#                 mask_sums = torch.sum(mask, dim=-1).squeeze() - 1  # for 0 indexing

#                 # convert mask_sums to dtype int
#                 mask_sums = mask_sums.type(torch.int64)

#                 restrict = lambda x: (
#                     torch.cumsum(x, dim=-2)
#                     / torch.arange(
#                         1, 1 + x.size(-2), device=x.device, dtype=x.dtype
#                     ).unsqueeze(-1)
#                 )[torch.arange(x.size(0)), mask_sums, :].unsqueeze(1)  # need to keep original shape

#         elif self.mode == "sum":
#             restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
#             # TODO use same restrict function as pool case
#         elif self.mode == 'ragged':
#             assert lengths is not None, "lengths must be provided for ragged mode"
#             # remove any additional padding (beyond max length of any sequence in the batch)
#             restrict = lambda x: x[..., : max(lengths), :]
#         else:
#             raise NotImplementedError(
#                 "Mode must be ['last' | 'first' | 'pool' | 'sum']"
#             )

#         # Restrict to actual length of sequence
#         if self.use_lengths:
#             assert lengths is not None
#             x = torch.stack(
#                 [
#                     restrict(out[..., :length, :])
#                     for out, length in zip(torch.unbind(x, dim=0), lengths)
#                 ],
#                 dim=0,
#             )
#         else:
#             x = restrict(x)

#         if squeeze:
#             assert x.size(-2) == 1
#             x = x.squeeze(-2)

#         x = self.output_transform(x)

#         return x

#     def step(self, x, state=None):
#         # Ignore all length logic
#         return self.output_transform(x)


class SequenceDecoder(Decoder):
    def __init__(self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last"):
        super().__init__()
        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)
        if l_output is None:
            self.l_output, self.squeeze = None, False
        elif l_output == 0:
            self.l_output, self.squeeze = 1, True
        else:
            assert l_output > 0
            self.l_output, self.squeeze = l_output, False
        self.use_lengths = use_lengths
        self.mode = mode
        if mode == 'ragged':
            assert not use_lengths

        # ---- 可视化运行态 ----
        self._viz_active: bool = False
        self._viz_cfg: Optional[VizDumpCfg] = None
        self._viz_rng: Optional[np.random.Generator] = None
        self._viz_out_path: Optional[str] = None
        self._viz_ctx_labels: Optional[torch.Tensor] = None          # [B]
        self._viz_ctx_attention_mask: Optional[torch.Tensor] = None  # [B,L]
        self._viz_ctx_special_tokens_mask: Optional[torch.Tensor] = None  # [B,L] or None
        self._viz_reservoir_X: Dict[int, List[np.ndarray]] = {}
        self._viz_seen: Dict[int, int] = {}
        self._viz_proj_R: Optional[np.ndarray] = None  # [D, D_common]

    # ---------- 可视化控制 API ----------
    def viz_begin(self, out_npz_path: str, cfg: VizDumpCfg = VizDumpCfg()):
        self._viz_active = True
        self._viz_cfg = cfg
        self._viz_rng = np.random.default_rng(cfg.seed)
        self._viz_out_path = out_npz_path
        self._viz_reservoir_X.clear()
        self._viz_seen.clear()
        self._viz_proj_R = None
        self._viz_ctx_labels = None
        self._viz_ctx_attention_mask = None
        self._viz_ctx_special_tokens_mask = None

    def set_viz_context(self, *, labels: torch.Tensor, attention_mask: torch.Tensor,
                        special_tokens_mask: Optional[torch.Tensor] = None):
        if not self._viz_active:
            return
        self._viz_ctx_labels = labels
        self._viz_ctx_attention_mask = attention_mask
        self._viz_ctx_special_tokens_mask = special_tokens_mask

    @torch.no_grad()
    def viz_finalize(self, *, model_name: Optional[str] = None,
                     dataset_name: Optional[str] = None, epoch: Optional[int] = None,
                     class_names: Optional[Dict[int, str]] = None):
        if not self._viz_active:
            return
        assert self._viz_out_path is not None and self._viz_cfg is not None
        Xs, Ys = [], []
        for c, parts in self._viz_reservoir_X.items():
            if not parts: continue
            arr = np.concatenate(parts, axis=0)
            Xs.append(arr)
            Ys.append(np.full(len(arr), int(c), dtype=np.int32))
        if not Xs:
            raise RuntimeError("[viz_finalize] No tokens collected.")
        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        X_save = X.astype(np.float16) if self._viz_cfg.save_fp16 else X.astype(np.float32)
        meta = {
            "seed": self._viz_cfg.seed,
            "stride": self._viz_cfg.stride,
            "max_per_class": self._viz_cfg.max_per_class,
            "save_fp16": self._viz_cfg.save_fp16,
            "project_to_common_dim": self._viz_cfg.project_to_common_dim,
            "project_seed": self._viz_cfg.project_seed,
            "model_name": model_name, "dataset_name": dataset_name, "epoch": epoch,
            "decoder_mode_runtime": self.mode, "l_output_runtime": self.l_output,
            "class_names": class_names or {},
        }
        np.savez_compressed(self._viz_out_path, X=X_save, Y=Y, meta=meta)
        print(f"[viz_finalize] saved: {self._viz_out_path} | X={X_save.shape} Y={Y.shape}")
        # reset
        self._viz_active = False
        self._viz_cfg = None
        self._viz_rng = None
        self._viz_out_path = None
        self._viz_reservoir_X.clear()
        self._viz_seen.clear()
        self._viz_proj_R = None
        self._viz_ctx_labels = None
        self._viz_ctx_attention_mask = None
        self._viz_ctx_special_tokens_mask = None

    # ---------- forward（原逻辑 + 可视化记录） ----------
    def forward(self, x, state=None, lengths=None, l_output=None, mask=None):
        # 保留 restrict 前的全长 token 表征
        x_full = x

        if self.l_output is None:
            if l_output is not None: assert isinstance(l_output, int)
            else: l_output = x.size(-2)
            squeeze = False
        else:
            l_output, squeeze = self.l_output, self.squeeze

        if self.mode == "last":
            restrict = lambda t: t[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda t: t[..., :l_output, :]
        elif self.mode == "pool":
            if mask is None:
                restrict = lambda t: (torch.cumsum(t, dim=-2) /
                    torch.arange(1, 1 + t.size(-2), device=t.device, dtype=t.dtype).unsqueeze(-1)
                )[..., -l_output:, :]
            else:
                mask_sums = torch.sum(mask, dim=-1).squeeze() - 1
                mask_sums = mask_sums.type(torch.int64)
                restrict = lambda tt: (
                    torch.cumsum(tt, dim=-2) /
                    torch.arange(1, 1 + tt.size(-2), device=tt.device, dtype=tt.dtype).unsqueeze(-1)
                )[torch.arange(tt.size(0)), mask_sums, :].unsqueeze(1)
        elif self.mode == "sum":
            restrict = lambda t: torch.cumsum(t, dim=-2)[..., -l_output:, :]
        elif self.mode == 'ragged':
            assert lengths is not None
            restrict = lambda t: t[..., : max(lengths), :]
        else:
            raise NotImplementedError

        if self.use_lengths:
            assert lengths is not None
            x_out = torch.stack([restrict(out[..., :length, :]) for out, length in zip(torch.unbind(x, dim=0), lengths)], dim=0)
        else:
            x_out = restrict(x)

        # 记录 token 特征（不影响 x_out）
        if self._viz_active:
            self._maybe_record_tokens_for_viz(x_full, attn_mask=mask)

        if squeeze:
            assert x_out.size(-2) == 1
            x_out = x_out.squeeze(-2)

        x_out = self.output_transform(x_out)
        return x_out

    def step(self, x, state=None):
        return self.output_transform(x)

    @torch.no_grad()
    def _maybe_record_tokens_for_viz(self, x_full: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        if not self._viz_active or self._viz_cfg is None: return
        if self._viz_ctx_labels is None or self._viz_ctx_attention_mask is None: return

        B, L, D = x_full.shape
        am = self._viz_ctx_attention_mask.to(x_full.device)  # [B,L]
        spm = self._viz_ctx_special_tokens_mask
        spm = (spm.to(x_full.device) if spm is not None else torch.zeros_like(am, device=x_full.device))

        valid = (am * (1 - spm[:, :am.shape[-1]])) != 0
        if self._viz_cfg.stride > 1:
            keep = (torch.arange(L, device=x_full.device) % self._viz_cfg.stride == 0)
            valid = valid & keep.unsqueeze(0)

        Xv = x_full[valid].detach().float().cpu().numpy()  # [N,D]
        y_seq = self._viz_ctx_labels.to(x_full.device)     # [B]
        Yv = (y_seq.unsqueeze(1).expand(B, L))[valid].cpu().numpy().astype(np.int32)

        if self._viz_cfg.project_to_common_dim is not None:
            D_common = int(self._viz_cfg.project_to_common_dim)
            if self._viz_proj_R is None:
                rng = np.random.default_rng(self._viz_cfg.project_seed)
                self._viz_proj_R = rng.standard_normal((D, D_common)).astype(np.float32)
            Xv = (Xv.astype(np.float32) @ self._viz_proj_R)

        maxK = self._viz_cfg.max_per_class
        rng = self._viz_rng
        for c in np.unique(Yv):
            m = (Yv == c)
            if not np.any(m): continue
            feats_c = Xv[m]
            Nc = feats_c.shape[0]
            if c not in self._viz_reservoir_X:
                self._viz_reservoir_X[c] = []
                self._viz_seen[c] = 0
            for i in range(Nc):
                self._viz_seen[c] += 1
                seen = self._viz_seen[c]
                buf = self._viz_reservoir_X[c]
                if len(buf) < maxK:
                    buf.append(feats_c[i:i+1])
                else:
                    j = int(rng.integers(0, seen))
                    if j < maxK:
                        buf[j] = feats_c[i:i+1]

class TokenDecoder(Decoder):
    """Decoder for token level classification"""
    def __init__(
        self, d_model, d_output=3
    ):
        super().__init__()

        self.output_transform = nn.Linear(d_model, d_output)

    def forward(self, x, state=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """
        x = self.output_transform(x)
        return x


class NDDecoder(Decoder):
    """Decoder for single target (e.g. classification or regression)"""
    def __init__(
        self, d_model, d_output=None, mode="pool"
    ):
        super().__init__()

        assert mode in ["pool", "full"]
        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        self.mode = mode

    def forward(self, x, state=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.mode == 'pool':
            x = reduce(x, 'b ... h -> b h', 'mean')
        x = self.output_transform(x)
        return x

class StateDecoder(Decoder):
    """Use the output state to decode (useful for stateful models such as RNNs or perhaps Transformer-XL if it gets implemented"""

    def __init__(self, d_model, state_to_tensor, d_output):
        super().__init__()
        self.output_transform = nn.Linear(d_model, d_output)
        self.state_transform = state_to_tensor

    def forward(self, x, state=None):
        return self.output_transform(self.state_transform(state))


class RetrievalHead(nn.Module):
    def __init__(self, d_input, d_model, n_classes, nli=True, activation="relu"):
        super().__init__()
        self.nli = nli

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise NotImplementedError

        if (
            self.nli
        ):  # Architecture from https://github.com/mlpen/Nystromformer/blob/6539b895fa5f798ea0509d19f336d4be787b5708/reorganized_code/LRA/model_wrapper.py#L74
            self.classifier = nn.Sequential(
                nn.Linear(4 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, n_classes),
            )
        else:  # Head from https://github.com/google-research/long-range-arena/blob/ad0ff01a5b3492ade621553a1caae383b347e0c1/lra_benchmarks/models/layers/common_layers.py#L232
            self.classifier = nn.Sequential(
                nn.Linear(2 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, d_model // 2),
                activation_fn,
                nn.Linear(d_model // 2, n_classes),
            )

    def forward(self, x):
        """
        x: (2*batch, dim)
        """
        outs = rearrange(x, "(z b) d -> z b d", z=2)
        outs0, outs1 = outs[0], outs[1]  # (n_batch, d_input)
        if self.nli:
            features = torch.cat(
                [outs0, outs1, outs0 - outs1, outs0 * outs1], dim=-1
            )  # (batch, dim)
        else:
            features = torch.cat([outs0, outs1], dim=-1)  # (batch, dim)
        logits = self.classifier(features)
        return logits


class RetrievalDecoder(Decoder):
    """Combines the standard FeatureDecoder to extract a feature before passing through the RetrievalHead"""

    def __init__(
        self,
        d_input,
        n_classes,
        d_model=None,
        nli=True,
        activation="relu",
        *args,
        **kwargs
    ):
        super().__init__()
        if d_model is None:
            d_model = d_input
        self.feature = SequenceDecoder(
            d_input, d_output=None, l_output=0, *args, **kwargs
        )
        self.retrieval = RetrievalHead(
            d_input, d_model, n_classes, nli=nli, activation=activation
        )

    def forward(self, x, state=None, **kwargs):
        x = self.feature(x, state=state, **kwargs)
        x = self.retrieval(x)
        return x

class PackedDecoder(Decoder):
    def forward(self, x, state=None):
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Decoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "sequence": SequenceDecoder,
    "nd": NDDecoder,
    "retrieval": RetrievalDecoder,
    "state": StateDecoder,
    "pack": PackedDecoder,
    "token": TokenDecoder,
}
model_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
    "token": ["d_output"],
}

dataset_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output", "l_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_output"],
    "forecast": ["d_output", "l_output"],
    "token": ["d_output"],
}


def _instantiate(decoder, model=None, dataset=None):
    """Instantiate a single decoder"""
    if decoder is None:
        return None

    if isinstance(decoder, str):
        name = decoder
    else:
        name = decoder["_name_"]

    # Extract arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    # Instantiate decoder
    obj = utils.instantiate(registry, decoder, *model_args, *dataset_args)
    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
