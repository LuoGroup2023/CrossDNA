# -*- coding: utf-8 -*-
import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from contextlib import contextmanager
from collections import namedtuple
from typing import Dict, Optional, Tuple, Any

import torch.utils.checkpoint as cp  # 引入梯度检查点工具

# [速度优化] 全局开启 TF32。能让运行在 FP32 下的算子享受接近半精度的计算速度
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from fla.layers import comba
from fla.layers.attn import Attention
from fla.modules import GatedMLP as SambaMLP
from fla.modules import RMSNorm


# ========================
# OmegaConf helpers (optional)
# ========================
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None


def _to_plain_container(x: Any) -> Any:
    """Convert OmegaConf containers to plain python containers; otherwise return x."""
    if OmegaConf is not None:
        try:
            if OmegaConf.is_config(x):
                return OmegaConf.to_container(x, resolve=True)
        except Exception:
            pass
    return x


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Get value from cfg that may be an object(attr), dict, or DictConfig."""
    if cfg is None:
        return default

    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
    except Exception:
        pass

    try:
        if hasattr(cfg, key):
            return getattr(cfg, key)
    except Exception:
        pass

    try:
        return cfg[key]
    except Exception:
        return default


# ========================
# Utils
# ========================
def complement(seq: torch.Tensor) -> torch.Tensor:
    """
    仅支持 compact DNA ids:
        A=0, C=1, G=2, T=3, N=4
    """
    perm = torch.tensor([3, 2, 1, 0, 4], device=seq.device, dtype=torch.long)
    return perm[seq.long()].to(seq.dtype)


def reverse_complement(seq: torch.Tensor) -> torch.Tensor:
    comp = complement(seq)
    return torch.flip(comp, dims=[1])


def make_complement_perm(C=5, device=None, dtype=torch.float32):
    """
    logits / labels 空间的互补映射:
        A(0) <-> T(3)
        C(1) <-> G(2)
        G(2) <-> C(1)
        T(3) <-> A(0)
        N(4) -> N(4)
    """
    perm = torch.arange(C, device=device)
    if C >= 4:
        perm[0] = 3
        perm[1] = 2
        perm[2] = 1
        perm[3] = 0
    if C >= 5:
        perm[4] = 4

    P = torch.zeros(C, C, device=device, dtype=dtype)
    P[torch.arange(C, device=device), perm] = 1.0
    return P, perm


def ensure_finite(x: torch.Tensor, name: str):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"Non-finite values detected in {name}")
    return x


def linear_warmup_weight(step: int, warmup_steps: int, max_w: float):
    if warmup_steps <= 0:
        return max_w
    if step <= 0:
        return 0.0
    if step >= warmup_steps:
        return max_w
    return max_w * (step / warmup_steps)


def preferred_amp_dtype():
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def one_hot_float(x: torch.Tensor, num_classes: int, *, dtype: torch.dtype) -> torch.Tensor:
    """
    内存友好的 one-hot：避免 F.one_hot 先产生 int64 再 cast 的峰值。
    返回 [B,L,C] 的 float/bf16/fp16 one-hot。
    """
    B, L = x.shape
    out = torch.zeros((B, L, num_classes), device=x.device, dtype=dtype)
    out.scatter_(2, x.unsqueeze(-1), 1.0)
    return out


# ========================
# RC 一致性 & Barlow & TV
# ========================
def rc_consistency_kl(logits_A, logits_B_fwd, P, tau: float = 1.0, eps: float = 1e-6):
    zA = logits_A.float() / tau
    zB = logits_B_fwd.float() / tau
    pA = F.softmax(zA, dim=-1)
    logpA = F.log_softmax(zA, dim=-1)
    pB = F.softmax(zB, dim=-1)
    pB_comp = torch.matmul(pB, P.t()).clamp_min(eps)
    logpB_comp = pB_comp.log()
    kl = (pA * (logpA - logpB_comp)).sum(dim=-1).mean()
    return kl * (tau * tau)


def rc_consistency_bidirectional_stopgrad(logits_A, logits_B_fwd, P, tau: float = 1.5, eps: float = 1e-6):
    zA = logits_A.float() / tau
    zB = logits_B_fwd.float() / tau
    with torch.no_grad():
        pB_t = torch.matmul(F.softmax(zB, dim=-1), P.t()).clamp_min(eps)
        logpB_t = pB_t.log()
    loss_A = F.kl_div(F.log_softmax(zA, dim=-1), logpB_t, reduction="batchmean", log_target=True)
    with torch.no_grad():
        pA_t = torch.matmul(F.softmax(zA, dim=-1), P.t()).clamp_min(eps)
        logpA_t = pA_t.log()
    loss_B = F.kl_div(F.log_softmax(zB, dim=-1), logpA_t, reduction="batchmean", log_target=True)
    return 0.5 * (tau * tau) * (loss_A + loss_B)


def barlow_strand_loss(z1, z2, λ_off=0.04, λ_diag=0.04, eps=1e-3):
    B, L, H = z1.shape
    n = B * L
    z1 = z1.reshape(n, H)
    z2 = z2.reshape(n, H)

    def _std(z):
        var = z.var(dim=0, unbiased=False)
        return torch.sqrt(var + eps)

    std1, std2 = _std(z1), _std(z2)
    var_term = (F.relu(1 - std1).pow(2).mean() + F.relu(1 - std2).pow(2).mean())

    z1 = (z1 - z1.mean(0)) / (std1 + eps)
    z2 = (z2 - z2.mean(0)) / (std2 + eps)
    c = (z1.t() @ z2) / max(1, n)
    diag = torch.diagonal(c)
    off = c - torch.diag_embed(diag)
    cov = λ_diag * (1 - diag).pow(2).mean() + λ_off * off.pow(2).mean()
    return var_term + cov


def tv_mixed(h: torch.Tensor):
    d1 = h[:, 1:, :] - h[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return d1.abs().mean() + d2.pow(2).mean()


class Mlp(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=None, output_dimension=None,
                 activation=F.gelu, return_residual=False):
        super().__init__()
        self.return_residual = return_residual
        hd = hidden_dimension or input_dimension
        od = output_dimension or input_dimension
        self.linear1 = nn.Linear(input_dimension, hd)
        self.activation = activation
        self.linear2 = nn.Linear(hd, od)

    def forward(self, x: torch.Tensor):
        h = self.activation(self.linear1(x))
        y = self.linear2(h)
        return (y, x) if self.return_residual else y


def create_comba_cls(comba_kwargs=None, device=None, dtype=None):
    factory_kwargs = {}
    if device is not None:
        factory_kwargs["device"] = device
    if dtype is not None:
        factory_kwargs["dtype"] = dtype
    try:
        base_kwargs = dict(comba_kwargs or {})
        mixer_cls = partial(comba.Comba, **base_kwargs, **factory_kwargs)
    except ImportError:
        class FallbackComba(nn.Module):
            def forward(self, x, *args, **kwargs):
                return x
        mixer_cls = lambda *args, **kwargs: FallbackComba()
    return mixer_cls


class SlidingWindowAttention(nn.Module):
    """
    transformer_cfg 允许传入:
      - dict / OmegaConf(DictConfig)
      - 或者具有属性的 config 对象
    """
    def __init__(self, config: Any):
        super().__init__()
        config = _to_plain_container(config)

        hidden_size = _cfg_get(config, "hidden_size")
        norm_eps = _cfg_get(config, "norm_eps", 1e-5)
        attn_cfg = _cfg_get(config, "attn", {}) or {}
        attn_cfg = _to_plain_container(attn_cfg)

        self.mixer_norm = RMSNorm(hidden_size=hidden_size, eps=norm_eps)
        self.mixer = Attention(
            hidden_size=hidden_size,
            num_heads=_cfg_get(attn_cfg, "num_heads"),
            num_kv_heads=_cfg_get(attn_cfg, "num_kv_heads"),
            qkv_bias=_cfg_get(attn_cfg, "qkv_bias"),
            window_size=_cfg_get(attn_cfg, "window_size"),
            rope_theta=_cfg_get(attn_cfg, "rope_theta"),
            max_position_embeddings=_cfg_get(config, "max_position_embeddings"),
        )

        self.mlp_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.mlp = SambaMLP(
            hidden_size=hidden_size,
            hidden_ratio=_cfg_get(config, "hidden_ratio", 4.0),
            hidden_act=_cfg_get(config, "hidden_act", "swish"),
            fuse_swiglu=_cfg_get(config, "fuse_swiglu", True),
        )
        self.pre_scale = 1.0 / math.sqrt(2.0)

    def forward(self, hidden_states: torch.Tensor, cache_params: Optional[Any] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        residual = hidden_states
        x = self.mixer_norm(hidden_states)

        amp_dtype = preferred_amp_dtype()
        device_type = x.device.type if x.device.type in ["cuda", "cpu", "xpu"] else "cuda"
        with torch.autocast(device_type=device_type, enabled=True, dtype=amp_dtype):
            x_scaled = x * self.pre_scale
            attn_out, _, cache_params = self.mixer(hidden_states=x_scaled, past_key_values=cache_params, **kwargs)
            attn_out = attn_out / self.pre_scale

        ensure_finite(attn_out, "attention_out")
        h = residual + attn_out.to(x.dtype)

        residual = h
        x = self.mlp_norm(h)
        with torch.autocast(device_type=device_type, enabled=True, dtype=amp_dtype):
            x = self.mlp(x, **kwargs)
        h = residual + x
        ensure_finite(h, "block_output")
        return h, cache_params


class EnhancedHybridCore(nn.Module):
    def __init__(self, hidden_dim, comba_cfg, transformer_cfg, layer_idx=0, device=None, dtype=None):
        super().__init__()
        comba_cfg = _to_plain_container(comba_cfg)
        transformer_cfg = _to_plain_container(transformer_cfg)

        self.comba_cls = create_comba_cls(comba_kwargs=comba_cfg, device=device, dtype=dtype)
        try:
            self.comba = self.comba_cls(layer_idx=layer_idx)
        except TypeError:
            self.comba = self.comba_cls()

        self.transformer = SlidingWindowAttention(config=transformer_cfg)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _first(x):
        return x[0] if isinstance(x, tuple) else x

    def forward(self, x):
        # [防崩溃与内存控制] 强制转换为 FP32 规避 Triton Bug
        orig_dtype = x.dtype
        x_fp32 = x.float()
        device_type = x.device.type if x.device.type in ['cuda', 'cpu', 'xpu'] else 'cuda'

        with torch.autocast(device_type=device_type, enabled=False):
            m_out = self._first(self.comba(x_fp32))

        m_out = m_out.to(orig_dtype)
        del x_fp32

        t_out, _ = self.transformer(m_out)

        concat = torch.cat([m_out, t_out], dim=-1)
        g = torch.sigmoid(self.gate(concat))
        fused = g * t_out + (1 - g) * m_out
        y = self.out_norm(fused)
        ensure_finite(y, "EnhancedHybridCore.out")
        return y


class DeepEnhancedBranch(nn.Module):
    """
    更细粒度 activation checkpointing（可选）：
    - checkpoint_core_layers=True 时，对 core 内部按层/按段 checkpoint。
    """
    def __init__(
        self,
        hidden_dim: int,
        comba_cfg: Dict | None,
        transformer_cfg: Any,
        depth: int = 4,
        drop_path_rates=None,
        *,
        device=None,
        dtype=None,
        checkpoint_core_layers: bool = False,
        core_checkpoint_chunk_size: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        transformer_cfg = _to_plain_container(transformer_cfg)
        comba_cfg = _to_plain_container(comba_cfg)

        self.checkpoint_core_layers = bool(checkpoint_core_layers)
        self.core_checkpoint_chunk_size = int(core_checkpoint_chunk_size)

        if drop_path_rates is None:
            rates = [0.05 * (i / max(1, depth - 1)) for i in range(depth)]
        elif isinstance(drop_path_rates, (float, int)):
            rates = [float(drop_path_rates)] * depth
        else:
            drop_path_rates = _to_plain_container(drop_path_rates)
            rates = list(drop_path_rates) + [list(drop_path_rates)[-1]] * (depth - len(list(drop_path_rates)))

        for i in range(depth):
            layer_cfg = dict(transformer_cfg) if isinstance(transformer_cfg, dict) else transformer_cfg.copy()
            layer_cfg["drop_path_prob"] = rates[i]
            self.layers.append(EnhancedHybridCore(hidden_dim, comba_cfg, layer_cfg, i, device, dtype))

        self.output_norm = nn.LayerNorm(hidden_dim)

    def _run_layers(self, x: torch.Tensor, start: int, end: int):
        out = x
        for i in range(start, end):
            out = self.layers[i](out)
        return out

    def forward(self, x: torch.Tensor):
        if self.training and self.checkpoint_core_layers:
            chunk = max(1, self.core_checkpoint_chunk_size)
            for s in range(0, len(self.layers), chunk):
                e = min(s + chunk, len(self.layers))

                def _seg(inp, s=s, e=e):
                    return self._run_layers(inp, s, e)

                x = cp.checkpoint(_seg, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)

        y = self.output_norm(x)
        ensure_finite(y, "DeepEnhancedBranch.out")
        return y


class TokenBridge(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0,
                 kernel_size: int = 9, dilations=(1, 2, 4, 8, 16),
                 use_global_token: bool = True):
        super().__init__()
        h = hidden_dim
        pad = lambda d: d * (kernel_size // 2)
        self.dw_B = nn.ModuleList([nn.Conv1d(h, h, kernel_size, padding=pad(d), dilation=d, groups=h, bias=False) for d in dilations])
        self.mix_B = nn.Conv1d(h * len(dilations), h, 1)
        self.dw_A = nn.ModuleList([nn.Conv1d(h, h, kernel_size, padding=pad(d), dilation=d, groups=h, bias=False) for d in dilations])
        self.mix_A = nn.Conv1d(h * len(dilations), h, 1)
        self.proj_B2A = nn.Linear(h, h)
        self.proj_A2B = nn.Linear(h, h)
        self.use_global_token = use_global_token
        if use_global_token:
            self.glb_B2A = nn.Linear(h, h)
            self.glb_A2B = nn.Linear(h, h)
        self.gate = nn.Linear(h * 4, h * 2)
        self.dropout = nn.Dropout(dropout)
        self.normA = nn.LayerNorm(h)
        self.normB = nn.LayerNorm(h)

    @staticmethod
    def _agg(x: torch.Tensor, branches: nn.ModuleList, mix: nn.Module) -> torch.Tensor:
        xch = x.transpose(1, 2)
        ys = [conv(xch) for conv in branches]
        y = torch.cat(ys, dim=1)
        y = mix(y).transpose(1, 2).contiguous()
        return y

    def forward(self, xA: torch.Tensor, xB: torch.Tensor):
        ctxB = self._agg(xB, self.dw_B, self.mix_B)
        ctxA = self._agg(xA, self.dw_A, self.mix_A)
        locA = self.proj_B2A(xB + ctxB)
        locB = self.proj_A2B(xA + ctxA)
        if self.use_global_token:
            gB = self.glb_B2A(xB.mean(dim=1, keepdim=True))
            gA = self.glb_A2B(xA.mean(dim=1, keepdim=True))
            locA = locA + gB.expand(-1, xA.size(1), -1)
            locB = locB + gA.expand(-1, xB.size(1), -1)
        z = torch.cat([xA, xB, xA - xB, xA * xB], dim=-1)
        gA, gB = self.gate(z).chunk(2, dim=-1)
        gA = torch.sigmoid(gA)
        gB = torch.sigmoid(gB)
        yA = self.normA(xA + self.dropout(gA * locA))
        yB = self.normB(xB + self.dropout(gB * locB))
        return yA, yB


def semantic_preservation_loss(R_plus: torch.Tensor, H_S_plus: torch.Tensor,
                               λ_recon: float = 1.0, λ_local: float = 0.5, λ_global: float = 0.2):
    recon = F.mse_loss(H_S_plus, R_plus)
    if R_plus.size(1) >= 2:
        d_ref = R_plus[:, 1:] - R_plus[:, :-1]
        d_S = H_S_plus[:, 1:] - H_S_plus[:, :-1]
        local = F.mse_loss(d_S, d_ref)
    else:
        local = torch.tensor(0., device=R_plus.device)

    def gram_norm(x):
        G = torch.einsum("b i d, b j d -> b i j", x, x)
        return G / (G.norm(dim=(1, 2), keepdim=True) + 1e-6)

    glob = F.mse_loss(gram_norm(H_S_plus), gram_norm(R_plus))
    return λ_recon * recon + λ_local * local + λ_global * glob


@contextmanager
def eval_mode(*modules):
    states = [m.training for m in modules]
    try:
        for m in modules:
            if m is not None:
                m.eval()
        yield
    finally:
        for m, s in zip(modules, states):
            if m is not None:
                m.train(s)


class SSScanDNAHybridModel(nn.Module):
    """
    Streaming loss 版本：
    - 不再返回整段 fused logits；每个 chunk 直接计算 CE(sum) 并累加标量
    - checkpoint 生效：反传时重算 chunk forward，不保存激活
    - 保留原 non-streaming 路径（兼容旧 loss/metrics）

    这里严格保持 one-hot 只编码 A/C/G/T/N 五类。
    MLM / special 信息通过位置标记注入：
      - mlm_mask_embed
      - special_token_embed
    """
    def __init__(
        self,
        config: Optional[Any] = None,
        alphabet_size=5,
        d_model=128,
        block_size=2048,
        comba_cfg=None,
        transformer_cfg=None,
        depth=4,
        drop_path_rates=None,
        pretrain=False,
        for_representation=False,
        use_final_conv=False,
        use_s_scan: bool = True,
        use_mem: bool = False,
        use_rc_kl: bool = False,
        use_barlow: bool = False,
        use_tv: bool = False,
        sem_max_weight: float = 0.2,
        sem_warmup_steps: int = 3000,
        rc_max_weight: float = 0.2,
        rc_warmup_steps: int = 2000,
        rc_tau: float = 1.5,
        rc_bidirectional_stopgrad: bool = True,
        aux_ce_weight: float = 0.1,
        gate_freeze_steps: int = 1000,
        detach_gate: bool = False,
        gate_sup_weight: float = 0.005,
        gate_sup_warmup_steps: int = 500,
        gate_temp: float = 2.0,
        dropout=0.1,
        use_ema_teacher: bool = True,
        ema_decay: float = 0.999,
        auto_update_ema_in_forward: bool = True,
        use_bridge: bool = True,
        bridge_dropout: float = 0.0,
        use_checkpointing: bool = True,
        checkpoint_chunk_size: int = 2,
        checkpoint_core_layers: bool = False,
        core_checkpoint_chunk_size: int = 1,
        return_ab_logits: bool = True,
        streaming_loss: bool = True,
        streaming_report_ab: bool = True,
        **unused_kwargs,
    ):
        super().__init__()

        self.config = config
        if config is not None:
            cfg = _to_plain_container(config)

            alphabet_size = _cfg_get(cfg, "alphabet_size", alphabet_size)
            d_model = _cfg_get(cfg, "d_model", d_model)
            block_size = _cfg_get(cfg, "block_size", block_size)
            depth = _cfg_get(cfg, "depth", depth)
            drop_path_rates = _cfg_get(cfg, "drop_path_rates", drop_path_rates)

            pretrain = _cfg_get(cfg, "pretrain", pretrain)
            for_representation = _cfg_get(cfg, "for_representation", for_representation)

            use_s_scan = _cfg_get(cfg, "use_s_scan", use_s_scan)
            use_mem = _cfg_get(cfg, "use_mem", use_mem)
            use_rc_kl = _cfg_get(cfg, "use_rc_kl", use_rc_kl)
            use_barlow = _cfg_get(cfg, "use_barlow", use_barlow)
            use_tv = _cfg_get(cfg, "use_tv", use_tv)

            sem_max_weight = _cfg_get(cfg, "sem_max_weight", sem_max_weight)
            sem_warmup_steps = _cfg_get(cfg, "sem_warmup_steps", sem_warmup_steps)
            rc_max_weight = _cfg_get(cfg, "rc_max_weight", rc_max_weight)
            rc_warmup_steps = _cfg_get(cfg, "rc_warmup_steps", rc_warmup_steps)
            rc_tau = _cfg_get(cfg, "rc_tau", rc_tau)
            rc_bidirectional_stopgrad = _cfg_get(cfg, "rc_bidirectional_stopgrad", rc_bidirectional_stopgrad)

            aux_ce_weight = _cfg_get(cfg, "aux_ce_weight", aux_ce_weight)
            gate_freeze_steps = _cfg_get(cfg, "gate_freeze_steps", gate_freeze_steps)
            detach_gate = _cfg_get(cfg, "detach_gate", detach_gate)
            gate_sup_weight = _cfg_get(cfg, "gate_sup_weight", gate_sup_weight)
            gate_sup_warmup_steps = _cfg_get(cfg, "gate_sup_warmup_steps", gate_sup_warmup_steps)
            gate_temp = _cfg_get(cfg, "gate_temp", gate_temp)
            dropout = _cfg_get(cfg, "dropout", dropout)

            use_bridge = _cfg_get(cfg, "use_bridge", use_bridge)
            bridge_dropout = _cfg_get(cfg, "bridge_dropout", bridge_dropout)

            use_checkpointing = _cfg_get(cfg, "use_checkpointing", use_checkpointing)
            checkpoint_chunk_size = _cfg_get(cfg, "checkpoint_chunk_size", checkpoint_chunk_size)

            checkpoint_core_layers = _cfg_get(cfg, "checkpoint_core_layers", checkpoint_core_layers)
            core_checkpoint_chunk_size = _cfg_get(cfg, "core_checkpoint_chunk_size", core_checkpoint_chunk_size)

            return_ab_logits = _cfg_get(cfg, "return_ab_logits", return_ab_logits)

            streaming_loss = _cfg_get(cfg, "streaming_loss", streaming_loss)
            streaming_report_ab = _cfg_get(cfg, "streaming_report_ab", streaming_report_ab)

            transformer_cfg = _cfg_get(cfg, "transformer_cfg", transformer_cfg)
            comba_cfg = _cfg_get(cfg, "comba_cfg", comba_cfg)

            transformer_cfg = _to_plain_container(transformer_cfg)
            comba_cfg = _to_plain_container(comba_cfg)
            drop_path_rates = _to_plain_container(drop_path_rates)

        self.alphabet_size = int(alphabet_size)  # 严格保持 5: A/C/G/T/N
        self.pretrain = bool(pretrain)
        self.for_representation = bool(for_representation)
        self.block_size = int(block_size)
        self.use_final_conv = bool(use_final_conv)
        self.d_model = int(d_model)

        self.use_checkpointing = bool(use_checkpointing)
        self.checkpoint_chunk_size = int(checkpoint_chunk_size)

        self.checkpoint_core_layers = bool(checkpoint_core_layers)
        self.core_checkpoint_chunk_size = int(core_checkpoint_chunk_size)

        self.return_ab_logits = bool(return_ab_logits)

        self.streaming_loss = bool(streaming_loss)
        self.streaming_report_ab = bool(streaming_report_ab)

        self.register_buffer("g_step", torch.zeros(1, dtype=torch.long))

        # embedding conv（one-hot -> Conv1d）
        self.linear = nn.Conv1d(self.alphabet_size, self.d_model, kernel_size=9, padding=4)
        self.rc_linear = nn.Conv1d(self.alphabet_size, self.d_model, kernel_size=9, padding=4)

        # 位置语义注入：
        # - mlm_mask_embed: 表示“这是被 MLM 选中的位置”
        # - special_token_embed: 表示“这原本是 special/pad/unk 位置”
        self.mlm_mask_embed = nn.Parameter(torch.zeros(self.d_model))
        self.special_token_embed = nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.mlm_mask_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.special_token_embed, mean=0.0, std=0.02)

        self.branchA_core = DeepEnhancedBranch(
            hidden_dim=self.d_model,
            comba_cfg=comba_cfg,
            transformer_cfg=transformer_cfg,
            depth=int(depth),
            drop_path_rates=drop_path_rates,
            checkpoint_core_layers=self.checkpoint_core_layers,
            core_checkpoint_chunk_size=self.core_checkpoint_chunk_size,
        )
        self.branchB_core = DeepEnhancedBranch(
            hidden_dim=self.d_model,
            comba_cfg=comba_cfg,
            transformer_cfg=transformer_cfg,
            depth=int(depth),
            drop_path_rates=drop_path_rates,
            checkpoint_core_layers=self.checkpoint_core_layers,
            core_checkpoint_chunk_size=self.core_checkpoint_chunk_size,
        )

        self.use_bridge = bool(use_bridge)
        if self.use_bridge:
            self.bridge = TokenBridge(self.d_model, dropout=float(bridge_dropout))

        self.use_ema_teacher = bool(use_ema_teacher)
        self.ema_decay = float(ema_decay)
        self.auto_update_ema_in_forward = bool(auto_update_ema_in_forward)
        if self.use_ema_teacher:
            self.branchA_core_ema = copy.deepcopy(self.branchA_core)
            self.branchB_core_ema = copy.deepcopy(self.branchB_core)
            for p in self.branchA_core_ema.parameters():
                p.requires_grad_(False)
            for p in self.branchB_core_ema.parameters():
                p.requires_grad_(False)
            if self.use_bridge:
                self.bridge_ema = copy.deepcopy(self.bridge)
                for p in self.bridge_ema.parameters():
                    p.requires_grad_(False)

        self.proj_A = Mlp(self.d_model, self.d_model * 2, self.d_model, activation=F.gelu, return_residual=True)
        self.proj_B = Mlp(self.d_model, self.d_model * 2, self.d_model, activation=F.gelu, return_residual=True)
        self.gate_fuse = nn.Linear(2 * self.d_model, self.d_model)
        self.out_linear = nn.Linear(self.d_model, self.alphabet_size)  # 仍只预测 A/C/G/T/N
        self.dropout = nn.Dropout(float(dropout))

        P_comp, _ = make_complement_perm(self.alphabet_size)
        self.register_buffer("P_comp", P_comp)

        self.use_s_scan = bool(use_s_scan)
        self.use_rc_kl = bool(use_rc_kl)
        self.use_barlow = bool(use_barlow)
        self.use_tv = bool(use_tv)
        self.sem_max_weight = float(sem_max_weight)
        self.sem_warmup_steps = int(sem_warmup_steps)
        self.rc_max_weight = float(rc_max_weight)
        self.rc_warmup_steps = int(rc_warmup_steps)
        self.rc_tau = float(rc_tau)
        self.rc_bidirectional_stopgrad = bool(rc_bidirectional_stopgrad)
        self.aux_ce_weight = float(aux_ce_weight)
        self.gate_freeze_steps = int(gate_freeze_steps)
        self.detach_gate = bool(detach_gate)
        self.gate_sup_weight = float(gate_sup_weight)
        self.gate_sup_warmup_steps = int(gate_sup_warmup_steps)
        self.gate_temp = float(gate_temp)

        if self.use_final_conv:
            self.final_conv = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1)

        self._unused_init_kwargs = dict(unused_kwargs) if unused_kwargs else {}

    @torch.no_grad()
    def update_ema(self):
        if not getattr(self, "use_ema_teacher", False):
            return
        d = float(getattr(self, "ema_decay", 0.999))
        for m_ema, m in [(self.branchA_core_ema, self.branchA_core),
                         (self.branchB_core_ema, self.branchB_core)]:
            for p_ema, p in zip(m_ema.parameters(), m.parameters()):
                p_ema.data.lerp_(p.data, 1.0 - d)
        if getattr(self, "use_bridge", False) and hasattr(self, "bridge_ema"):
            for p_ema, p in zip(self.bridge_ema.parameters(), self.bridge.parameters()):
                p_ema.data.lerp_(p.data, 1.0 - d)

    # ------------------------------------------------------------------
    # Streaming chunk：直接返回 ce_sum/n_masked + 统计 + total_aux（不返回整段 logits）
    # ------------------------------------------------------------------
    def _forward_s_scan_chunk_streaming(
        self,
        X_A: torch.Tensor,
        X_B: torch.Tensor,
        A_emb_fwd: torch.Tensor,
        B_emb_rc: torch.Tensor,
        mlm_labels: torch.Tensor,        # [BC, L_blk], -100 outside masked
        chunk_start_t: torch.Tensor,
        num_blocks_t: torch.Tensor,
        step_t: torch.Tensor,
        report_ab_t: torch.Tensor,        # 0/1
    ):
        chunk_start = int(chunk_start_t.item())
        num_blocks = int(num_blocks_t.item())
        step = int(step_t.item())
        report_ab = bool(int(report_ab_t.item()))

        BC, L_blk, H = X_A.shape
        B = BC // max(1, num_blocks)
        device = X_A.device

        # -------- core forward --------
        H_A = self.branchA_core(X_A)
        H_B = self.branchB_core(X_B)

        H_A = H_A.view(num_blocks, B, L_blk, H)
        H_B = H_B.view(num_blocks, B, L_blk, H)

        for c in range(num_blocks):
            actual_t = chunk_start + c
            if (actual_t % 2) != 0:
                H_A[c] = torch.flip(H_A[c], dims=[1])
            if (actual_t % 2) == 0:
                H_B[c] = torch.flip(H_B[c], dims=[1])

        H_A = H_A.reshape(BC, L_blk, H)
        H_B = H_B.reshape(BC, L_blk, H)

        if self.use_bridge:
            H_A, H_B = self.bridge(H_A, H_B)

        fA, rA = self.proj_A(H_A)
        FA = fA + rA
        fB, rB = self.proj_B(H_B)
        FB = fB + rB

        gate_in = torch.cat([FA, FB], dim=-1)
        g_logits = self.gate_fuse(gate_in)
        g_raw = torch.sigmoid(g_logits / max(1e-6, getattr(self, "gate_temp", 1.0)))

        if step < getattr(self, "gate_freeze_steps", 0):
            g = 0.5 * torch.ones_like(g_raw)
        else:
            g = g_raw

        if getattr(self, "detach_gate", False):
            mix = g.detach() * FA + (1 - g.detach()) * FB
        else:
            mix = g * FA + (1 - g) * FB

        fused = F.layer_norm(mix, (mix.size(-1),))
        fused = ensure_finite(fused, "fused_blk")

        if self.use_final_conv:
            fused = self.final_conv(fused.permute(0, 2, 1)).permute(0, 2, 1)

        logits = self.out_linear(fused)  # [BC, L_blk, C]
        C = logits.size(-1)

        # -------- Streaming CE(sum) --------
        logits2d = logits.reshape(-1, C)
        labels1d = mlm_labels.reshape(-1)

        ce_sum = F.cross_entropy(logits2d, labels1d, ignore_index=-100, reduction="sum")

        with torch.no_grad():
            valid = (labels1d != -100)
            n_masked = valid.sum()

        # -------- fused topk stats --------
        with torch.no_grad():
            correct1 = torch.zeros([], device=device, dtype=torch.long)
            correct3 = torch.zeros([], device=device, dtype=torch.long)
            if n_masked.item() > 0:
                sel_logits = logits2d[valid]
                sel_labels = labels1d[valid]
                pred1 = sel_logits.argmax(dim=-1)
                correct1 = pred1.eq(sel_labels).sum()
                top3 = sel_logits.topk(3, dim=-1).indices
                correct3 = top3.eq(sel_labels.unsqueeze(-1)).any(dim=-1).sum()

        total_aux = torch.zeros([], device=device, dtype=torch.float32)

        # -------- aux losses (pretrain only) --------
        if self.pretrain:
            parities = torch.tensor([(chunk_start + c) % 2 == 1 for c in range(num_blocks)],
                                    device=device, dtype=torch.bool)
            maskA_row = parities.repeat_interleave(B).unsqueeze(1)     # [BC,1]
            maskA = maskA_row.expand(-1, L_blk)                        # [BC,L]
            maskB = ~maskA

            with torch.no_grad():
                teacherA = self.branchA_core_ema if self.use_ema_teacher else self.branchA_core
                teacherB = self.branchB_core_ema if self.use_ema_teacher else self.branchB_core
                tbridge = self.bridge_ema if (self.use_bridge and self.use_ema_teacher and hasattr(self, "bridge_ema")) else (
                    self.bridge if self.use_bridge else None
                )

                mods = [teacherA, teacherB] + ([tbridge] if tbridge is not None else [])
                with eval_mode(*mods):
                    R_plus_A = teacherA(A_emb_fwd)
                    R_plus_B = teacherB(A_emb_fwd)
                    if tbridge is not None:
                        R_plus_A, R_plus_B = tbridge(R_plus_A, R_plus_B)

                    R_minus_A_rc = teacherA(B_emb_rc)
                    R_minus_B_rc = teacherB(B_emb_rc)
                    R_minus_A_fwd = torch.flip(R_minus_A_rc, dims=[1])
                    R_minus_B_fwd = torch.flip(R_minus_B_rc, dims=[1])
                    if tbridge is not None:
                        R_minus_A_fwd, R_minus_B_fwd = tbridge(R_minus_A_fwd, R_minus_B_fwd)

            R_A_teacher = torch.where(maskA.unsqueeze(-1), R_minus_A_fwd, R_plus_A)
            R_B_teacher = torch.where(maskB.unsqueeze(-1), R_minus_B_fwd, R_plus_B)

            sem_A = semantic_preservation_loss(R_A_teacher.float(), FA.float())
            sem_B = semantic_preservation_loss(R_B_teacher.float(), FB.float())
            w_sem = linear_warmup_weight(step, getattr(self, "sem_warmup_steps", 0), getattr(self, "sem_max_weight", 1.0))
            total_aux = total_aux + w_sem * (sem_A + sem_B)

            if (getattr(self, "gate_sup_weight", 0.0) > 0.0) and (step >= getattr(self, "gate_freeze_steps", 0)):
                g_target = (~maskA).float().unsqueeze(-1)
                g_token_logits = g_logits.mean(dim=-1, keepdim=True) / max(1e-6, getattr(self, "gate_temp", 1.0))
                w_gate = linear_warmup_weight(
                    step - getattr(self, "gate_freeze_steps", 0),
                    getattr(self, "gate_sup_warmup_steps", 0),
                    getattr(self, "gate_sup_weight", 0.0),
                )
                total_aux = total_aux + w_gate * F.binary_cross_entropy_with_logits(g_token_logits.float(), g_target.float())

            need_rc = bool(self.use_rc_kl and getattr(self, "rc_max_weight", 0.0) > 0.0)
            need_ab = bool(report_ab or need_rc)

            logitsA = logitsB = None
            if need_ab:
                logitsA = self.out_linear(FA)
                logitsB = self.out_linear(FB)

            if need_rc:
                if getattr(self, "rc_bidirectional_stopgrad", True):
                    rc = rc_consistency_bidirectional_stopgrad(logitsA, logitsB, self.P_comp, tau=getattr(self, "rc_tau", 1.5))
                else:
                    rc = rc_consistency_kl(logitsA, logitsB, self.P_comp, tau=getattr(self, "rc_tau", 1.5))
                w_rc = linear_warmup_weight(step, getattr(self, "rc_warmup_steps", 0), getattr(self, "rc_max_weight", 0.0))
                total_aux = total_aux + w_rc * rc

            if self.use_barlow:
                total_aux = total_aux + barlow_strand_loss(H_A.float(), H_B.float())
            if self.use_tv:
                total_aux = total_aux + tv_mixed(fused.float())

        # -------- A/B topk stats（可选）--------
        with torch.no_grad():
            correctA1 = torch.zeros([], device=device, dtype=torch.long)
            correctB1 = torch.zeros([], device=device, dtype=torch.long)
            correctA3 = torch.zeros([], device=device, dtype=torch.long)
            correctB3 = torch.zeros([], device=device, dtype=torch.long)

            if report_ab and n_masked.item() > 0:
                if 'logitsA' not in locals() or logitsA is None:
                    logitsA = self.out_linear(FA)
                    logitsB = self.out_linear(FB)

                _, perm = make_complement_perm(C, device=device)

                valid = (labels1d != -100)
                labels_safe = labels1d.clamp_min(0)
                labels_comp = perm[labels_safe]

                parities = torch.tensor([(chunk_start + c) % 2 == 1 for c in range(num_blocks)],
                                        device=device, dtype=torch.bool)
                maskA_tok = parities.repeat_interleave(B).unsqueeze(1).expand(-1, L_blk).reshape(-1)
                maskB_tok = ~maskA_tok

                yA = torch.where(maskA_tok, labels_comp, labels_safe)[valid]
                yB = torch.where(maskB_tok, labels_comp, labels_safe)[valid]

                A2d = logitsA.reshape(-1, C)[valid]
                B2d = logitsB.reshape(-1, C)[valid]

                predA1 = A2d.argmax(dim=-1)
                predB1 = B2d.argmax(dim=-1)
                correctA1 = predA1.eq(yA).sum()
                correctB1 = predB1.eq(yB).sum()

                topA3 = A2d.topk(3, dim=-1).indices
                topB3 = B2d.topk(3, dim=-1).indices
                correctA3 = topA3.eq(yA.unsqueeze(-1)).any(dim=-1).sum()
                correctB3 = topB3.eq(yB.unsqueeze(-1)).any(dim=-1).sum()

        return ce_sum, n_masked, total_aux, correct1, correct3, correctA1, correctB1, correctA3, correctB3

    # ------------------------------------------------------------------
    # 原来的 non-streaming chunk（用于回退/对齐/推理）
    # ------------------------------------------------------------------
    def _forward_s_scan_chunk(
        self,
        X_A: torch.Tensor,
        X_B: torch.Tensor,
        A_emb_fwd: torch.Tensor,
        B_emb_rc: torch.Tensor,
        chunk_start_t: torch.Tensor,
        num_blocks_t: torch.Tensor,
        step_t: torch.Tensor,
        need_logits_t: torch.Tensor,
        need_ab_t: torch.Tensor,
    ):
        chunk_start = int(chunk_start_t.item())
        num_blocks = int(num_blocks_t.item())
        step = int(step_t.item())
        need_logits = bool(int(need_logits_t.item()))
        need_ab = bool(int(need_ab_t.item()))

        BC, L_blk, H = X_A.shape
        B = BC // max(1, num_blocks)
        device = X_A.device

        H_A = self.branchA_core(X_A)
        H_B = self.branchB_core(X_B)

        H_A = H_A.view(num_blocks, B, L_blk, H)
        H_B = H_B.view(num_blocks, B, L_blk, H)

        for c in range(num_blocks):
            actual_t = chunk_start + c
            if (actual_t % 2) != 0:
                H_A[c] = torch.flip(H_A[c], dims=[1])
            if (actual_t % 2) == 0:
                H_B[c] = torch.flip(H_B[c], dims=[1])

        H_A = H_A.reshape(BC, L_blk, H)
        H_B = H_B.reshape(BC, L_blk, H)

        if self.use_bridge:
            H_A, H_B = self.bridge(H_A, H_B)

        fA, rA = self.proj_A(H_A)
        FA = fA + rA
        fB, rB = self.proj_B(H_B)
        FB = fB + rB

        gate_in_blk = torch.cat([FA, FB], dim=-1)
        g_logits_blk = self.gate_fuse(gate_in_blk)
        g_raw_blk = torch.sigmoid(g_logits_blk / max(1e-6, getattr(self, "gate_temp", 1.0)))

        if step < getattr(self, "gate_freeze_steps", 0):
            g_blk = 0.5 * torch.ones_like(g_raw_blk)
        else:
            g_blk = g_raw_blk

        if getattr(self, "detach_gate", False):
            mix_blk = g_blk.detach() * FA + (1 - g_blk.detach()) * FB
        else:
            mix_blk = g_blk * FA + (1 - g_blk) * FB

        fused_blk = F.layer_norm(mix_blk, (mix_blk.size(-1),))
        fused_blk = ensure_finite(fused_blk, "fused_blk")

        if self.use_final_conv:
            fused_blk = self.final_conv(fused_blk.permute(0, 2, 1)).permute(0, 2, 1)

        logits_blk = self.out_linear(fused_blk) if need_logits else fused_blk.new_empty((0,))

        need_rc_logits = bool(self.use_rc_kl and (getattr(self, "rc_max_weight", 0.0) > 0.0))
        need_ab_internal = bool(need_ab or need_rc_logits)

        logitsA_blk = self.out_linear(FA) if need_ab_internal else fused_blk.new_empty((0,))
        logitsB_blk = self.out_linear(FB) if need_ab_internal else fused_blk.new_empty((0,))

        total_aux_blk = torch.zeros([], device=device, dtype=torch.float32)

        if self.pretrain:
            parities = torch.tensor([(chunk_start + c) % 2 == 1 for c in range(num_blocks)],
                                    device=device, dtype=torch.bool)
            maskA_row = parities.repeat_interleave(B).unsqueeze(1)
            maskA = maskA_row.expand(-1, L_blk)
            maskB = ~maskA

            with torch.no_grad():
                teacherA = self.branchA_core_ema if self.use_ema_teacher else self.branchA_core
                teacherB = self.branchB_core_ema if self.use_ema_teacher else self.branchB_core
                tbridge = self.bridge_ema if (self.use_bridge and self.use_ema_teacher and hasattr(self, "bridge_ema")) else (
                    self.bridge if self.use_bridge else None
                )

                mods = [teacherA, teacherB] + ([tbridge] if tbridge is not None else [])
                with eval_mode(*mods):
                    R_plus_A = teacherA(A_emb_fwd)
                    R_plus_B = teacherB(A_emb_fwd)
                    if tbridge is not None:
                        R_plus_A, R_plus_B = tbridge(R_plus_A, R_plus_B)

                    R_minus_A_rc = teacherA(B_emb_rc)
                    R_minus_B_rc = teacherB(B_emb_rc)
                    R_minus_A_fwd = torch.flip(R_minus_A_rc, dims=[1])
                    R_minus_B_fwd = torch.flip(R_minus_B_rc, dims=[1])
                    if tbridge is not None:
                        R_minus_A_fwd, R_minus_B_fwd = tbridge(R_minus_A_fwd, R_minus_B_fwd)

            R_A_teacher = torch.where(maskA.unsqueeze(-1), R_minus_A_fwd, R_plus_A)
            R_B_teacher = torch.where(maskB.unsqueeze(-1), R_minus_B_fwd, R_plus_B)

            sem_A = semantic_preservation_loss(R_A_teacher.float(), FA.float())
            sem_B = semantic_preservation_loss(R_B_teacher.float(), FB.float())
            w_sem = linear_warmup_weight(step, getattr(self, "sem_warmup_steps", 0), getattr(self, "sem_max_weight", 1.0))
            total_aux_blk = total_aux_blk + w_sem * (sem_A + sem_B)

            if (getattr(self, "gate_sup_weight", 0.0) > 0.0) and (step >= getattr(self, "gate_freeze_steps", 0)):
                g_target_blk = (~maskA).float().unsqueeze(-1)
                g_token_logits_blk = g_logits_blk.mean(dim=-1, keepdim=True) / max(1e-6, getattr(self, "gate_temp", 1.0))
                w_gate = linear_warmup_weight(
                    step - getattr(self, "gate_freeze_steps", 0),
                    getattr(self, "gate_sup_warmup_steps", 0),
                    getattr(self, "gate_sup_weight", 0.0),
                )
                total_aux_blk = total_aux_blk + w_gate * F.binary_cross_entropy_with_logits(
                    g_token_logits_blk.float(), g_target_blk.float()
                )

            if self.use_rc_kl and getattr(self, "rc_max_weight", 0.0) > 0:
                if logitsA_blk.numel() == 0:
                    logitsA_blk = self.out_linear(FA)
                if logitsB_blk.numel() == 0:
                    logitsB_blk = self.out_linear(FB)

                if getattr(self, "rc_bidirectional_stopgrad", True):
                    rc = rc_consistency_bidirectional_stopgrad(
                        logitsA_blk, logitsB_blk, self.P_comp, tau=getattr(self, "rc_tau", 1.5)
                    )
                else:
                    rc = rc_consistency_kl(
                        logitsA_blk, logitsB_blk, self.P_comp, tau=getattr(self, "rc_tau", 1.5)
                    )
                w_rc = linear_warmup_weight(step, getattr(self, "rc_warmup_steps", 0), getattr(self, "rc_max_weight", 0.0))
                total_aux_blk = total_aux_blk + w_rc * rc

            if self.use_barlow:
                total_aux_blk = total_aux_blk + barlow_strand_loss(H_A.float(), H_B.float())
            if self.use_tv:
                total_aux_blk = total_aux_blk + tv_mixed(fused_blk.float())

        return fused_blk, logits_blk, logitsA_blk, logitsB_blk, total_aux_blk

    def forward(self, seq, t=None, cls=None, return_embedding=False, state=None, mask=None, **kwargs):
        step = int(self.g_step.item())
        if self.training:
            self.g_step += 1
        
    # 下游分类任务传入的 padding/attention mask
    # 当前 backbone 不直接使用，仅为兼容 NT / GB 微调
        input_mask = mask

        # -----------------------------
        # batch[0] 现在是:
        #   [masked_seq, mlm_mask, mlm_labels, special_mask]
        # 兼容老格式:
        #   [masked_seq, mask, mlm_labels]
        # -----------------------------

        mlm_labels = None
        special_mask = None
        if self.pretrain:
            if isinstance(seq, (tuple, list)):
                mlm_mask = seq[1] if len(seq) >= 2 else None
                mlm_labels = seq[2] if len(seq) >= 3 else None
                special_mask = seq[3] if len(seq) >= 4 else None
                seq = seq[0]
            else:
                mlm_mask = None
                mlm_labels = None
                special_mask = None
        else:
            mlm_mask = None
            special_mask = None

        device_type = seq.device.type if seq.device.type in ['cuda', 'cpu', 'xpu'] else 'cuda'
        amp_dtype = preferred_amp_dtype()

        rc_seq = reverse_complement(seq)

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
            seq_oh = one_hot_float(seq, self.alphabet_size, dtype=amp_dtype)      # [B,L,5]
            rc_oh = one_hot_float(rc_seq, self.alphabet_size, dtype=amp_dtype)    # [B,L,5]

            # -------------------------------------------------
            # special positions:
            # 不让 [PAD]/[SEP]/[UNK] 等位置携带普通碱基 one-hot 语义
            # -------------------------------------------------
            if special_mask is not None:
                special_mask = special_mask.to(dtype=torch.bool, device=seq.device)
                rc_special_mask = torch.flip(special_mask, dims=[1])

                seq_oh = seq_oh.masked_fill(special_mask.unsqueeze(-1), 0.0)
                rc_oh = rc_oh.masked_fill(rc_special_mask.unsqueeze(-1), 0.0)
            else:
                rc_special_mask = None

            h = F.gelu(self.linear(seq_oh.permute(0, 2, 1)))      # [B,H,L]
            rc_h = F.gelu(self.rc_linear(rc_oh.permute(0, 2, 1))) # [B,H,L]
            del seq_oh, rc_oh

            # 再次把 special 位置上的普通卷积响应清掉，避免 Conv1d bias 引入伪信号
            if special_mask is not None:
                non_special = (~special_mask).to(dtype=h.dtype).unsqueeze(1)           # [B,1,L]
                rc_non_special = (~rc_special_mask).to(dtype=rc_h.dtype).unsqueeze(1)  # [B,1,L]
                h = h * non_special
                rc_h = rc_h * rc_non_special

            # -------------------------------------------------
            # MLM indicator：被 MLM 选中的位置
            # -------------------------------------------------
            if mlm_mask is not None:
                mlm_mask_f = mlm_mask.to(dtype=h.dtype, device=h.device).unsqueeze(1)  # [B,1,L]
                rc_mlm_mask_f = torch.flip(mlm_mask, dims=[1]).to(dtype=rc_h.dtype, device=rc_h.device).unsqueeze(1)

                h = h + mlm_mask_f * self.mlm_mask_embed.view(1, -1, 1)
                rc_h = rc_h + rc_mlm_mask_f * self.mlm_mask_embed.view(1, -1, 1)

            # -------------------------------------------------
            # Special indicator：原本是 special 位置
            # -------------------------------------------------
            if special_mask is not None:
                special_mask_f = special_mask.to(dtype=h.dtype, device=h.device).unsqueeze(1)
                rc_special_mask_f = rc_special_mask.to(dtype=rc_h.dtype, device=rc_h.device).unsqueeze(1)

                h = h + special_mask_f * self.special_token_embed.view(1, -1, 1)
                rc_h = rc_h + rc_special_mask_f * self.special_token_embed.view(1, -1, 1)

            # ==========================================================
            #  Streaming 路径：pretrain + s_scan + 有 mlm_labels + 非 representation
            # ==========================================================
            use_streaming = bool(
                self.pretrain
                and self.use_s_scan
                and self.streaming_loss
                and (mlm_labels is not None)
                and (not self.for_representation)
            )

            if use_streaming:
                B, H, L = h.shape
                l = self.block_size
                K = (L + l - 1) // l
                chunk_size = max(1, getattr(self, "checkpoint_chunk_size", 2))

                ce_sum_total = torch.zeros([], device=seq.device, dtype=torch.float32)
                n_total = torch.zeros([], device=seq.device, dtype=torch.long)
                total_aux = torch.zeros([], device=seq.device, dtype=torch.float32)

                correct1 = torch.zeros([], device=seq.device, dtype=torch.long)
                correct3 = torch.zeros([], device=seq.device, dtype=torch.long)
                correctA1 = torch.zeros([], device=seq.device, dtype=torch.long)
                correctB1 = torch.zeros([], device=seq.device, dtype=torch.long)
                correctA3 = torch.zeros([], device=seq.device, dtype=torch.long)
                correctB3 = torch.zeros([], device=seq.device, dtype=torch.long)

                keep_rate = mlm_mask.float().mean() if mask is not None else torch.tensor(1.0, device=seq.device)
                report_ab_t = torch.tensor(int(self.streaming_report_ab), device=seq.device)

                for chunk_start in range(0, K, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, K)

                    X_A_batch, X_B_batch = [], []
                    Aemb_batch, Bemb_batch = [], []
                    labels_batch = []
                    lengths = []

                    for t_block in range(chunk_start, chunk_end):
                        start = t_block * l
                        end = min(start + l, L)
                        blk_len = end - start
                        lengths.append(blk_len)

                        # dropout 前（teacher 复用）
                        fwd_emb = h[:, :, start:end].transpose(1, 2).contiguous()
                        rc_emb = rc_h[:, :, start:end].transpose(1, 2).contiguous()

                        # dropout 后（student）
                        fwd_in = self.dropout(h[:, :, start:end]).transpose(1, 2).contiguous()
                        rc_in = self.dropout(rc_h[:, :, start:end]).transpose(1, 2).contiguous()

                        if (t_block % 2) == 0:
                            X_A_batch.append(fwd_in)
                            X_B_batch.append(rc_in)
                        else:
                            X_A_batch.append(rc_in)
                            X_B_batch.append(fwd_in)

                        Aemb_batch.append(fwd_emb)
                        Bemb_batch.append(rc_emb)

                        labels_batch.append(mlm_labels[:, start:end])

                    if len(set(lengths)) == 1:
                        blk_len = lengths[0]
                        nb = len(X_A_batch)

                        X_A_tensor = torch.cat(X_A_batch, dim=0)
                        X_B_tensor = torch.cat(X_B_batch, dim=0)
                        Aemb_tensor = torch.cat(Aemb_batch, dim=0)
                        Bemb_tensor = torch.cat(Bemb_batch, dim=0)
                        labels_tensor = torch.cat(labels_batch, dim=0)

                        if self.training and self.use_checkpointing:
                            ce_sum, n_masked, aux_blk, c1, c3, a1, b1, a3, b3 = cp.checkpoint(
                                self._forward_s_scan_chunk_streaming,
                                X_A_tensor, X_B_tensor, Aemb_tensor, Bemb_tensor, labels_tensor,
                                torch.tensor(chunk_start, device=seq.device),
                                torch.tensor(nb, device=seq.device),
                                torch.tensor(step, device=seq.device),
                                report_ab_t,
                                use_reentrant=False
                            )
                        else:
                            ce_sum, n_masked, aux_blk, c1, c3, a1, b1, a3, b3 = self._forward_s_scan_chunk_streaming(
                                X_A_tensor, X_B_tensor, Aemb_tensor, Bemb_tensor, labels_tensor,
                                torch.tensor(chunk_start, device=seq.device),
                                torch.tensor(nb, device=seq.device),
                                torch.tensor(step, device=seq.device),
                                report_ab_t,
                            )

                        ce_sum_total = ce_sum_total + ce_sum
                        n_total = n_total + n_masked
                        total_aux = total_aux + aux_blk

                        correct1 += c1
                        correct3 += c3
                        correctA1 += a1
                        correctB1 += b1
                        correctA3 += a3
                        correctB3 += b3

                        del X_A_tensor, X_B_tensor, Aemb_tensor, Bemb_tensor, labels_tensor

                    else:
                        for idx, t_block in enumerate(range(chunk_start, chunk_end)):
                            if self.training and self.use_checkpointing:
                                ce_sum, n_masked, aux_blk, c1, c3, a1, b1, a3, b3 = cp.checkpoint(
                                    self._forward_s_scan_chunk_streaming,
                                    X_A_batch[idx], X_B_batch[idx], Aemb_batch[idx], Bemb_batch[idx], labels_batch[idx],
                                    torch.tensor(t_block, device=seq.device),
                                    torch.tensor(1, device=seq.device),
                                    torch.tensor(step, device=seq.device),
                                    report_ab_t,
                                    use_reentrant=False
                                )
                            else:
                                ce_sum, n_masked, aux_blk, c1, c3, a1, b1, a3, b3 = self._forward_s_scan_chunk_streaming(
                                    X_A_batch[idx], X_B_batch[idx], Aemb_batch[idx], Bemb_batch[idx], labels_batch[idx],
                                    torch.tensor(t_block, device=seq.device),
                                    torch.tensor(1, device=seq.device),
                                    torch.tensor(step, device=seq.device),
                                    report_ab_t,
                                )

                            ce_sum_total = ce_sum_total + ce_sum
                            n_total = n_total + n_masked
                            total_aux = total_aux + aux_blk

                            correct1 += c1
                            correct3 += c3
                            correctA1 += a1
                            correctB1 += b1
                            correctA3 += a3
                            correctB3 += b3

                del h, rc_h

                if self.training and self.use_ema_teacher and self.auto_update_ema_in_forward:
                    self.update_ema()

                HybridOutput = namedtuple("HybridOutput", ["logits"])
                step_t = torch.tensor(step, device=seq.device, dtype=torch.long)

                stats = torch.stack([
                    keep_rate.to(torch.float32),
                    correct1.to(torch.float32),
                    correct3.to(torch.float32),
                    correctA1.to(torch.float32),
                    correctB1.to(torch.float32),
                    correctA3.to(torch.float32),
                    correctB3.to(torch.float32),
                ], dim=0)

                return HybridOutput(logits=(ce_sum_total, n_total, total_aux, stats, step_t)), None

            # ==========================================================
            # non-streaming：原来的“预分配写入”逻辑（保持兼容）
            # ==========================================================
            fused = None

            if self.use_s_scan:
                B, H, L = h.shape
                l = self.block_size
                K = (L + l - 1) // l
                chunk_size = max(1, getattr(self, "checkpoint_chunk_size", 2))

                collect_fused = bool(self.for_representation)
                collect_logits = (not self.for_representation) or self.pretrain
                need_ab_logits = bool((self.pretrain and self.return_ab_logits) or self.use_rc_kl)

                fused_out = torch.empty((B, L, self.d_model), device=seq.device, dtype=amp_dtype) if collect_fused else None
                logits_out = torch.empty((B, L, self.alphabet_size), device=seq.device, dtype=amp_dtype) if collect_logits else None
                logitsA_out = torch.empty((B, L, self.alphabet_size), device=seq.device, dtype=amp_dtype) if need_ab_logits else None
                logitsB_out = torch.empty((B, L, self.alphabet_size), device=seq.device, dtype=amp_dtype) if need_ab_logits else None

                mask_A_rc = torch.empty((B, L), device=seq.device, dtype=torch.bool)
                mask_B_rc = torch.empty((B, L), device=seq.device, dtype=torch.bool)

                total_aux = torch.zeros([], device=seq.device, dtype=torch.float32)

                for chunk_start in range(0, K, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, K)

                    X_A_batch, X_B_batch = [], []
                    Aemb_batch, Bemb_batch = [], []
                    lengths = []

                    for t_block in range(chunk_start, chunk_end):
                        start = t_block * l
                        end = min(start + l, L)
                        blk_len = end - start
                        lengths.append(blk_len)

                        fwd_emb = h[:, :, start:end].transpose(1, 2)
                        rc_emb = rc_h[:, :, start:end].transpose(1, 2)

                        fwd_in = self.dropout(h[:, :, start:end]).transpose(1, 2).contiguous()
                        rc_in = self.dropout(rc_h[:, :, start:end]).transpose(1, 2).contiguous()

                        if (t_block % 2) == 0:
                            X_A_batch.append(fwd_in)
                            X_B_batch.append(rc_in)
                        else:
                            X_A_batch.append(rc_in)
                            X_B_batch.append(fwd_in)

                        Aemb_batch.append(fwd_emb.contiguous())
                        Bemb_batch.append(rc_emb.contiguous())

                        valA = (t_block % 2) == 1
                        mask_A_rc[:, start:end] = valA
                        mask_B_rc[:, start:end] = (not valA)

                    if len(set(lengths)) == 1:
                        blk_len = lengths[0]
                        X_A_tensor = torch.cat(X_A_batch, dim=0)
                        X_B_tensor = torch.cat(X_B_batch, dim=0)
                        Aemb_tensor = torch.cat(Aemb_batch, dim=0)
                        Bemb_tensor = torch.cat(Bemb_batch, dim=0)

                        need_logits_t = torch.tensor(int(collect_logits), device=seq.device)
                        need_ab_t = torch.tensor(int(need_ab_logits), device=seq.device)

                        if self.training and self.use_checkpointing:
                            fused_blk, logits_blk, logitsA_blk, logitsB_blk, aux_blk = cp.checkpoint(
                                self._forward_s_scan_chunk,
                                X_A_tensor, X_B_tensor, Aemb_tensor, Bemb_tensor,
                                torch.tensor(chunk_start, device=seq.device),
                                torch.tensor(len(X_A_batch), device=seq.device),
                                torch.tensor(step, device=seq.device),
                                need_logits_t, need_ab_t,
                                use_reentrant=False
                            )
                        else:
                            fused_blk, logits_blk, logitsA_blk, logitsB_blk, aux_blk = self._forward_s_scan_chunk(
                                X_A_tensor, X_B_tensor, Aemb_tensor, Bemb_tensor,
                                torch.tensor(chunk_start, device=seq.device),
                                torch.tensor(len(X_A_batch), device=seq.device),
                                torch.tensor(step, device=seq.device),
                                need_logits_t, need_ab_t,
                            )

                        total_aux = total_aux + aux_blk

                        nb = len(X_A_batch)
                        fused_view = fused_blk.view(nb, B, blk_len, -1)

                        logits_view = logits_blk.view(nb, B, blk_len, -1) if (collect_logits and logits_blk.numel() > 0) else None
                        logitsA_view = logitsA_blk.view(nb, B, blk_len, -1) if (need_ab_logits and logitsA_blk.numel() > 0) else None
                        logitsB_view = logitsB_blk.view(nb, B, blk_len, -1) if (need_ab_logits and logitsB_blk.numel() > 0) else None

                        for c, t_block in enumerate(range(chunk_start, chunk_end)):
                            start = t_block * l
                            end = min(start + l, L)

                            if collect_fused:
                                fused_out[:, start:end, :] = fused_view[c]
                            if collect_logits:
                                logits_out[:, start:end, :] = logits_view[c]
                            if need_ab_logits:
                                logitsA_out[:, start:end, :] = logitsA_view[c]
                                logitsB_out[:, start:end, :] = logitsB_view[c]

                        del X_A_tensor, X_B_tensor, Aemb_tensor, Bemb_tensor
                        del fused_blk, logits_blk, logitsA_blk, logitsB_blk

                    else:
                        for idx, t_block in enumerate(range(chunk_start, chunk_end)):
                            start = t_block * l
                            end = min(start + l, L)

                            X_A_tensor = X_A_batch[idx]
                            X_B_tensor = X_B_batch[idx]
                            Aemb_tensor = Aemb_batch[idx]
                            Bemb_tensor = Bemb_batch[idx]

                            need_logits_t = torch.tensor(int(collect_logits), device=seq.device)
                            need_ab_t = torch.tensor(int(need_ab_logits), device=seq.device)

                            if self.training and self.use_checkpointing:
                                fused_blk, logits_blk, logitsA_blk, logitsB_blk, aux_blk = cp.checkpoint(
                                    self._forward_s_scan_chunk,
                                    X_A_tensor, X_B_tensor, Aemb_tensor, Bemb_tensor,
                                    torch.tensor(t_block, device=seq.device),
                                    torch.tensor(1, device=seq.device),
                                    torch.tensor(step, device=seq.device),
                                    need_logits_t, need_ab_t,
                                    use_reentrant=False
                                )
                            else:
                                fused_blk, logits_blk, logitsA_blk, logitsB_blk, aux_blk = self._forward_s_scan_chunk(
                                    X_A_tensor, X_B_tensor, Aemb_tensor, Bemb_tensor,
                                    torch.tensor(t_block, device=seq.device),
                                    torch.tensor(1, device=seq.device),
                                    torch.tensor(step, device=seq.device),
                                    need_logits_t, need_ab_t,
                                )

                            total_aux = total_aux + aux_blk

                            if collect_fused:
                                fused_out[:, start:end, :] = fused_blk
                            if collect_logits:
                                logits_out[:, start:end, :] = logits_blk
                            if need_ab_logits and logitsA_blk.numel() > 0:
                                logitsA_out[:, start:end, :] = logitsA_blk
                                logitsB_out[:, start:end, :] = logitsB_blk

                            del fused_blk, logits_blk, logitsA_blk, logitsB_blk

                del h, rc_h

                logits = logits_out if collect_logits else None
                logits_A_only = logitsA_out if need_ab_logits else None
                logits_B_only = logitsB_out if need_ab_logits else None
                fused = fused_out if collect_fused else None

            else:
                feat = self.dropout(h).transpose(1, 2).contiguous()
                rc_feat = self.dropout(rc_h).transpose(1, 2).contiguous()

                H_A = self.branchA_core(feat)
                H_Br = self.branchB_core(rc_feat)
                R_A = H_A
                R_B = torch.flip(H_Br, dims=[1])

                if self.use_bridge:
                    R_A, R_B = self.bridge(R_A, R_B)

                fA, rA = self.proj_A(R_A)
                FA = fA + rA
                fB, rB = self.proj_B(R_B)
                FB = fB + rB

                gate_in = torch.cat([FA, FB], dim=-1)
                g_logits = self.gate_fuse(gate_in)
                g_raw = torch.sigmoid(g_logits / max(1e-6, getattr(self, "gate_temp", 1.0)))

                if step < getattr(self, "gate_freeze_steps", 0):
                    g = 0.5 * torch.ones_like(g_raw)
                else:
                    g = g_raw

                if getattr(self, "detach_gate", False):
                    mix = g.detach() * FA + (1 - g.detach()) * FB
                else:
                    mix = g * FA + (1 - g) * FB

                fused = F.layer_norm(mix, (mix.size(-1),))
                fused = ensure_finite(fused, "fused")

                if self.use_final_conv:
                    fused = self.final_conv(fused.permute(0, 2, 1)).permute(0, 2, 1)

                logits = self.out_linear(fused) if (not self.for_representation or self.pretrain) else None

                need_ab_logits = bool((self.pretrain and self.return_ab_logits) or self.use_rc_kl)
                logits_A_only = self.out_linear(FA) if need_ab_logits else None
                logits_B_only = self.out_linear(FB) if need_ab_logits else None

                mask_A_rc = torch.zeros(FA.size()[:2], dtype=torch.bool, device=FA.device)
                mask_B_rc = torch.zeros_like(mask_A_rc)

                total_aux = logits.new_zeros(()) if self.pretrain else None

                del h, rc_h, feat, rc_feat

        if self.for_representation:
            return fused, None

        if self.training and self.use_ema_teacher and self.auto_update_ema_in_forward:
            self.update_ema()

        if self.pretrain:
            if logits_A_only is None:
                logits_A_only = self.out_linear(FA)
            if logits_B_only is None:
                logits_B_only = self.out_linear(FB)

            HybridOutput = namedtuple("HybridOutput", ["logits"])
            return HybridOutput(
                logits=(
                    logits, mlm_mask, total_aux,
                    logits_A_only.detach(), logits_B_only.detach(),
                    mask_A_rc.detach(), mask_B_rc.detach(), int(step)
                )
            ), None

        return logits, None

    @property
    def d_output(self):
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model