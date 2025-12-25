
import math
import torch
import copy
import torch.nn as nn
from torch import amp
import torch.nn.functional as F
from functools import partial
from contextlib import contextmanager
from collections import namedtuple
from typing import Dict, Optional, Tuple, Any

from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

# 编译ckpt到bin文件时候打开注释。
import os as _os, torch as _torch
if _os.environ.get("DISABLE_TORCH_COMPILE", "1") == "1" and hasattr(_torch, "compile"):
    def _no_compile(fn=None, *args, **kwargs):
        if fn is None:
            def deco(f): return f
            return deco
        return fn
    _torch.compile = _no_compile
print("torch.compile =>", torch.compile)


from fla.layers import comba
from fla.layers.attn import Attention
from fla.modules import GatedMLP as SambaMLP
from fla.modules import RMSNorm
from torch.cuda.amp import autocast


try:
    
    from .configuration_crossdna import CrossDNAConfig
except ImportError:
    
    from configuration_crossdna import CrossDNAConfig




def complement(seq: torch.Tensor) -> torch.Tensor:
    # A=0, C=1, G=2, T=3, N=4
    comp = 3 - seq
    comp[seq == 4] = 4
    return comp


def reverse_complement(seq: torch.Tensor) -> torch.Tensor:
    comp = complement(seq)
    return torch.flip(comp, dims=[1])


def make_complement_perm(C=5, device=None, dtype=torch.float32):
    # A=0,C=1,G=2,T=3,N=4  ->  T,A,C,G,N
    perm = torch.tensor([3, 0, 2, 1, 4], device=device)
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


# ========================
# RC 一致性（可选损失）
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


# ========================
# Barlow & TV（可选）
# ========================
def barlow_strand_loss_v2(z1, z2, λ_off=0.04, λ_diag=0.04, eps=1e-3):
    
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
    c = (z1.t() @ z2) / max(1, n)  # [H,H]
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
    RMSNorm -> Sliding-window Attention -> Residual -> RMSNorm -> Gated MLP -> Residual
    """

    def __init__(self, config: Any):
        super().__init__()

        
        if isinstance(config, dict):
            c = config
        else:
            try:
                c = vars(config)  
            except Exception as e:
                raise TypeError(f"transformer_cfg must be dict-like, got {type(config)}") from e

        attn_cfg = c["attn"]
       
        self.mixer_norm = RMSNorm(hidden_size=c["hidden_size"], eps=c.get("norm_eps", 1e-5))
        self.mixer = Attention(
            hidden_size=c["hidden_size"],
            num_heads=attn_cfg["num_heads"],
            num_kv_heads=attn_cfg["num_kv_heads"],
            qkv_bias=attn_cfg["qkv_bias"],
            window_size=attn_cfg["window_size"],
            rope_theta=attn_cfg["rope_theta"],
            max_position_embeddings=c["max_position_embeddings"]
        )
        self.mlp_norm = RMSNorm(c["hidden_size"], eps=c.get("norm_eps", 1e-5))
        self.mlp = SambaMLP(
            hidden_size=c["hidden_size"],
            hidden_ratio=c["hidden_ratio"],
            hidden_act=c["hidden_act"],
            fuse_swiglu=c["fuse_swiglu"]
        )
        self.pre_scale = 1.0 / math.sqrt(2.0)

    def forward(self, hidden_states: torch.Tensor,
                cache_params: Optional[Any] = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        residual = hidden_states
        x = self.mixer_norm(hidden_states)

        amp_dtype = preferred_amp_dtype()
        with amp.autocast("cuda", enabled=True, dtype=amp_dtype):
            x_scaled = x * self.pre_scale
            attn_out, _, cache_params = self.mixer(
                hidden_states=x_scaled,
                past_key_values=cache_params,
                **kwargs
            )
            attn_out = attn_out / self.pre_scale

        ensure_finite(attn_out, "attention_out")
        h = residual + attn_out.to(x.dtype)

        residual = h
        x = self.mlp_norm(h)
        with amp.autocast("cuda", enabled=True, dtype=amp_dtype):
            x = self.mlp(x, **kwargs)
        h = residual + x
        ensure_finite(h, "block_output")
        return h, cache_params



class EnhancedHybridCore(nn.Module):
    def __init__(self, hidden_dim, comba_cfg, transformer_cfg, layer_idx=0, device=None, dtype=None):
        super().__init__()
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
        # x: [B, l, H]
        m_out = self._first(self.comba(x))
        t_out, _ = self.transformer(m_out)
        concat = torch.cat([m_out, t_out], dim=-1)
        g = torch.sigmoid(self.gate(concat))
        fused = g * t_out + (1 - g) * m_out
        y = self.out_norm(fused)
        ensure_finite(y, "EnhancedHybridCore.out")
        return y



class DeepEnhancedBranch(nn.Module):
    def __init__(self, hidden_dim: int, comba_cfg: Dict | None, transformer_cfg: Any, depth: int = 4,
                 drop_path_rates=None, *, device=None, dtype=None):
        super().__init__()
        self.layers = nn.ModuleList()
        if drop_path_rates is None:
            rates = [0.05 * (i / max(1, depth - 1)) for i in range(depth)]
        elif isinstance(drop_path_rates, (float, int)):
            rates = [float(drop_path_rates)] * depth
        else:
            rates = list(drop_path_rates) + [drop_path_rates[-1]] * (depth - len(drop_path_rates))
        for i in range(depth):
            layer_cfg = transformer_cfg.copy()
            layer_cfg["drop_path_prob"] = rates[i]
            self.layers.append(EnhancedHybridCore(hidden_dim, comba_cfg, layer_cfg, i, device, dtype))
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):  # x:[B,l,H]
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

        
        self.dw_B = nn.ModuleList([
            nn.Conv1d(h, h, kernel_size, padding=pad(d), dilation=d,
                      groups=h, bias=False) for d in dilations
        ])
        self.mix_B = nn.Conv1d(h * len(dilations), h, 1)

        
        self.dw_A = nn.ModuleList([
            nn.Conv1d(h, h, kernel_size, padding=pad(d), dilation=d,
                      groups=h, bias=False) for d in dilations
        ])
        self.mix_A = nn.Conv1d(h * len(dilations), h, 1)

        
        self.proj_B2A = nn.Linear(h, h)
        self.proj_A2B = nn.Linear(h, h)

        
        self.use_global_token = use_global_token
        if use_global_token:
            self.glb_B2A = nn.Linear(h, h)
            self.glb_A2B = nn.Linear(h, h)

        
        self.gate = nn.Linear(h * 4, h * 2)  # -> [gA, gB]
        self.dropout = nn.Dropout(dropout)
        self.normA = nn.LayerNorm(h)
        self.normB = nn.LayerNorm(h)

    @staticmethod
    def _agg(x: torch.Tensor, branches: nn.ModuleList, mix: nn.Module) -> torch.Tensor:
        # x:[B,L,H] -> [B,L,H]
        xch = x.transpose(1, 2)                         # [B,H,L]
        ys = [conv(xch) for conv in branches]           # list of [B,H,L]
        y = torch.cat(ys, dim=1)                        # [B,H*len,L]
        y = mix(y).transpose(1, 2).contiguous()         # [B,L,H]
        return y

    def forward(self, xA: torch.Tensor, xB: torch.Tensor):
        
        ctxB = self._agg(xB, self.dw_B, self.mix_B)     
        ctxA = self._agg(xA, self.dw_A, self.mix_A)     

        
        locA = self.proj_B2A(xB + ctxB)                 # B→A
        locB = self.proj_A2B(xA + ctxA)                 # A→B

        
        if self.use_global_token:
            gB = self.glb_B2A(xB.mean(dim=1, keepdim=True))  # [B,1,H]
            gA = self.glb_A2B(xA.mean(dim=1, keepdim=True))  # [B,1,H]
            locA = locA + gB.expand(-1, xA.size(1), -1)
            locB = locB + gA.expand(-1, xB.size(1), -1)

        
        z = torch.cat([xA, xB, xA - xB, xA * xB], dim=-1)    # [B,L,4H]
        gA, gB = self.gate(z).chunk(2, dim=-1)
        gA = torch.sigmoid(gA)
        gB = torch.sigmoid(gB)

        yA = self.normA(xA + self.dropout(gA * locA))
        yB = self.normB(xB + self.dropout(gB * locB))

        ensure_finite(yA, "TokenBridgeLite.A")
        ensure_finite(yB, "TokenBridgeLite.B")
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
        for m in modules: m.eval()
        yield
    finally:
        for m, s in zip(modules, states): m.train(s)


class SSScanDNAHybridModel(nn.Module):


    def __init__(
        self,
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
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.pretrain = pretrain
        self.for_representation = for_representation
        self.block_size = block_size
        self.use_final_conv = use_final_conv
        self.d_model = d_model

        
        self.register_buffer("g_step", torch.zeros(1, dtype=torch.long))

        
        self.linear = nn.Conv1d(alphabet_size, d_model, kernel_size=9, padding=4)
        self.rc_linear = nn.Conv1d(alphabet_size, d_model, kernel_size=9, padding=4)

        
        self.branchA_core = DeepEnhancedBranch(
            hidden_dim=d_model, comba_cfg=comba_cfg, transformer_cfg=transformer_cfg,
            depth=depth, drop_path_rates=drop_path_rates
        )
        self.branchB_core = DeepEnhancedBranch(
            hidden_dim=d_model, comba_cfg=comba_cfg, transformer_cfg=transformer_cfg,
            depth=depth, drop_path_rates=drop_path_rates
        )

        
        self.use_bridge = use_bridge
        if self.use_bridge:
            self.bridge = TokenBridge(d_model, dropout=bridge_dropout)

        
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self.auto_update_ema_in_forward = auto_update_ema_in_forward
        if self.use_ema_teacher:
            self.branchA_core_ema = copy.deepcopy(self.branchA_core)
            self.branchB_core_ema = copy.deepcopy(self.branchB_core)
            for p in self.branchA_core_ema.parameters(): p.requires_grad_(False)
            for p in self.branchB_core_ema.parameters(): p.requires_grad_(False)
            if self.use_bridge:
                self.bridge_ema = copy.deepcopy(self.bridge)
                for p in self.bridge_ema.parameters(): p.requires_grad_(False)

        
        self.proj_A = Mlp(d_model, d_model * 2, d_model, activation=F.gelu, return_residual=True)
        self.proj_B = Mlp(d_model, d_model * 2, d_model, activation=F.gelu, return_residual=True)
        self.gate_fuse = nn.Linear(2 * d_model, d_model)
        self.out_linear = nn.Linear(d_model, alphabet_size)
        self.dropout = nn.Dropout(dropout)

        
        P_comp, _ = make_complement_perm(self.alphabet_size)
        self.register_buffer("P_comp", P_comp)

        
        self.use_s_scan = use_s_scan
        self.use_rc_kl = use_rc_kl
        self.use_barlow = use_barlow
        self.use_tv = use_tv
        self.sem_max_weight = sem_max_weight
        self.sem_warmup_steps = sem_warmup_steps
        self.rc_max_weight = rc_max_weight
        self.rc_warmup_steps = rc_warmup_steps
        self.rc_tau = rc_tau
        self.rc_bidirectional_stopgrad = rc_bidirectional_stopgrad
        self.aux_ce_weight = aux_ce_weight
        self.gate_freeze_steps = gate_freeze_steps
        self.detach_gate = detach_gate
        self.gate_sup_weight = gate_sup_weight
        self.gate_sup_warmup_steps = gate_sup_warmup_steps
        self.gate_temp = gate_temp

        if use_final_conv:
            self.final_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    @torch.no_grad()
    def update_ema(self):
        if not getattr(self, "use_ema_teacher", False): return
        d = float(getattr(self, "ema_decay", 0.999))
        for m_ema, m in [(self.branchA_core_ema, self.branchA_core),
                         (self.branchB_core_ema, self.branchB_core)]:
            for p_ema, p in zip(m_ema.parameters(), m.parameters()):
                p_ema.data.lerp_(p.data, 1.0 - d)
        if getattr(self, "use_bridge", False) and hasattr(self, "bridge_ema"):
            for p_ema, p in zip(self.bridge_ema.parameters(), self.bridge.parameters()):
                p_ema.data.lerp_(p.data, 1.0 - d)

    
    def forward(self, seq, t=None, cls=None, return_embedding=False, state=None):
        step = int(self.g_step.item())
        if self.training:
            self.g_step += 1

        if self.pretrain:
            mask = seq[1]
            seq = seq[0]
        else:
            mask = None
        
        mn, mx = int(seq.min()), int(seq.max())
        assert 0 <= mn and mx < self.alphabet_size, f"seq ids out of range: [{mn}, {mx}] vs alpha={self.alphabet_size}"

        
        rc_seq = reverse_complement(seq)
        seq_oh = F.one_hot(seq, num_classes=self.alphabet_size).float()
        rc_oh = F.one_hot(rc_seq, num_classes=self.alphabet_size).float()
        

        h = F.gelu(self.linear(seq_oh.permute(0, 2, 1)))       # [B,H,L]
        rc_h = F.gelu(self.rc_linear(rc_oh.permute(0, 2, 1)))  # [B,H,L]
        feat = self.dropout(h).permute(0, 2, 1)                # [B,L,H]
        rc_feat = self.dropout(rc_h).permute(0, 2, 1)

        fused = None  

        
        if self.use_s_scan:
            B, L, H = feat.shape
            l = self.block_size
            K = (L + l - 1) // l

            
            collect_fused = bool(self.for_representation)
            collect_logits = (not self.for_representation) or self.pretrain
            collect_ab_logits = (self.pretrain or self.use_rc_kl)

            fused_chunks = [] if collect_fused else None
            logits_chunks = [] if collect_logits else None
            logitsA_chunks = [] if collect_ab_logits else None
            logitsB_chunks = [] if collect_ab_logits else None
            maskA_list, maskB_list = [], []

            total_aux = feat.new_zeros([])  
            mem_A = mem_B = None

            for t_block in range(K):
                start = t_block * l
                end = min(start + l, L)

                X_fwd = feat[:, start:end, :]
                X_rc  = rc_feat[:, start:end, :]

                if (t_block % 2) == 0:
                    X_A, X_B = X_fwd, X_rc
                    rev_A, rev_B = False, True
                    maskA_rc_blk = torch.zeros(B, end - start, dtype=torch.bool, device=feat.device)
                    maskB_rc_blk = torch.ones (B, end - start, dtype=torch.bool, device=feat.device)
                else:
                    X_A, X_B = X_rc, X_fwd
                    rev_A, rev_B = True, False
                    maskA_rc_blk = torch.ones (B, end - start, dtype=torch.bool, device=feat.device)
                    maskB_rc_blk = torch.zeros(B, end - start, dtype=torch.bool, device=feat.device)


                
                H_A = self.branchA_core(X_A)
                H_B = self.branchB_core(X_B)
                if rev_A: H_A = torch.flip(H_A, dims=[1])
                if rev_B: H_B = torch.flip(H_B, dims=[1])

                if self.use_bridge:
                    H_A, H_B = self.bridge(H_A, H_B)

                
                fA, rA = self.proj_A(H_A); FA = fA + rA
                fB, rB = self.proj_B(H_B); FB = fB + rB

                gate_in_blk = torch.cat([FA, FB], dim=-1)
                g_logits_blk = self.gate_fuse(gate_in_blk)
                g_raw_blk = torch.sigmoid(g_logits_blk / getattr(self, "gate_temp", 1.0))
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

                if collect_fused:
                    fused_chunks.append(fused_blk)

                if collect_logits:
                    logits_blk = self.out_linear(fused_blk)
                    logits_chunks.append(logits_blk)

                if collect_ab_logits:
                    logitsA_chunks.append(self.out_linear(FA))
                    logitsB_chunks.append(self.out_linear(FB))

                maskA_list.append(maskA_rc_blk)
                maskB_list.append(maskB_rc_blk)

                
                if self.pretrain:
                    with torch.no_grad():
                        A_feat_blk = F.gelu(self.linear(
                            F.one_hot(seq[:, start:end], self.alphabet_size).float().permute(0, 2, 1)
                        )).permute(0, 2, 1)
                        B_feat_blk_rc = F.gelu(self.rc_linear(
                            F.one_hot(rc_seq[:, start:end], self.alphabet_size).float().permute(0, 2, 1)
                        )).permute(0, 2, 1)

                        teacherA = self.branchA_core_ema if self.use_ema_teacher else self.branchA_core
                        teacherB = self.branchB_core_ema if self.use_ema_teacher else self.branchB_core
                        tbridge  = self.bridge_ema if (self.use_bridge and self.use_ema_teacher and hasattr(self, "bridge_ema")) else (self.bridge if self.use_bridge else None)

                        mods = [teacherA, teacherB] + ([tbridge] if tbridge is not None else [])
                        with eval_mode(*mods):
                            R_plus_A_blk = teacherA(A_feat_blk)
                            R_plus_B_blk = teacherB(A_feat_blk)
                            if tbridge is not None:
                                R_plus_A_blk, R_plus_B_blk = tbridge(R_plus_A_blk, R_plus_B_blk)

                            R_minus_A_blk_rc = teacherA(B_feat_blk_rc)
                            R_minus_B_blk_rc = teacherB(B_feat_blk_rc)
                            R_minus_A_blk_fwd = torch.flip(R_minus_A_blk_rc, dims=[1])
                            R_minus_B_blk_fwd = torch.flip(R_minus_B_blk_rc, dims=[1])
                            if tbridge is not None:
                                R_minus_A_blk_fwd, R_minus_B_blk_fwd = tbridge(R_minus_A_blk_fwd, R_minus_B_blk_fwd)

                    R_A_teacher_blk = torch.where(maskA_rc_blk.unsqueeze(-1), R_minus_A_blk_fwd, R_plus_A_blk)
                    R_B_teacher_blk = torch.where(maskB_rc_blk.unsqueeze(-1), R_minus_B_blk_fwd, R_plus_B_blk)

                    sem_A = semantic_preservation_loss(R_A_teacher_blk, FA)
                    sem_B = semantic_preservation_loss(R_B_teacher_blk, FB)
                    w_sem = linear_warmup_weight(step, getattr(self, "sem_warmup_steps", 0),
                                                 getattr(self, "sem_max_weight", 1.0))
                    total_aux = total_aux + w_sem * (sem_A + sem_B)

                    if (getattr(self, "gate_sup_weight", 0.0) > 0.0) and (step >= getattr(self, "gate_freeze_steps", 0)):
                        g_target_blk = (~maskA_rc_blk).float().unsqueeze(-1)
                        g_token_logits_blk = g_logits_blk.mean(dim=-1, keepdim=True) / getattr(self, "gate_temp", 1.0)
                        w_gate = linear_warmup_weight(
                            step - getattr(self, "gate_freeze_steps", 0),
                            getattr(self, "gate_sup_warmup_steps", 0),
                            getattr(self, "gate_sup_weight", 0.0),
                        )
                        total_aux = total_aux + w_gate * F.binary_cross_entropy_with_logits(
                            g_token_logits_blk, g_target_blk
                        )

                    if self.use_rc_kl and getattr(self, "rc_max_weight", 0.0) > 0:
                        if getattr(self, "rc_bidirectional_stopgrad", True):
                            rc = rc_consistency_bidirectional_stopgrad(
                                logitsA_chunks[-1], logitsB_chunks[-1], self.P_comp, tau=getattr(self, "rc_tau", 1.5)
                            )
                        else:
                            rc = rc_consistency_kl(
                                logitsA_chunks[-1], logitsB_chunks[-1], self.P_comp, tau=getattr(self, "rc_tau", 1.5)
                            )
                        w_rc = linear_warmup_weight(step, getattr(self, "rc_warmup_steps", 0),
                                                    getattr(self, "rc_max_weight", 0.0))
                        total_aux = total_aux + w_rc * rc

                    if self.use_barlow:
                        total_aux = total_aux + barlow_strand_loss_v2(H_A, H_B)
                    if self.use_tv:
                        total_aux = total_aux + tv_mixed(fused_blk)

            
            logits = torch.cat(logits_chunks, dim=1)[:, :L, :] if collect_logits else None
            logits_A_only = torch.cat(logitsA_chunks, dim=1)[:, :L, :] if collect_ab_logits else None
            logits_B_only = torch.cat(logitsB_chunks, dim=1)[:, :L, :] if collect_ab_logits else None
            mask_A_rc = torch.cat(maskA_list, dim=1)[:, :L]
            mask_B_rc = torch.cat(maskB_list, dim=1)[:, :L]
            if collect_fused:
                fused = torch.cat(fused_chunks, dim=1)[:, :L, :]

        else:
            
            H_A = self.branchA_core(feat)
            H_Br = self.branchB_core(rc_feat)
            R_A = H_A
            R_B = torch.flip(H_Br, dims=[1])

            if self.use_bridge:
                R_A, R_B = self.bridge(R_A, R_B)

            fA, rA = self.proj_A(R_A); FA = fA + rA
            fB, rB = self.proj_B(R_B); FB = fB + rB

            gate_in = torch.cat([FA, FB], dim=-1)
            g_logits = self.gate_fuse(gate_in)
            g_raw = torch.sigmoid(g_logits / getattr(self, "gate_temp", 1.0))
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
            logits_A_only = self.out_linear(FA) if (self.pretrain or self.use_rc_kl) else None
            logits_B_only = self.out_linear(FB) if (self.pretrain or self.use_rc_kl) else None
            mask_A_rc = torch.zeros(FA.size()[:2], dtype=torch.bool, device=FA.device)
            mask_B_rc = torch.zeros_like(mask_A_rc)
            total_aux = logits.new_zeros(()) if self.pretrain else None

        
        if self.for_representation:
            
            return fused, None

        if self.training and self.use_ema_teacher and self.auto_update_ema_in_forward:
            self.update_ema()

        current_step = int(step)

        if self.pretrain:
            
            if logits_A_only is None: logits_A_only = self.out_linear(FA)
            if logits_B_only is None: logits_B_only = self.out_linear(FB)
            HybridOutput = namedtuple("HybridOutput", ["logits"])
            return HybridOutput(
                logits=(logits,
                        mask,
                        total_aux,
                        logits_A_only.detach(),
                        logits_B_only.detach(),
                        mask_A_rc.detach(),
                        mask_B_rc.detach(),
                        current_step)
            ), None

        return logits, None

    @property
    def d_output(self):
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model



class CrossDNAForMaskedLM(PreTrainedModel):
    config_class = CrossDNAConfig
    base_model_prefix = "backbone"

    def __init__(self, config: CrossDNAConfig):
        super().__init__(config)
        self.config = config

        
        self.backbone = SSScanDNAHybridModel(
            alphabet_size=config.alphabet_size,
            d_model=config.d_model,
            block_size=config.block_size,
            comba_cfg=config.comba_cfg,
            transformer_cfg=config.transformer_cfg,
            depth=config.depth,
            drop_path_rates=config.drop_path_rates,
            pretrain=config.pretrain,
            for_representation=config.for_representation,
            use_s_scan=config.use_s_scan,
            use_mem=config.use_mem,
            use_rc_kl=config.use_rc_kl,
            use_barlow=config.use_barlow,
            use_tv=config.use_tv,
            sem_max_weight=config.sem_max_weight,
            sem_warmup_steps=config.sem_warmup_steps,
            aux_ce_weight=config.aux_ce_weight,
            gate_freeze_steps=config.gate_freeze_steps,
            detach_gate=config.detach_gate,
            gate_sup_weight=config.gate_sup_weight,
            gate_sup_warmup_steps=config.gate_sup_warmup_steps,
            gate_temp=config.gate_temp,
            dropout=config.dropout,
            use_bridge=config.use_bridge,
            bridge_dropout=0.0,
        )
        
        self.post_init()

    @property
    def mask_token_id(self) -> int:
        return getattr(self.config, "mask_token_id", 3)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,  # 未使用
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> MaskedLMOutput:


        if labels is not None:
            mlm_mask = (labels != -100)
        else:
            mlm_mask = (input_ids == self.mask_token_id)

        if self.config.pretrain:
            
            outputs, _ = self.backbone((input_ids, mlm_mask))
            
            logits = outputs.logits[0]  # [B, L, vocab_size]
        else:
            logits, _ = self.backbone(input_ids)  # [B, L, vocab_size]

        loss = None
        if labels is not None:
            
            vocab = self.config.alphabet_size
            logits_2d = logits.view(-1, vocab)
            labels_1d = labels.view(-1)
            loss = F.cross_entropy(logits_2d, labels_1d, ignore_index=-100)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )