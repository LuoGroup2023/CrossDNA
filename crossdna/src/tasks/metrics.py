import os,math
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from functools import partial
import torchmetrics.functional as tm_f
import torch.distributions as dist
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from torchmetrics import Metric
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision
from scipy import stats


class CorrectAggregatedMetric(Metric):
    """This is needed to calculate some metrics b/c small batch sizes cause aggregation via a simple
        average to be off, as some classes might not be present in batch but will get penalized with a 0."""

    def __init__(self, class_idx: int, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.class_idx = torch.tensor(class_idx)
        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _update(self, numerator, denominator, preds, y) -> tuple:
        raise NotImplemented

    def update(self, logits: torch.Tensor, y: torch.Tensor):
        # update metric states
        preds = torch.argmax(logits, dim=-1)
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        assert preds.shape == y.shape, f"preds shape {preds.shape} != y shape {y.shape}"
        self.numerator, self.denominator = self._update(self.numerator, self.denominator, preds, y)

    def compute(self):
        # compute final result
        value = self.numerator.float() / self.denominator if self.denominator > 0 else torch.tensor(0.0)
        return value

    def reset(self):
        self.numerator = torch.tensor(0.0)
        self.denominator = torch.tensor(0.0)


class AccuracyPerClass(CorrectAggregatedMetric):
    """Calculate per class accuracy, i.e. P(y_hat = class_idx AND y = class_idx OR y_hat != class_idx AND y != class_idx)
    """

    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (y == class_idx)
        numerator += (preds[relevant_idxs] == class_idx).sum()
        denominator += relevant_idxs.sum()
        relevant_idxs = (y != class_idx)
        numerator += (preds[relevant_idxs] != class_idx).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator


class PrecisionPerClass(CorrectAggregatedMetric):
    """Calculate per class precision, i.e. P(y_hat = y | y_hat = class_idx)
    """

    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (preds == class_idx)
        numerator += (preds[relevant_idxs] == y[relevant_idxs]).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator


class RecallPerClass(CorrectAggregatedMetric):
    """Calculate per class recall, i.e. P(y_hat = y | y = class_idx)
    """

    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (y == class_idx)
        numerator += (preds[relevant_idxs] == y[relevant_idxs]).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator


def mcc(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return matthews_corrcoef(y.cpu().numpy(), y_hat.cpu().numpy())

def mcc_logits(logits, y=None, unlabeled_output="logits"):  # unlabeled_output: "logits" | "pred"
    # 统一展平到 [N, C]
    flat_logits = logits.view(-1, logits.shape[-1])

    # 无标签：返回 view 之后的 logits 或 pred
    if y is None or (isinstance(y, torch.Tensor) and y.numel() == 0):
        if unlabeled_output == "pred":
            return torch.argmax(flat_logits, dim=-1)   # [N]
        return flat_logits                             # [N, C]

    # 有标签：计算 MCC
    print('y.shape in train:', y.shape)
    y = y.view(-1)
    y_hat = torch.argmax(flat_logits, dim=-1)
    print('label:', y)
    print('logits:', torch.softmax(flat_logits, dim=-1))
    print('label_predict:', y_hat)
    print('mcc:', matthews_corrcoef(y.cpu().numpy(), y_hat.cpu().numpy()))
    return matthews_corrcoef(y.cpu().numpy(), y_hat.cpu().numpy())


def last_k_ppl(logits, y, seq_len=1024, k=None):
    '''
    Calculate perplexity for last k tokens in a sequence.

    logits: (batch_size * seq_len, vocab_size), note, already flattened
    y: (batch_size * seq_len), note, already flattened
    seq_len: int, length of each sequence in the batch
    k: if None, use all tokens in sequence
    
    returns: (batch_size,)  ppl for each sequence in the batch
    '''

    if k is None:
        k = 0  # use the entire sequence

    # need to reshape logits and y to be (batch_size, seq_len, vocab_size) and (batch_size, seq_len)
    # respectively
    # breakpoint()
    logits = logits.view(-1, seq_len, logits.shape[-1])
    y = y.view(-1, seq_len)

    # only use the last k values of seq dim in logits and y
    logits = logits[:, -k:, :]
    y = y[:, -k:]

    # reshape to flatten the batch and seq_len dimensions
    logits = logits.reshape(-1, logits.shape[-1])
    y = y.reshape(-1)
    # get avg and put on cpu
    return F.cross_entropy(logits, y, reduction='none').view(y.shape[0], -1).mean().exp().cpu()


def _student_t_map(mu, sigma, nu):
    sigma = F.softplus(sigma)
    nu = 2.0 + F.softplus(nu)
    return mu.squeeze(axis=-1), sigma.squeeze(axis=-1), nu.squeeze(axis=-1)


def student_t_loss(outs, y):
    mu, sigma, nu = outs[..., 0], outs[..., 1], outs[..., 2]
    mu, sigma, nu = _student_t_map(mu, sigma, nu)
    y = y.squeeze(axis=-1)

    nup1_half = (nu + 1.0) / 2.0
    part1 = 1.0 / nu * torch.square((y - mu) / sigma)
    Z = (
            torch.lgamma(nup1_half)
            - torch.lgamma(nu / 2.0)
            - 0.5 * torch.log(math.pi * nu)
            - torch.log(sigma)
    )

    ll = Z - nup1_half * torch.log1p(part1)
    return -ll.mean()


def gaussian_ll_loss(outs, y):
    mu, sigma = outs[..., 0], outs[..., 1]
    y = y.squeeze(axis=-1)
    sigma = F.softplus(sigma)
    ll = -1.0 * (
            torch.log(sigma)
            + 0.5 * math.log(2 * math.pi)
            + 0.5 * torch.square((y - mu) / sigma)
    )
    return -ll.mean()


def binary_cross_entropy(logits, y):
    # BCE loss requires squeezing last dimension of logits so it has the same shape as y
    # requires y to be float, since it's overloaded to represent a probability
    logits = logits.reshape(y.shape)
    return F.binary_cross_entropy_with_logits(logits, y.float())


def binary_accuracy(logits, y):
    return torch.eq(logits.squeeze(-1) >= 0, y).float().mean()


def padded_cross_entropy(logits, y, pad_mask, pad_value=-1):
    """Will ignore the pad value in label (eg, -1)
    
    logits: (batch_size, seq_len, vocab_size)
    y: (batch_size, seq_len)
    pad_mask: (batch_size, seq_len)
    
    """

    # need to apply pad mask to y
    y_pad = y + pad_mask * pad_value

    logits = logits.view(-1, logits.shape[-1])
    y_pad = y_pad.view(-1)
    return F.cross_entropy(logits, y_pad, ignore_index=pad_value)


def cross_entropy(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)

def cross_entropy_no_update(logits, y, ignore_index=-100):
    # 只做监控用的 CE（不建图，不反传）
    with torch.no_grad():
        _ = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            y.view(-1),
            ignore_index=ignore_index
        )
    # 返回占位 0-loss：可 backward，但不依赖模型参数 → 不会更新
    return torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)


def soft_cross_entropy(logits, y, label_smoothing=0.0):
    logits = logits.view(-1, logits.shape[-1])
    # target is now 2d (no target flattening)
    return F.cross_entropy(logits, y, label_smoothing=label_smoothing)


def accuracy(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.eq(preds, y).float().mean()


def accuracy_ignore_index(logits, y, ignore_index=-100):
    num_classes = logits.shape[-1]
    preds = torch.argmax(logits, dim=-1)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    accuracy = tm_f.classification.accuracy(preds, y, 'multiclass', num_classes=num_classes, ignore_index=ignore_index,
                                            average='micro')
    return accuracy


def accuracy_at_k(logits, y, k=1):
    logits = logits.view(-1, logits.shape[-1])
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.topk(logits, k, dim=-1)[1].eq(y.unsqueeze(-1)).any(dim=-1).float().mean()


def f1_binary(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="binary")


def f1_macro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="macro")


def f1_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="micro")


# def roc_auc_macro(logits, y):
#     logits = logits.view(
#         -1, logits.shape[-1]
#     ).detach()  # KS: had to add detach to eval while training
#     y = y.view(-1)
#     return roc_auc_score(
#         y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="macro"
#     )


def roc_auc_macro(logits, y):
    logits = logits.view(
        -1, logits.shape[-1]
    ).detach()  # KS: had to add detach to eval while training
    y = y.view(-1)
    return roc_auc_score(
        y.cpu().numpy(), F.softmax(logits, dim=-1).view(-1).cpu().numpy(), average="macro"
    )
###################################################################





###################################################################


def roc_auc_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return roc_auc_score(
        y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="micro"
    )

# 染色质图谱多标签预测任务的 AUC
def roc_auc_multilabel_macro(logits, y):
    # logits: (N, L); y: (N, L) in {0,1}
    y_true = y.detach().cpu().numpy()
    y_score = torch.sigmoid(logits).detach().cpu().numpy()
    # 某些标签在该批中可能只有单一取值，AUC 不定义；跳过这些标签再做平均
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if np.unique(col).size < 2:  # 全 0 或全 1，跳过
            continue
        aucs.append(roc_auc_score(col, y_score[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def bert_cross_entropy(x, y):
    # print('x in bert_cross_entropy:', x)
    # print('y in bert_cross_entropy:', y)
    # print('y.shape in bert_cross_entropy:', y.shape)
    logits = x[0]
    mask = x[1].reshape(y.shape)
    logits = logits[mask]
    y = y[mask]
    return F.cross_entropy(logits, y)


# def bert_cross_entropy_with_semantic_and_dual(hyout, labels):
#     # 解构
#     logits, mask, total_aux = hyout
#
#     # 1) 标准 LM CE
#     mask_flat = mask.reshape(labels.shape)
#     flat_logits = logits[mask_flat]  # shape [N, C]
#     flat_labels = labels[mask_flat]  # shape [N]
#     ce = F.cross_entropy(flat_logits, flat_labels)
#     print(f'CE {ce.item():.3f}')
#
#     total_loss = ce + total_aux
#
#     return total_loss

def _topk_acc(flat_logits: torch.Tensor, flat_labels: torch.Tensor, k: int = 1) -> float:
    if flat_logits.numel() == 0:
        return float("nan")
    with torch.no_grad():
        topk = flat_logits.topk(k, dim=-1).indices  # [N, k]
        ok = topk.eq(flat_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return ok


def make_complement_perm(C=5, device=None, dtype=torch.float32):
    # A=0,C=1,G=2,T=3,N=4  ->  T,A,C,G,N
    perm = torch.tensor([3, 0, 2, 1, 4], device=device)
    P = torch.zeros(C, C, device=device, dtype=dtype)
    P[torch.arange(C, device=device), perm] = 1.0
    return P, perm



def bert_cross_entropy_with_semantic_and_dual(hyout, labels, report_topk=(1, 3)):
    """
    hyout.logits:
      - 新格式: (fused_logits, mask, total_aux, logits_A_only, logits_B_only, mask_A_rc, mask_B_rc)
    labels : [B, L] or [N]
    """
    pack = hyout
    if hasattr(pack, "logits"):
        pack = pack.logits

    # 解包（兼容不同返回格式）
    fused_logits, mask, total_aux = pack[0], pack[1], pack[2]
    logits_A_only = pack[3] if len(pack) > 3 else None
    logits_B_only = pack[4] if len(pack) > 4 else None
    mask_A_rc = pack[5] if len(pack) > 5 else None
    mask_B_rc = pack[6] if len(pack) > 6 else None

    # ------- 规范化 mask -------
    if mask is None:
        # 没有 mask：全 True
        if fused_logits.dim() == 3:
            B, L, _ = fused_logits.shape
            mask_bool = torch.ones(B, L, dtype=torch.bool, device=fused_logits.device)
        else:
            N, _ = fused_logits.shape
            mask_bool = torch.ones(N, dtype=torch.bool, device=fused_logits.device)
    else:
        mask_bool = mask.to(dtype=torch.bool)

    # ------- 展平 fused 到 [N,C] 并对齐标签 -------
    if fused_logits.dim() == 3:
        B, L, C = fused_logits.shape
        fused_all = fused_logits.reshape(B * L, C)
        labels_all = labels.reshape(B * L)
        mask_flat = mask_bool.view(-1)
        flat_fused = fused_all[mask_flat]
        flat_labels = labels_all[mask_flat]
        keep_rate = float(mask_flat.float().mean().item())
    elif fused_logits.dim() == 2:
        N, C = fused_logits.shape
        flat_fused = fused_logits
        labels_all = labels.reshape(-1)
        if mask_bool.numel() == labels_all.numel():
            # mask 与 [B*L] 对齐，说明 fused 还没按 mask 过滤
            mask_flat = mask_bool.reshape(-1)
            flat_fused = fused_logits[mask_flat]
            flat_labels = labels_all[mask_flat]
            keep_rate = float(mask_flat.float().mean().item())
        else:
            # fused 已按 mask 过滤成 [N,C]
            flat_labels = labels_all[:N]
            keep_rate = float(N / max(1, labels_all.numel()))
    else:
        raise ValueError("Unexpected fused_logits shape")

    # ------- 交叉熵（主头） -------
    ce = F.cross_entropy(flat_fused, flat_labels) if flat_labels.numel() else flat_fused.new_zeros(())
    ppl = math.exp(ce.item()) if torch.isfinite(ce) else float("inf")

    # ------- 指标：fused -------
    msgs = [f"CE {ce.item():.3f}", f"ppl_masked {ppl:.3f}", f"mask_rate {keep_rate:.2%}"]
    for k in report_topk:
        msgs.append(f"Acc@{k} fused {_topk_acc(flat_fused, flat_labels, k) * 100:.2f}%")

    # ------- 指标：A / B（关键：RC 段换成互补标签） -------
    if (logits_A_only is not None) and (logits_B_only is not None) and (mask_A_rc is not None) and (
            logits_A_only.dim() == 3):
        BA, LA, CA = logits_A_only.shape
        # 统一展平
        mask2d = mask_bool.view(BA, LA)
        keep = mask2d.reshape(-1)

        # 准备正向与互补标签
        _, perm = make_complement_perm(CA, device=labels.device)
        labels2d = labels.view(BA, LA)
        labels_comp2d = perm[labels2d]

        # A：按位选标签
        labels_A_2d = torch.where(mask_A_rc, labels_comp2d, labels2d)
        flat_A = logits_A_only.view(BA * LA, CA)[keep]
        flat_labels_A = labels_A_2d.reshape(-1)[keep]

        # B：按位选标签
        labels_B_2d = torch.where(mask_B_rc, labels_comp2d, labels2d)
        flat_B = logits_B_only.view(BA * LA, CA)[keep]
        flat_labels_B = labels_B_2d.reshape(-1)[keep]

        for k in report_topk:
            msgs.append(
                f"A {_topk_acc(flat_A, flat_labels_A, k) * 100:.2f}% "
                f"B {_topk_acc(flat_B, flat_labels_B, k) * 100:.2f}% (top-{k})"
            )

    print(" | ".join(msgs))

    return ce + total_aux


def bert_cross_entropy_with_semantic_and_dual_dankai(hyout, labels, report_topk=(1, 3)):
    """
    hyout.logits:
      - 新格式: (fused_logits, mask, total_aux, logits_A_only, logits_B_only, mask_A_rc, mask_B_rc[, step])
    labels : [B, L] or [N]
    """
    pack = hyout
    if hasattr(pack, "logits"):
        pack = pack.logits

    # ---------------- 解包（兼容不同返回格式） ----------------
    fused_logits, mask, total_aux = pack[0], pack[1], pack[2]
    logits_A_only = pack[3] if len(pack) > 3 else None
    logits_B_only = pack[4] if len(pack) > 4 else None
    mask_A_rc     = pack[5] if len(pack) > 5 else None
    mask_B_rc     = pack[6] if len(pack) > 6 else None
    step_from_pack = pack[7] if len(pack) > 7 else None  # 若模型 forward 传了 step，这里可用

    # ---------------- 规范化 mask ----------------
    if mask is None:
        if fused_logits.dim() == 3:
            B, L, _ = fused_logits.shape
            mask_bool = torch.ones(B, L, dtype=torch.bool, device=fused_logits.device)
        else:
            N, _ = fused_logits.shape
            mask_bool = torch.ones(N, dtype=torch.bool, device=fused_logits.device)
    else:
        mask_bool = mask.to(dtype=torch.bool)

    # ---------------- 展平 fused 到 [N,C] 并对齐标签 ----------------
    if fused_logits.dim() == 3:
        B, L, C = fused_logits.shape
        fused_all = fused_logits.reshape(B * L, C)
        labels_all = labels.reshape(B * L)
        mask_flat = mask_bool.view(-1)
        flat_fused = fused_all[mask_flat]
        flat_labels = labels_all[mask_flat]
        keep_rate = float(mask_flat.float().mean().item())
    elif fused_logits.dim() == 2:
        N, C = fused_logits.shape
        flat_fused = fused_logits
        labels_all = labels.reshape(-1)
        if mask_bool.numel() == labels_all.numel():
            mask_flat = mask_bool.reshape(-1)
            flat_fused = fused_logits[mask_flat]
            flat_labels = labels_all[mask_flat]
            keep_rate = float(mask_flat.float().mean().item())
        else:
            flat_labels = labels_all[:N]
            keep_rate = float(N / max(1, labels_all.numel()))
    else:
        raise ValueError("Unexpected fused_logits shape")

    # ---------------- 交叉熵（主头） ----------------
    ce = F.cross_entropy(flat_fused, flat_labels) if flat_labels.numel() else flat_fused.new_zeros(())
    ppl = math.exp(ce.item()) if torch.isfinite(ce) else float("inf")

    # ---------------- Acc@1（fused） ----------------
    if flat_labels.numel():
        acc1_val = (flat_fused.argmax(dim=-1) == flat_labels).float().mean().item()
    else:
        acc1_val = float("nan")

    # ---------------- 指标打印（fused + 可选 A/B） ----------------
    def _topk_acc(logits, gold, k):
        if gold.numel() == 0:
            return 0.0
        topk = logits.topk(k, dim=-1).indices
        return (topk == gold.unsqueeze(-1)).any(dim=-1).float().mean().item()

    msgs = [f"CE {ce.item():.3f}", f"ppl_masked {ppl:.3f}", f"mask_rate {keep_rate:.2%}"]
    for k in report_topk:
        msgs.append(f"Acc@{k} fused {_topk_acc(flat_fused, flat_labels, k) * 100:.2f}%")

    if (logits_A_only is not None) and (logits_B_only is not None) and (mask_A_rc is not None) and (
            logits_A_only.dim() == 3):
        BA, LA, CA = logits_A_only.shape
        mask2d = mask_bool.view(BA, LA)
        keep = mask2d.reshape(-1)
        # 互补标签
        _, perm = make_complement_perm(CA, device=labels.device)
        labels2d = labels.view(BA, LA)
        labels_comp2d = perm[labels2d]
        # A
        labels_A_2d = torch.where(mask_A_rc, labels_comp2d, labels2d)
        flat_A = logits_A_only.view(BA * LA, CA)[keep]
        flat_labels_A = labels_A_2d.reshape(-1)[keep]
        # B
        labels_B_2d = torch.where(mask_B_rc, labels_comp2d, labels2d)
        flat_B = logits_B_only.view(BA * LA, CA)[keep]
        flat_labels_B = labels_B_2d.reshape(-1)[keep]
        for k in report_topk:
            msgs.append(
                f"A {_topk_acc(flat_A, flat_labels_A, k) * 100:.2f}% "
                f"B {_topk_acc(flat_B, flat_labels_B, k) * 100:.2f}% (top-{k})"
            )

    print(" | ".join(msgs))

    # ---------------- 写入 txt：step, CE, Acc@1（单卡，去重） ----------------
    try:
        # 1) step_id：优先用模型 forward 传来的 step；否则在本函数内部自增
        if step_from_pack is not None:
            step_id = int(step_from_pack) if isinstance(step_from_pack, int) else int(step_from_pack.item())
        else:
            if not hasattr(bert_cross_entropy_with_semantic_and_dual, "_step"):
                bert_cross_entropy_with_semantic_and_dual._step = 0
            bert_cross_entropy_with_semantic_and_dual._step += 1
            step_id = bert_cross_entropy_with_semantic_and_dual._step

        log_path = "/gpfs/essfs/zhaol/projects/Cross_Hnet/src/models/CrossDNA/crossdna_log/cross_hnet_ce_acc1_blocksize2048_bs75_epoch60.txt"

        # 2) 去重：同一个 step 只写一次（有些循环会在同一 step 调两次 loss）
        if getattr(bert_cross_entropy_with_semantic_and_dual, "_last_written_step", None) == step_id:
            pass
        else:
            bert_cross_entropy_with_semantic_and_dual._last_written_step = step_id

            # 3) 确保目录存在
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

            # 4) 首次写入：覆盖表头，其后追加
            if not hasattr(bert_cross_entropy_with_semantic_and_dual, "_ceacc_header_written"):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("step\tce\tacc1\n")
                bert_cross_entropy_with_semantic_and_dual._ceacc_header_written = True
                try:
                    print(f"[ce-acc1] logging to: {os.path.abspath(log_path)}")
                except Exception:
                    pass

            # 5) 追加一行
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{int(step_id)}\t{float(ce.item()):.6f}\t{float(acc1_val):.6f}\n")

    except Exception as e:
        if not hasattr(bert_cross_entropy_with_semantic_and_dual, "_log_err"):
            print(f"[warn] write CE/Acc@1 txt failed: {e}")
            bert_cross_entropy_with_semantic_and_dual._log_err = True

    return ce + total_aux


def bert_cross_entropy_with_DualStrandNoCrossNoTokenBridge(hyout, labels, report_topk=(1, 3)):
    """
    hyout.logits:
      - 新格式: (fused_logits, mask, total_aux, logits_A_only, logits_B_only, mask_A_rc, mask_B_rc[, step])
    labels : [B, L] or [N]
    """
    pack = hyout
    if hasattr(pack, "logits"):
        pack = pack.logits

    # ---------------- 解包（兼容不同返回格式） ----------------
    fused_logits, mask, total_aux = pack[0], pack[1], pack[2]
    logits_A_only = pack[3] if len(pack) > 3 else None
    logits_B_only = pack[4] if len(pack) > 4 else None
    mask_A_rc     = pack[5] if len(pack) > 5 else None
    mask_B_rc     = pack[6] if len(pack) > 6 else None
    step_from_pack = pack[7] if len(pack) > 7 else None  # 若模型 forward 传了 step，这里可用

    # ---------------- 规范化 mask ----------------
    if mask is None:
        if fused_logits.dim() == 3:
            B, L, _ = fused_logits.shape
            mask_bool = torch.ones(B, L, dtype=torch.bool, device=fused_logits.device)
        else:
            N, _ = fused_logits.shape
            mask_bool = torch.ones(N, dtype=torch.bool, device=fused_logits.device)
    else:
        mask_bool = mask.to(dtype=torch.bool)

    # ---------------- 展平 fused 到 [N,C] 并对齐标签 ----------------
    if fused_logits.dim() == 3:
        B, L, C = fused_logits.shape
        fused_all = fused_logits.reshape(B * L, C)
        labels_all = labels.reshape(B * L)
        mask_flat = mask_bool.view(-1)
        flat_fused = fused_all[mask_flat]
        flat_labels = labels_all[mask_flat]
        keep_rate = float(mask_flat.float().mean().item())
    elif fused_logits.dim() == 2:
        N, C = fused_logits.shape
        flat_fused = fused_logits
        labels_all = labels.reshape(-1)
        if mask_bool.numel() == labels_all.numel():
            mask_flat = mask_bool.reshape(-1)
            flat_fused = fused_logits[mask_flat]
            flat_labels = labels_all[mask_flat]
            keep_rate = float(mask_flat.float().mean().item())
        else:
            flat_labels = labels_all[:N]
            keep_rate = float(N / max(1, labels_all.numel()))
    else:
        raise ValueError("Unexpected fused_logits shape")

    # ---------------- 交叉熵（主头） ----------------
    ce = F.cross_entropy(flat_fused, flat_labels) if flat_labels.numel() else flat_fused.new_zeros(())
    ppl = math.exp(ce.item()) if torch.isfinite(ce) else float("inf")

    # ---------------- Acc@1（fused） ----------------
    if flat_labels.numel():
        acc1_val = (flat_fused.argmax(dim=-1) == flat_labels).float().mean().item()
    else:
        acc1_val = float("nan")

    # ---------------- 指标打印（fused + 可选 A/B） ----------------
    def _topk_acc(logits, gold, k):
        if gold.numel() == 0:
            return 0.0
        topk = logits.topk(k, dim=-1).indices
        return (topk == gold.unsqueeze(-1)).any(dim=-1).float().mean().item()

    msgs = [f"CE {ce.item():.3f}", f"ppl_masked {ppl:.3f}", f"mask_rate {keep_rate:.2%}"]
    for k in report_topk:
        msgs.append(f"Acc@{k} fused {_topk_acc(flat_fused, flat_labels, k) * 100:.2f}%")

    if (logits_A_only is not None) and (logits_B_only is not None) and (mask_A_rc is not None) and (
            logits_A_only.dim() == 3):
        BA, LA, CA = logits_A_only.shape
        mask2d = mask_bool.view(BA, LA)
        keep = mask2d.reshape(-1)
        # 互补标签
        _, perm = make_complement_perm(CA, device=labels.device)
        labels2d = labels.view(BA, LA)
        labels_comp2d = perm[labels2d]
        # A
        labels_A_2d = torch.where(mask_A_rc, labels_comp2d, labels2d)
        flat_A = logits_A_only.view(BA * LA, CA)[keep]
        flat_labels_A = labels_A_2d.reshape(-1)[keep]
        # B
        labels_B_2d = torch.where(mask_B_rc, labels_comp2d, labels2d)
        flat_B = logits_B_only.view(BA * LA, CA)[keep]
        flat_labels_B = labels_B_2d.reshape(-1)[keep]
        for k in report_topk:
            msgs.append(
                f"A {_topk_acc(flat_A, flat_labels_A, k) * 100:.2f}% "
                f"B {_topk_acc(flat_B, flat_labels_B, k) * 100:.2f}% (top-{k})"
            )

    print(" | ".join(msgs))

    # ---------------- 写入 txt：step, CE, Acc@1（单卡，去重） ----------------
    try:
        # 1) step_id：优先用模型 forward 传来的 step；否则在本函数内部自增
        if step_from_pack is not None:
            step_id = int(step_from_pack) if isinstance(step_from_pack, int) else int(step_from_pack.item())
        else:
            if not hasattr(bert_cross_entropy_with_semantic_and_dual, "_step"):
                bert_cross_entropy_with_semantic_and_dual._step = 0
            bert_cross_entropy_with_semantic_and_dual._step += 1
            step_id = bert_cross_entropy_with_semantic_and_dual._step

        log_path = "/gpfs/essfs/zhaol/projects/Cross_Hnet/src/models/CrossDNA/crossdna_notokenbridge_baseline_log/cross_hnet_notokenbridge_ce_acc1_blocksize2048_bs75_epoch60.txt"

        # 2) 去重：同一个 step 只写一次（有些循环会在同一 step 调两次 loss）
        if getattr(bert_cross_entropy_with_semantic_and_dual, "_last_written_step", None) == step_id:
            pass
        else:
            bert_cross_entropy_with_semantic_and_dual._last_written_step = step_id

            # 3) 确保目录存在
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

            # 4) 首次写入：覆盖表头，其后追加
            if not hasattr(bert_cross_entropy_with_semantic_and_dual, "_ceacc_header_written"):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("step\tce\tacc1\n")
                bert_cross_entropy_with_semantic_and_dual._ceacc_header_written = True
                try:
                    print(f"[ce-acc1] logging to: {os.path.abspath(log_path)}")
                except Exception:
                    pass

            # 5) 追加一行
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{int(step_id)}\t{float(ce.item()):.6f}\t{float(acc1_val):.6f}\n")

    except Exception as e:
        if not hasattr(bert_cross_entropy_with_semantic_and_dual, "_log_err"):
            print(f"[warn] write CE/Acc@1 txt failed: {e}")
            bert_cross_entropy_with_semantic_and_dual._log_err = True

    return ce + total_aux


def _topk_acc_(logits: torch.Tensor, gold: torch.Tensor, k: int) -> float:
    if gold.numel() == 0: 
        return 0.0
    topk = logits.topk(k, dim=-1).indices
    correct = (topk == gold.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return correct

def bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly(
    hyout, labels, report_topk=(1, 3),
    label_smoothing: float = 0.18,
    log_path: str = "/gpfs/essfs/zhaol/projects/Cross_Hnet/src/models/CrossDNA/crossdna_dual_nocross_baseline_log/DualStrandNoCrossNoTeacher_ce_acc1_blocksize2048_bs75_epoch60_head8_hiddensize128_onlyce.txt",
):
    import os, math, torch
    import torch.nn.functional as F

    # ---- 解包 ----
    pack = hyout.logits if hasattr(hyout, "logits") else hyout
    fused_logits, mask, total_aux = pack[0], pack[1], pack[2]
    step_from_pack = pack[7] if len(pack) > 7 else None  # 兼容：若未来 forward 附加 step

    # ---- mask ----
    if mask is None:
        if fused_logits.dim() == 3:
            B, L, _ = fused_logits.shape
            mask_bool = torch.ones(B, L, dtype=torch.bool, device=fused_logits.device)
        else:
            N, _ = fused_logits.shape
            mask_bool = torch.ones(N, dtype=torch.bool, device=fused_logits.device)
    else:
        mask_bool = mask.to(torch.bool)

    # ---- 展平到 [N, C] 并对齐标签 ----
    C = fused_logits.shape[-1]
    if fused_logits.dim() == 3:
        B, L, C = fused_logits.shape
        fused_all  = fused_logits.reshape(B * L, C)
        labels_all = labels.reshape(B * L)
        mask_flat  = mask_bool.reshape(-1)
        flat_fused  = fused_all[mask_flat]
        flat_labels = labels_all[mask_flat]
        keep_rate   = float(mask_flat.float().mean().item())
    else:
        N, C = fused_logits.shape
        labels_all = labels.reshape(-1)
        if mask_bool.numel() == labels_all.numel():
            mask_flat  = mask_bool.reshape(-1)
            flat_fused = fused_logits[mask_flat]
            flat_labels= labels_all[mask_flat]
            keep_rate  = float(mask_flat.float().mean().item())
        else:
            flat_fused  = fused_logits
            flat_labels = labels_all[:N]
            keep_rate   = float(N / max(1, labels_all.numel()))

    # ---- 主 CE ----
    ce = F.cross_entropy(flat_fused, flat_labels, label_smoothing=label_smoothing) if flat_labels.numel() \
         else flat_fused.new_zeros(())
    ppl = math.exp(ce.item()) if torch.isfinite(ce) else float("inf")

    # ---- 指标（no grad）----
    with torch.no_grad():
        def _topk_acc(x, y, k):
            if y.numel()==0: return 0.0
            topk = x.topk(k, dim=-1).indices
            return (topk == y.unsqueeze(-1)).any(dim=-1).float().mean().item()
        acc1 = _topk_acc(flat_fused, flat_labels, 1)
        acc3 = _topk_acc(flat_fused, flat_labels, 3)
        print(f"CE {ce.item():.3f} | ppl_masked {ppl:.3f} | mask_rate {keep_rate:.2%} | "
              f"Acc@1 fused {acc1*100:.2f}% | Acc@3 fused {acc3*100:.2f}%")

    # ---- step（模型未传则本地自增）----
    if step_from_pack is None:
        if not hasattr(bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly, "_step"):
            bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly._step = 0
        bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly._step += 1
        step_i = bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly._step
    else:
        step_i = int(step_from_pack) if isinstance(step_from_pack, int) else int(step_from_pack.item())

    # ---- 写 txt（同一步只写一次）----
    try:
        if getattr(bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly, "_last_written_step", None) != step_i:
            bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly._last_written_step = step_i
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            if not hasattr(bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly, "_header_written"):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("step\tce\tacc1\n")
                bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly._header_written = True
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step_i}\t{float(ce.item()):.6f}\t{float(acc1):.6f}\n")
    except Exception as e:
        if not hasattr(bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly, "_log_err"):
            print(f"[warn] write CE/Acc@1 txt failed: {e}")
            bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly._log_err = True

    # ---- 返回（保留 total_aux 兼容性；通常为 0）----
    return ce + total_aux



def bert_cross_entropy_with_DualStrandNoTeacher(hyout, labels, report_topk=(1, 3)):
    """
    期望 hyout.logits 结构：
      (fused_logits, mask, total_aux, logits_A_only, logits_B_only, mask_A_rc, mask_B_rc, step?)
    其中 step 可选（模型 forward 附加）。
    """
    pack = hyout
    if hasattr(pack, "logits"):
        pack = pack.logits

    # ------- 解包 -------
    fused_logits, mask, total_aux = pack[0], pack[1], pack[2]
    logits_A_only = pack[3] if len(pack) > 3 else None
    logits_B_only = pack[4] if len(pack) > 4 else None
    mask_A_rc     = pack[5] if len(pack) > 5 else None
    mask_B_rc     = pack[6] if len(pack) > 6 else None
    step_from_pack = pack[7] if len(pack) > 7 else None

    # ------- 规范化 mask -------
    if mask is None:
        if fused_logits.dim() == 3:
            B, L, _ = fused_logits.shape
            mask_bool = torch.ones(B, L, dtype=torch.bool, device=fused_logits.device)
        else:
            N, _ = fused_logits.shape
            mask_bool = torch.ones(N, dtype=torch.bool, device=fused_logits.device)
    else:
        mask_bool = mask.to(dtype=torch.bool)

    # ------- 展平 fused 到 [N,C] 并对齐标签 -------
    if fused_logits.dim() == 3:
        B, L, C = fused_logits.shape
        fused_all = fused_logits.reshape(B * L, C)
        labels_all = labels.reshape(B * L)
        mask_flat = mask_bool.view(-1)
        flat_fused = fused_all[mask_flat]
        flat_labels = labels_all[mask_flat]
        keep_rate = float(mask_flat.float().mean().item())
    elif fused_logits.dim() == 2:
        N, C = fused_logits.shape
        labels_all = labels.reshape(-1)
        if mask_bool.numel() == labels_all.numel():
            mask_flat = mask_bool.reshape(-1)
            flat_fused = fused_logits[mask_flat]
            flat_labels = labels_all[mask_flat]
            keep_rate = float(mask_flat.float().mean().item())
        else:
            flat_fused = fused_logits
            flat_labels = labels_all[:N]
            keep_rate = float(N / max(1, labels_all.numel()))
    else:
        raise ValueError("Unexpected fused_logits shape")

    # ------- 交叉熵（主头） -------
    ce = F.cross_entropy(flat_fused, flat_labels) if flat_labels.numel() else flat_fused.new_zeros(())
    ppl = math.exp(ce.item()) if torch.isfinite(ce) else float("inf")

    # ------- Acc@1（fused） -------
    if flat_labels.numel():
        acc1_val = (flat_fused.argmax(dim=-1) == flat_labels).float().mean().item()
    else:
        acc1_val = float("nan")

    # ------- 指标打印 -------
    msgs = [f"CE {ce.item():.3f}", f"ppl_masked {ppl:.3f}", f"mask_rate {keep_rate:.2%}"]
    for k in report_topk:
        msgs.append(f"Acc@{k} fused {_topk_acc_(flat_fused, flat_labels, k) * 100:.2f}%")
    print(" | ".join(msgs))

    # ------- 写入 txt：step, ce, acc1（单卡，内联写） -------
    try:
        # 1) 计算 step_id（优先用模型 forward 传来的 step）
        if step_from_pack is not None:
            step_id = int(step_from_pack) if isinstance(step_from_pack, int) else int(step_from_pack.item())
        else:
            if not hasattr(bert_cross_entropy_with_DualStrandNoTeacher, "_step"):
                bert_cross_entropy_with_DualStrandNoTeacher._step = 0
            bert_cross_entropy_with_DualStrandNoTeacher._step += 1
            step_id = bert_cross_entropy_with_DualStrandNoTeacher._step

        log_path = "/gpfs/essfs/zhaol/projects/Cross_Hnet/src/models/CrossDNA/crossdna_nosd_baseline_log/SSScanDNAHybridModel_NoTeacher_ce_acc1_blocksize2048_bs75_epoch60.txt"

        # 2) 去重：同一个 step 只写一次
        if getattr(bert_cross_entropy_with_DualStrandNoTeacher, "_last_written_step", None) == step_id:
            pass  # 跳过这次重复调用
        else:
            bert_cross_entropy_with_DualStrandNoTeacher._last_written_step = step_id

            # 3) 确保目录存在
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

            # 4) 首次写入：覆盖表头 & 打印路径；其后追加
            if not hasattr(bert_cross_entropy_with_DualStrandNoTeacher, "_ceacc_header_written"):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("step\tce\tacc1\n")
                bert_cross_entropy_with_DualStrandNoTeacher._ceacc_header_written = True
                try:
                    print(f"[ce-acc1] logging to: {os.path.abspath(log_path)}")
                except Exception:
                    pass

            # 5) 追加一行
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{int(step_id)}\t{float(ce.item()):.6f}\t{float(acc1_val):.6f}\n")

    except Exception as e:
        if not hasattr(bert_cross_entropy_with_DualStrandNoTeacher, "_log_err"):
            print(f"[warn] write CE/Acc@1 txt failed: {e}")
            bert_cross_entropy_with_DualStrandNoTeacher._log_err = True

    return ce + total_aux

# deepSea loss fn:
def roc(logits, y):
    if len(logits.shape) == 3:
        logits = logits.squeeze(1)
    return roc_auc_macro(logits, y)
    # for binary classification, roc_auc_macro and roc_auc_micro are the same
    # deepSea is binary


def deepsea_loss(logits, y):
    if len(logits.shape) == 3:
        logits = logits.squeeze(1)
    return binary_cross_entropy(logits, y)


def mse(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.mse_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        # TODO document the use case of this
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.mse_loss(outs_masked, y_masked)


def forecast_rmse(outs, y, len_batch=None):
    # TODO: generalize, currently for Monash dataset
    return torch.sqrt(F.mse_loss(outs, y, reduction='none').mean(1)).mean()


def mae(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.l1_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.l1_loss(outs_masked, y_masked)


def pearsonr_mean(outs, y, len_batch=None):
    metrics = {}
    outs = outs.cpu().detach().numpy()
    y = y.cpu().detach().numpy().reshape(outs.shape)
    for i, label in enumerate(['dev', 'hk']):
        gt = y[:, i]
        p = outs[:, i]
        r = stats.pearsonr(gt, p)[0]
        metrics[f'pearsonr_{label}'] = r
        metrics[f'pearsonr2_{label}'] = r ** 2
    return (metrics['pearsonr_dev'] + metrics['pearsonr_hk']) / 2


def pearsonr_dev(outs, y, len_batch=None):
    outs = outs.cpu().detach().numpy()
    y = y.cpu().detach().numpy().reshape(outs.shape)
    gt = y[:, 0]
    p = outs[:, 0]
    r = stats.pearsonr(gt, p)[0]
    return r


def pearsonr_hk(outs, y, len_batch=None):
    outs = outs.cpu().detach().numpy()
    y = y.cpu().detach().numpy().reshape(outs.shape)
    gt = y[:, 1]
    p = outs[:, 1]
    r = stats.pearsonr(gt, p)[0]
    return r


def customMSE(outs, y, len_batch=None):
    y = y.reshape(outs.shape)
    return torch.mean((outs[:, 0] - y[:, 0]) ** 2) + torch.mean((outs[:, 1] - y[:, 1]) ** 2)


# Metrics that can depend on the loss
def loss(x, y, loss_fn):
    """ This metric may be useful because the training loss may add extra regularization (e.g. weight decay implemented as L2 penalty), while adding this as a metric skips the additional losses """
    return loss_fn(x, y)


def bpb(x, y, loss_fn):
    """ bits per byte (image density estimation, speech generation, char LM) """
    return loss_fn(x, y) / math.log(2)


def ppl(x, y, loss_fn):
    return torch.exp(loss_fn(x, y))


# should have a better way to do this
output_metric_fns = {
    "binary_cross_entropy": binary_cross_entropy,
    "cross_entropy": cross_entropy,
    "cross_entropy_no_update": cross_entropy_no_update,
    "padded_cross_entropy": padded_cross_entropy,
    "binary_accuracy": binary_accuracy,
    "precision": MulticlassPrecision,
    "precision_per_class": PrecisionPerClass,
    "recall": MulticlassRecall,
    "recall_per_class": RecallPerClass,
    "accuracy": accuracy,
    "accuracy_per_class": AccuracyPerClass,
    "accuracy_ignore_index": accuracy_ignore_index,
    'accuracy@3': partial(accuracy_at_k, k=3),
    'accuracy@5': partial(accuracy_at_k, k=5),
    'accuracy@10': partial(accuracy_at_k, k=10),
    "eval_loss": loss,
    "mcc": mcc,
    "mcc_logits": mcc_logits,
    "mse": mse,
    "mae": mae,
    "forecast_rmse": forecast_rmse,
    "f1_binary": f1_binary,
    "f1_macro": f1_macro,
    "f1_micro": f1_micro,
    "roc_auc_macro": roc_auc_macro,
    "roc_auc_micro": roc_auc_micro,
    "roc_auc_multilabel_macro": roc_auc_multilabel_macro,
    "soft_cross_entropy": soft_cross_entropy,  # only for pytorch 1.10+
    "student_t": student_t_loss,
    "gaussian_ll": gaussian_ll_loss,
    "pearsonr_mean": pearsonr_mean,
    "pearsonr_dev": pearsonr_dev,
    "pearsonr_hk": pearsonr_hk,
    "customMSE": customMSE,
    "roc": roc,
    "deepsea_loss": deepsea_loss,
    "bert_cross_entropy": bert_cross_entropy,
    "bert_cross_entropy_with_semantic_and_dual": bert_cross_entropy_with_semantic_and_dual,
    'bert_cross_entropy_with_semantic_and_dual_dankai': bert_cross_entropy_with_semantic_and_dual_dankai,
    "bert_cross_entropy_with_DualStrandNoTeacher": bert_cross_entropy_with_DualStrandNoTeacher,
    "bert_cross_entropy_with_DualStrandNoCrossNoTokenBridge": bert_cross_entropy_with_DualStrandNoCrossNoTokenBridge,
    "bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly": bert_cross_entropy_with_DualStrandNoCrossNoTeacher_CEonly,
}

loss_metric_fns = {
    "loss": loss,
    "bpb": bpb,
    "ppl": ppl,
}
metric_fns = {**output_metric_fns, **loss_metric_fns}  # TODO py3.9
