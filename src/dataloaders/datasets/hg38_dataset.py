
from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np


"""

Dataset for sampling arbitrary intervals from the human genome.

"""


# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
# string_complement_map = {'A': 'T', 'C': 'G', 'P': 'Q', 'Q': 'P', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'p': 'q', 'q': 'p', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
        # max_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False,
        pad_interval = False,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        # self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval        

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}

        for chr_name in self.seqs.keys():
            # remove tail end, might be gibberish code
            # truncate_len = int(len(self.seqs[chr_name]) * 0.9)
            # self.chr_lens[chr_name] = truncate_len
            self.chr_lens[chr_name] = len(self.seqs[chr_name])


    def __call__(self, chr_name, start, end, max_length, return_augs = False):
        """
        max_length passed from dataset, not from init
        """
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        # chromosome_length = len(chromosome)
        chromosome_length = self.chr_lens[chr_name]

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        # checks if not enough sequence to fill up the start to end
        if interval_length < max_length:
            extra_seq = max_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        # Added support!  need to allow shorter seqs
        if interval_length > max_length:
            end = start + max_length

        seq = str(chromosome[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq

class HG38Dataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    
    '''

    def __init__(
        self,
        split,
        bed_file,
        fasta_file,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,  # replace N token with pad token
        pad_interval = False,  # options for different padding
    ):

        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.pad_interval = pad_interval         
        print('bed_file:',bed_file)
        bed_path = Path(bed_file)
        print('bed_path', bed_path)
        assert bed_path.exists(), 'path to .bed file must exist'

        # read bed file
        df_raw = pd.read_csv(str(bed_path), sep = '\t', names=['chr_name', 'start', 'end', 'split'])
        # select only split df
        self.df = df_raw[df_raw['split'] == split]

        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            # max_length = max_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug,
            pad_interval = pad_interval,
        )

    def __len__(self):
        return len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random row from df
        row = self.df.iloc[idx]
        # row = (chr, start, end, split)
        chr_name, start, end = (row[0], row[1], row[2])

        seq = self.fasta(chr_name, start, end, max_length=self.max_length, return_augs=self.return_augs)

        if self.tokenizer_name == 'char':

            seq = self.tokenizer(seq,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        data = seq[:-1].clone()  # remove eos
        target = seq[1:].clone()  # offset by 1, includes eos

        return data, target


def random_mask(seq, mask_token_id, mask_prob=0.15):
    rand = torch.rand(seq.shape)
    
    mask = rand < mask_prob
    
    masked_seq = seq.clone()
    masked_seq[mask] = mask_token_id
    
    return (masked_seq, mask)

def bert_mask(seq, mask_token_id, pad_token_id, vocab_size, mask_prob=0.15, random_token_prob=0.1, unchanged_token_prob=0.1, special_token_ids=None):
    """
    Applies BERT masking strategy to a sequence of BPE tokens.

    Args:
        seq: Input sequence of BPE tokens (shape: [batch_size, seq_length]).
        mask_token_id: ID of the [MASK] token.
        pad_token_id: ID of the padding token.
        vocab_size: Size of the vocabulary.
        mask_prob: Probability of masking a token.
        random_token_prob: Probability of replacing a masked token with a random token.
        unchanged_token_prob: Probability of keeping a masked token unchanged.
    
    Returns:
        A tuple containing:
            - masked_seq: The masked sequence.
            - labels: The ground truth labels for the masked positions.
            - mask: The mask used to identify which tokens were masked.
    """
    # 避免遮蔽padding
    mask = (seq != pad_token_id) & (torch.rand(seq.shape) < mask_prob)

    # 复制一份用于保存label
    labels = seq.clone()
    labels[~mask] = -100  # PyTorch忽略-100的label

    # 生成随机数矩阵
    rand = torch.rand(seq.shape)

    # 80% [MASK]
    indices_masked = mask & (rand < (1 - random_token_prob - unchanged_token_prob))
    seq[indices_masked] = mask_token_id

    # 10% random token
    indices_random = mask & (rand >= (1 - random_token_prob - unchanged_token_prob)) & (rand < (1 - unchanged_token_prob))
    # 生成随机token，排除特殊token
    random_tokens = torch.randint(0, vocab_size, seq.shape, dtype=torch.long)
    
    # 确保不会选到special tokens
    special_token_ids = torch.tensor(special_token_ids)
    while (torch.isin(random_tokens, special_token_ids)).any():
        random_tokens[torch.isin(random_tokens, special_token_ids)] = torch.randint(0, vocab_size, (random_tokens[torch.isin(random_tokens, special_token_ids)].shape[0],), dtype=torch.long)

    seq[indices_random] = random_tokens[indices_random]

    # 10% unchanged
    # 不需要修改seq，因为它已经保持不变

    return (seq, mask, labels)

# class BertHG38Dataset(torch.utils.data.Dataset):

#     '''
#     Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    
#     '''

#     def __init__(
#         self,
#         split,
#         bed_file,
#         fasta_file,
#         max_length,
#         pad_max_length=None,
#         tokenizer=None,
#         tokenizer_name=None,
#         add_eos=False,
#         return_seq_indices=False,
#         shift_augs=None,
#         rc_aug=False,
#         return_augs=False,
#         replace_N_token=False,  # replace N token with pad token
#         pad_interval = False,  # options for different padding
#         use_tokenizer = True,
#         objective = "stdmlm",
#     ):

#         self.max_length = max_length
#         self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
#         self.tokenizer_name = tokenizer_name
#         self.tokenizer = tokenizer
#         self.return_augs = return_augs
#         self.add_eos = add_eos
#         self.replace_N_token = replace_N_token  
#         self.pad_interval = pad_interval   
#         self.use_tokenizer = use_tokenizer 
#         self.objective = objective     

#         print('bed_file in BertHG38Dataset:',bed_file)
#         print(bed_file)
#         bed_path = Path(bed_file)
#         assert bed_path.exists(), 'path to .bed file must exist'

#         # read bed file
#         df_raw = pd.read_csv(str(bed_path), sep = '\t', names=['chr_name', 'start', 'end', 'split'])
#         # select only split df
#         self.df = df_raw[df_raw['split'] == split]
#         print('fasta_file in BertHG38Dataset:',fasta_file)
#         self.fasta = FastaInterval(
#             fasta_file = fasta_file,
#             # max_length = max_length,
#             return_seq_indices = return_seq_indices,
#             shift_augs = shift_augs,
#             rc_aug = rc_aug,
#             pad_interval = pad_interval,
#         )

#     def __len__(self):
#         return len(self.df)

#     def replace_value(self, x, old_value, new_value):
#         return torch.where(x == old_value, new_value, x)

#     def __getitem__(self, idx):
#         """Returns a sequence of specified len"""
#         # sample a random row from df
#         row = self.df.iloc[idx]
#         # row = (chr, start, end, split)
#         chr_name, start, end = (row[0], row[1], row[2])

#         seq = self.fasta(chr_name, start, end, max_length=self.max_length, return_augs=self.return_augs)

#         if self.tokenizer_name == 'char':

#             seq = self.tokenizer(seq,
#                 add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
#                 padding="max_length",
#                 max_length=self.max_length,
#                 truncation=True,
#             )
#             seq = seq["input_ids"]  # get input_ids

#         elif self.tokenizer_name == 'bpe':
#             seq = self.tokenizer(seq, 
#                 # add_special_tokens=False, 
#                 padding="max_length",
#                 max_length=self.pad_max_length,
#                 truncation=True,
#             ) 
#             # get input_ids
#             if self.add_eos:
#                 seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
#             else:
#                 seq = seq["input_ids"][1:-1]  # remove both special tokens
        
#         # convert to tensor
#         seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now
#         if not self.use_tokenizer:
#             seq = seq-7
#             mask = (seq >= 4) | (seq < 0)
#             seq[mask] = 4
#             data = seq
#             target = seq.clone()
#             seq, mask, labels = bert_mask(data, 4, 5, 5, special_token_ids=[4])
#             return bert_mask(data, 4, 5, 5, special_token_ids=[4]), target

#         if self.replace_N_token:
#             # replace N token with a pad token, so we can ignore it in the loss
#             seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

#         data = seq.clone()  # remove eos
#         target = seq.clone()  # offset by 1, includes eos
#         special_token_ids = self.tokenizer.all_special_ids

#         if self.objective=="stdmlm":
#                 return bert_mask(data, self.tokenizer.mask_token_id, self.tokenizer.pad_token_id, self.tokenizer.vocab_size, special_token_ids=special_token_ids), target
#         else:
#             return random_mask(data, self.tokenizer.mask_token_id), target

# v2：DNA 5 类 one-hot
def dna_bert_mask(
    seq: torch.Tensor,
    mask_prob: float = 0.15,
    random_token_prob: float = 0.10,
    unchanged_token_prob: float = 0.10,
    n_token_id: int = 4,
    vocab_size: int = 5,
    special_mask: torch.Tensor = None,
):
    """
    适配 DNA 5 类 one-hot 的 MLM masking。

    约定：
        A=0, C=1, G=2, T=3, N=4
    不再引入 [MASK]/[PAD] 类别。

    规则：
        - 15% eligible positions 被选中做 MLM
        - 80% 置为 N
        - 10% 随机替换成 0~4 中任意 token
        - 10% 保持不变
        - special_mask=True 的位置不参与 masking
    """
    seq = seq.clone()

    if special_mask is None:
        special_mask = torch.zeros_like(seq, dtype=torch.bool)
    else:
        special_mask = special_mask.to(dtype=torch.bool, device=seq.device)

    eligible = ~special_mask

    mlm_mask = eligible & (torch.rand(seq.shape, device=seq.device) < mask_prob)

    labels = seq.clone()
    labels[~mlm_mask] = -100

    rand = torch.rand(seq.shape, device=seq.device)

    p_to_n = 1.0 - random_token_prob - unchanged_token_prob  # 默认 0.8

    # 80% -> replace with N
    indices_to_n = mlm_mask & (rand < p_to_n)
    seq[indices_to_n] = n_token_id

    # 10% -> random token in {0,1,2,3,4}
    indices_random = mlm_mask & (rand >= p_to_n) & (rand < (1.0 - unchanged_token_prob))
    if indices_random.any():
        random_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=seq.shape,
            device=seq.device,
            dtype=seq.dtype,
        )
        seq[indices_random] = random_tokens[indices_random]

    # 10% unchanged -> do nothing
    return seq, mlm_mask, labels

class BertHG38Dataset(torch.utils.data.Dataset):
    """
    修正版：
    - 严格保持模型输入 one-hot 只编码 A/C/G/T/N 五类
    - 不再使用 [MASK]/[PAD] 新 id
    - 通过 mlm_mask 和 special_mask 传递“位置语义”
    """

    COMPACT_A_ID = 0
    COMPACT_C_ID = 1
    COMPACT_G_ID = 2
    COMPACT_T_ID = 3
    COMPACT_N_ID = 4
    COMPACT_VOCAB_SIZE = 5

    def __init__(
        self,
        split,
        bed_file,
        fasta_file,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,
        pad_interval=False,
        use_tokenizer=True,
        objective="stdmlm",
    ):
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token
        self.pad_interval = pad_interval
        self.use_tokenizer = use_tokenizer
        self.objective = objective

        print("bed_file in BertHG38Dataset:", bed_file)
        bed_path = Path(bed_file)
        assert bed_path.exists(), "path to .bed file must exist"

        df_raw = pd.read_csv(
            str(bed_path),
            sep="\t",
            names=["chr_name", "start", "end", "split"]
        )
        self.df = df_raw[df_raw["split"] == split]

        print("fasta_file in BertHG38Dataset:", fasta_file)
        self.fasta = FastaInterval(
            fasta_file=fasta_file,
            return_seq_indices=return_seq_indices,
            shift_augs=shift_augs,
            rc_aug=rc_aug,
            pad_interval=pad_interval,
        )

    def __len__(self):
        return len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def _char_tokenizer_ids_to_compact_dna_ids_and_special_mask(self, seq: torch.Tensor):
        """
        把 CharacterTokenizer 输出映射成 compact DNA ids:
            A=0, C=1, G=2, T=3, N=4

        同时返回 special_mask:
            True 代表该位置原本不是正常 A/C/G/T/N，而是 special/pad/unk 等位置
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer cannot be None when tokenizer_name='char'")

        vocab = self.tokenizer._vocab_str_to_int

        out = torch.full_like(seq, fill_value=self.COMPACT_N_ID)
        special_mask = torch.ones_like(seq, dtype=torch.bool)

        if "A" in vocab:
            out[seq == vocab["A"]] = self.COMPACT_A_ID
            special_mask[seq == vocab["A"]] = False

        if "C" in vocab:
            out[seq == vocab["C"]] = self.COMPACT_C_ID
            special_mask[seq == vocab["C"]] = False

        if "G" in vocab:
            out[seq == vocab["G"]] = self.COMPACT_G_ID
            special_mask[seq == vocab["G"]] = False

        if "T" in vocab:
            out[seq == vocab["T"]] = self.COMPACT_T_ID
            special_mask[seq == vocab["T"]] = False

        if "N" in vocab:
            out[seq == vocab["N"]] = self.COMPACT_N_ID
            special_mask[seq == vocab["N"]] = False

        # 其余特殊 token:
        # [CLS]/[SEP]/[BOS]/[PAD]/[UNK]/[RESERVED]
        # 统一映射成 N=4，但 special_mask=True 保留下来
        return out, special_mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chr_name, start, end = (row[0], row[1], row[2])

        seq = self.fasta(
            chr_name,
            start,
            end,
            max_length=self.max_length,
            return_augs=self.return_augs,
        )

        if self.tokenizer_name == "char":
            seq = self.tokenizer(
                seq,
                add_special_tokens=True if self.add_eos else False,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]

        elif self.tokenizer_name == "bpe":
            seq = self.tokenizer(
                seq,
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            )
            if self.add_eos:
                seq = seq["input_ids"][1:]
            else:
                seq = seq["input_ids"][1:-1]

        seq = torch.LongTensor(seq)

        # =========================================================
        # 推荐的预训练路径：保持 one-hot 只编码 A/C/G/T/N
        # =========================================================
        if not self.use_tokenizer:
            if self.tokenizer_name != "char":
                raise ValueError(
                    "use_tokenizer=False 且严格 ACGTN one-hot 的实现，只支持 tokenizer_name='char'"
                )

            compact_seq, special_mask = self._char_tokenizer_ids_to_compact_dna_ids_and_special_mask(seq)

            target = compact_seq.clone()
            data = compact_seq.clone()

            masked_seq, mlm_mask, labels = dna_bert_mask(
                data,
                mask_prob=0.15,
                random_token_prob=0.10,
                unchanged_token_prob=0.10,
                n_token_id=self.COMPACT_N_ID,         # 被 mask 的 80% 位置用 N 表示
                vocab_size=self.COMPACT_VOCAB_SIZE,   # 5
                special_mask=special_mask,
            )

            # 返回四元组：
            #   masked_seq : [L] 取值仅 0~4
            #   mlm_mask   : [L] bool
            #   labels     : [L] 取值 0~4 或 -100
            #   special_mask: [L] bool
            return (masked_seq, mlm_mask, labels, special_mask), target

        # =========================================================
        # 原 tokenizer 路线：把双重 bert_mask 的 bug 一并修掉
        # =========================================================
        if self.replace_N_token:
            seq = self.replace_value(
                seq,
                self.tokenizer._vocab_str_to_int["N"],
                self.tokenizer.pad_token_id,
            )

        data = seq.clone()
        target = seq.clone()
        special_token_ids = self.tokenizer.all_special_ids

        if self.objective == "stdmlm":
            # 这里只修复“双重 masking” bug
            masked_seq, mask, labels = bert_mask(
                data,
                self.tokenizer.mask_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.vocab_size,
                special_token_ids=special_token_ids,
            )
            return (masked_seq, mask, labels), target
        else:
            return random_mask(data, self.tokenizer.mask_token_id), target