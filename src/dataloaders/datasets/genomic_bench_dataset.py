from pathlib import Path
from typing import Optional

import torch

from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded

from src.dataloaders.base import default_data_path


def coin_flip(p: float = 0.5) -> bool:
    return bool(torch.rand(()) < p)


_RC_MAP = str.maketrans({
    "A": "T", "C": "G", "G": "C", "T": "A", "N": "N",
    "a": "t", "c": "g", "g": "c", "t": "a", "n": "n",
})


def string_reverse_complement(seq: str) -> str:
    return seq.translate(_RC_MAP)[::-1]


_BASE_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}


def _read_sequence_file(path: Path) -> str:
    """
    Robust reader for GenomicBenchmarks sequence files.
    Removes whitespace/newlines and ignores FASTA-style headers if present.
    """
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            lines.append(line)
    return "".join(lines).upper()


class GenomicBenchmarkDataset(torch.utils.data.Dataset):
    """
    CrossDNA-compatible GenomicBenchmarks dataset.

    Key behavior:
    - base-level input
    - A/C/G/T/N -> 0/1/2/3/4
    - optional RC augmentation
    - return attention mask for padded positions
    - optional legacy tokenizer path preserved for compatibility
    """

    def __init__(
        self,
        split,
        max_length,
        dataset_name="human_enhancers_cohn",
        d_output=2,
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=True,
        use_tokenizer=False,
        token_id_offset: int = 7,
        pad_value_id: int = 4,
        **unused_kwargs,
    ):
        self.max_length = int(max_length)
        self.use_padding = bool(use_padding)
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        self.use_tokenizer = use_tokenizer
        self.token_id_offset = token_id_offset
        self.pad_value_id = pad_value_id

        if (not self.use_tokenizer) and self.add_eos:
            raise ValueError(
                "For CrossDNA fine-tuning with base-level input, "
                "set add_eos=False when use_tokenizer=False."
            )

        if self.use_tokenizer and self.tokenizer is None:
            raise ValueError("use_tokenizer=True requires a valid `tokenizer`.")

        if dest_path is None:
            dest_path = Path(default_data_path) / "genomic_benchmark"
        else:
            dest_path = Path(dest_path)

        if not is_downloaded(dataset_name, cache_path=dest_path):
            print(f"downloading {dataset_name} to {dest_path}")
            download_dataset(dataset_name, version=0, dest_path=dest_path)
        else:
            print(f"already downloaded: {dataset_name}")

        if split == "val":
            split = "test"

        base_path = dest_path / dataset_name / split
        if not base_path.exists():
            raise FileNotFoundError(f"Split path does not exist: {base_path}")

        self.all_seqs = []
        self.all_labels = []

        label_mapper = {}
        label_dirs = sorted([x for x in base_path.iterdir() if x.is_dir()])
        for i, x in enumerate(label_dirs):
            label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            label_dir = base_path / label_type
            for path in sorted(label_dir.iterdir()):
                if not path.is_file():
                    continue
                seq = _read_sequence_file(path)
                self.all_seqs.append(seq)
                self.all_labels.append(label_mapper[label_type])

    def __len__(self):
        return len(self.all_labels)

    def _encode_direct(self, x: str):
        x = x.upper()
        x = x[:self.max_length]
        valid_len = len(x)

        ids_trimmed = torch.tensor(
            [_BASE_TO_ID.get(ch, 4) for ch in x],
            dtype=torch.long,
        )

        if self.use_padding:
            input_ids = torch.full(
                (self.max_length,),
                fill_value=self.pad_value_id,
                dtype=torch.long,
            )
            if valid_len > 0:
                input_ids[:valid_len] = ids_trimmed

            attn_mask = torch.zeros((self.max_length,), dtype=torch.bool)
            if valid_len > 0:
                attn_mask[:valid_len] = True
        else:
            input_ids = ids_trimmed
            attn_mask = torch.ones((valid_len,), dtype=torch.bool)

        return input_ids, attn_mask

    def _encode_with_tokenizer(self, x: str):
        seq = self.tokenizer(
            x,
            add_special_tokens=True if self.add_eos else False,
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = torch.as_tensor(seq["input_ids"], dtype=torch.long)

        ids = input_ids - self.token_id_offset
        ids[(ids < 0) | (ids > 4)] = 4
        input_ids = ids

        attn_mask = torch.as_tensor(
            seq.get("attention_mask", [1] * len(input_ids)),
            dtype=torch.bool,
        )
        return input_ids, attn_mask

    def __getitem__(self, idx):
        x = self.all_seqs[idx]
        y = self.all_labels[idx]

        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        if self.use_tokenizer:
            seq_ids, attn_mask = self._encode_with_tokenizer(x)
        else:
            seq_ids, attn_mask = self._encode_direct(x)

        target = torch.LongTensor([y])

        if self.return_mask:
            return seq_ids, target, {"mask": attn_mask}
        else:
            return seq_ids, target