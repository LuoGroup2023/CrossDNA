import torch
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import liftover
from pathlib import Path
from pyfaidx import Fasta
from random import randrange, random


def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

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
        rc_aug = False
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        # self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

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
        # print('seq:', seq)
        # print('seq length:', len(seq))

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        seq = ('.' * left_padding) + seq + ('.' * right_padding)
        # seq = ('N' * left_padding) + seq + ('N' * right_padding)

        return seq


class ChromatinProfileDataset(torch.utils.data.Dataset):
    '''
    Recreation of chromatin profile prediction benchmark from BigBird paper https://arxiv.org/abs/2007.14062
    Original sequence coordinates and target labels are provided via a csv.
    Original sequences have a length of 1000. This is changed to be max_length on the fly.
    Target labels are read into a LongTensor. Coordinates are read into a DataFrame with columns "Chr_No" (0-based), "Start" and "End".
    Original coordinates are in hg19 format named as train_hg19_coords_targets.csv etc. 
    Hg19 coordinates will be translated to hg38 if ref_genome_version=='hg38'. 
    The translated coordinated can be saved to a new file e.g. train_hg19_coords_targets.csv so this only needs to be done once.
    Returns a generator that retrieves the sequence.
    '''
    def __init__(
        self,
        max_length,
        ref_genome_path=None,
        ref_genome_version=None,
        coords_target_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        save_liftover=False,
    ):

        self.max_length      = max_length
        assert max_length%2==0 # check window is divisible by 2
        self.use_padding     = use_padding
        self.tokenizer_name  = tokenizer_name
        self.tokenizer       = tokenizer
        self.return_augs     = return_augs
        self.add_eos         = add_eos
        self.rc_aug          = rc_aug
        self.ref_genome_version = ref_genome_version
        
        # self.ref_genome = FastaInterval(fasta_file=ref_genome_path, max_length=self.max_length)
        self.ref_genome = FastaInterval(fasta_file=ref_genome_path)
        
        # Original data coordinates are from hg19. 
        # If ref genome is hg38 and original coordinates are provided these must be translated by liftover.
        # Conversion only needs to be done once so save liftover coordinates to file optionally.
        if self.ref_genome_version=='hg19':
            if 'hg19' in coords_target_path.split('/')[-1]:
                self.load_csv_data(coords_target_path)

            else:
                raise ValueError('Make sure data coordinates are in hg19 format (and put "hg19" in filename)')
        elif self.ref_genome_version=='hg38':
            if 'hg38' in coords_target_path.split('/')[-1]:
                self.load_csv_data(coords_target_path)
            elif 'hg19' in coords_target_path.split('/')[-1]:
                self.load_csv_data(coords_target_path)
                print('ref_genome_version = "hg38" but target coordinates are labelled "hg19"')
                self.convert_coordinates(coords_target_path, save_liftover)
            else:
                raise ValueError('Make sure data coordinates have correct hg19/hg38 in filename')
        else:
            raise ValueError('ref_genome_version must be "hg19" or "hg38"')
             
        # Move start/end to new window 
        # Window = 1000 used in raw coordinate data
        self.coords['Start'] = self.coords['Start']-int((max_length-1000)/2)
        self.coords['End']   = self.coords['End']+int((max_length-1000)/2)
        
    def load_csv_data(self, coords_target_path):
        # Grab sequence coordinates from csv
        self.coords = pd.read_csv(
            coords_target_path, 
            usecols=['Chr_No','Start','End'],
            dtype={'Chr_No':np.int64,'Start':np.int64,'End':np.int64}
        ).reset_index(drop=True) # Note Chr_No is zero-based
        
        # Quickly grab target column names
        with open(coords_target_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.target_columns = [col for col in header if col[:2]=='y_' ]
        
        # 把 919 个标签名保存下来，供 task 使用
        self.feature_names = list(self.target_columns)
            
        # Grab targets from csv and convert to torch long format
        self.targets = torch.from_numpy(
                    pd.read_csv(
                        coords_target_path, 
                        usecols=self.target_columns,
                        dtype={k:bool for k in self.target_columns}
                    ).to_numpy()
                ).long()
        assert len(self.feature_names) == self.targets.shape[1], \
        f"feature_names({len(self.feature_names)}) != targets.shape[1]({self.targets.shape[1]})"


    def __len__(self):
        return len(self.coords)
    
    def _map_chars_to_0_4(self, s: str) -> torch.LongTensor:
        """将字符串序列映射为 0..4（A,C,G,T,N/其它->4），并保证长度为 max_length。"""
        # 统一大写 + 将 '.' 视为 'N'
        s = s.upper().replace('.', 'N')

        # 长度保障（通常 FastaInterval 已经对齐；这里做兜底）
        if len(s) < self.max_length:
            s = s + ('N' * (self.max_length - len(s)))
        elif len(s) > self.max_length:
            s = s[:self.max_length]

        map4 = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        x = torch.tensor([map4.get(ch, 4) for ch in s], dtype=torch.long)
        # 运行期防呆
        assert x.numel() == self.max_length, f"mapped length={x.numel()} != max_length={self.max_length}"
        return x
    
    def _tokenize(self, s: str) -> torch.LongTensor:
        """
        使用 CharacterTokenizer，但强制：
        - add_special_tokens=False（不加 [SEP]）
        - padding/truncation 到 max_length
        - 将 tokenizer 的 id 映射为 0..4：id-7，超界全并 4
        """
        s = s.upper().replace('.', 'N')

        tok = self.tokenizer(
            s,
            add_special_tokens=False,                               # 不加 [SEP]
            padding=("max_length" if self.use_padding else "do_not_pad"),
            max_length=self.max_length,
            truncation=True,
        )
        ids = torch.LongTensor(tok["input_ids"])                    # 长度应等于 max_length（若 padding=True）
                # 将 A/C/G/T/N 从 7..11 平移到 0..4；特殊符号/未知（<7 或 >11）并到 4
        ids = ids - 7
        mask = (ids < 0) | (ids >= 5)
        ids[mask] = 4

        # 长度保障（若未 padding 则做兜底）
        if ids.numel() < self.max_length:
            pad = torch.full((self.max_length - ids.numel(),), 4, dtype=torch.long)
            ids = torch.cat([ids, pad], dim=0)
        elif ids.numel() > self.max_length:
            ids = ids[:self.max_length]

        # 运行期防呆
        assert ids.numel() == self.max_length
        mn, mx = int(ids.min()), int(ids.max())
        assert 0 <= mn and mx < 5, f"ids out of range: [{mn},{mx}]"
        return ids


    def __getitem__(self, idx):
        y = self.targets[idx]
        coord = self.coords.iloc[idx]
        seq = self.ref_genome(
            'chr{}'.format(coord['Chr_No']+1), # Make chromosome id 1-based 
            coord['Start'], 
            coord['End'],
            max_length=self.max_length,
        )
        
        # # apply rc_aug here if using
        # if self.rc_aug and coin_flip():
        #     seq = string_reverse_complement(seq)
        
        # if self.tokenizer==None:
        #     return seq, y
        if self.tokenizer is not None:
            x = self._tokenize(seq)
            # print('x1:', x)
            # print('x1 shape:', x.shape)
        else:
            x = self._map_chars_to_0_4(seq)
        
        return x, y
    
    
    def convert_coordinates(self, coords_target_path, save_liftover):
        '''
        Loop through coordinates and translate from hg19 to hg38.
        Filter entries where liftover fails.
        Save this to file so we only have to do it once.
        '''
        converter = liftover.get_lifter('hg19', 'hg38')
        
        print("Translating coordinates from hg19 to hg38:")
        for i in tqdm(range(len(self.coords))):
            row = self.coords.iloc[i]
            new_start = converter['chr{}'.format(row['Chr_No']+1)][row['Start']]
            new_end = converter['chr{}'.format(row['Chr_No']+1)][row['End']]
            if (len(new_start) == 0) or (len(new_end) == 0):
                # If liftover fails set -999 for filtering
                self.coords.iloc[i]['Start']=-999
            else:
                self.coords.iloc[i]['Start']=new_start[0][1]
                self.coords.iloc[i]['End']=new_end[0][1]
                
        # Filter unmapped coordinates
        n_before = len(self.coords)
        self.coords = self.coords.query('Start!=-999')
        n_after = len(self.coords)
        print('Filtered {} unmapped coordinates. There are {} samples remaining'.format(n_before-n_after, n_after))
        
        # Filter incorrect window sizes
        n_before=n_after
        self.coords = self.coords.query('End-Start==1000')
        n_after = len(self.coords)
        print('Filtered {} incorrect window sizes. There are {} samples remaining'.format(n_before-n_after, n_after))
        
        # Reindex targets based on filtered coordinates and reset coordinate index
        self.targets = self.targets[self.coords.index.to_numpy()]
        self.coords.reset_index(inplace=True, names=['filter_index'])
        
        assert len(self.targets) == len(self.coords) # Sanity check
        
        
        if save_liftover: # save liftover coords in original format and change filename accordingly
            hg38_coords_targets = pd.concat([self.coords, pd.DataFrame(columns=self.target_columns, data=self.targets)], axis=1)
            print('Saving translated and filtered data to {}'.format(coords_target_path.replace('hg19','hg38')))
            hg38_coords_targets.to_csv(coords_target_path.replace('hg19','hg38'))
            del hg38_coords_targets