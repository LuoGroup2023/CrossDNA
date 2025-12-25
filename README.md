# Explicit Dynamic Cross-Strand Interactions for DNA Sequence Language Modeling


---

## 🚩 Plan
- [x] Scripts for Pretraining, NT & Genomic Benchmarks.
- [ ] Paper Released.
- [x] Pretrained Weights of CrossDNA (8.1M).
- [x] [[HuggingFace 🤗]](https://huggingface.co/chengCCC) includes variants of the CrossDNA model.
- [x] Source Code and Pretrained Weights on transformers.
---

<h2>1 Quick start</h2>

<h3>1.1 Clone the repo and cd CrossDNA/crossdna.</h3>
<pre>
git clone https://github.com/LuoGroup2023/CrossDNA.git
cd CrossDNA/crossdna
</pre>


<h3>1.2 Prepare conda env.</h3>
<pre>
conda create -n CrossDNA python=3.11
conda activate CrossDNA
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121
pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention --no-deps
pip install --no-cache-dir triton==3.2.0
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-deps "selene_sdk==0.6.0"
pip install -U cython plotly pytabix ruamel.yaml ruamel.yaml.clib seaborn statsmodels narwhals patsy
pip install transformer pytorch-lightning==1.8.6 wandb hydra-core==1.3.2 omegaconf==2.3.0 datasets polars genomic_benchmarks liftover psutil kipoiseq pyBigWig timm
</pre>

<h3>1.3 Download the data.(Pretrain)</h3>
<pre>
  mkdir data
  mkdir -p data/hg38/
  curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > data/hg38/hg38.ml.fa.gz
  gunzip data/hg38/hg38.ml.fa.gz  # unzip the fasta file
  curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > data/hg38/human-sequences.bed
</pre>




You can check out the <a href="https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1">Nucleotide Transformer</a> ang <a href="https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks">Genomic Benchmarks</a> paper for how to download and process NT benchmark & Genomic Benchmark datasets.

The final file structure (data directory) should look like

<pre>
  |____bert_hg38
| |____hg38.ml.fa
| |____hg38.ml.fa.fai
| |____human-sequences.bed
|____nucleotide_transformer
| |____H3K36me3
| |____......
|____genomic_benchmark
| |____dummy_mouse_enhancers_ensembl
| |____....
</pre>


---

<h2>2 Reproducing the paper</h2>

<h3>2.1 Pre-training on the Human Reference Genome</h3>

<pre>
  python train.py experiment='hg38-pretrain/crossdna'
</pre>

you can adjust the hyperparameters by using cmd like following, detailed hyperparameters setting can be seen in configs/experiment/xxx/xxx.yaml
<pre>
  python train.py experiment='hg38-pretrain/crossdna' wandb=null trainer.devices=4
</pre>

<h3>2.2 Genomic Benchmarks (short-range)</h3>
<p>GenomicBenchmarks provides 8 binary- and multi-class tasks packaged as a Python library. </p>

Remeber to adjust the setting for different dataset like max seq length.
<pre>
  python train.py experiment='genomic-benchmark/crossdna' 
</pre>

<h3>2.3 Nucleotide Transformer Benchmark</h3>
<p>Datasets are hosted on the Hub as <code>InstaDeepAI/nucleotide_transformer_downstream_tasks</code>. </p>
Remeber to adjust the setting for different dataset like max seq length.
<pre>
  python train.py experiment='nt-benchmark/crossdna'
</pre>

---

<h2>3 Model Loading and Testing</h2>


<pre>
import os
os.environ.setdefault("DISABLE_TORCH_COMPILE", "1")  

import torch
if hasattr(torch, "compile"):
    def _no_compile(fn=None, *args, **kwargs):
        if fn is None:
            def deco(f): return f
            return deco
        return fn
    torch.compile = _no_compile

from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_DIR = "/home/zhaol/projects/huggingface_crossdna/crossdna_inference" # This directory must contain either the `model.safetensors` or `pytorch_model.bin` file.

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model.eval()

# ---- Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- Base mapping (logits indexed 0..4 <-> A/C/G/T/N)
labels = ["A", "C", "G", "T", "N"]
base_map = {ch: i for i, ch in enumerate(labels)}

def dna_to_base_ids(seq: str, device=None):
    t = torch.tensor([base_map.get(ch.upper(), base_map["N"]) for ch in seq], dtype=torch.long)
    return t.to(device) if device is not None else t

# ========== Test (MaskedLM Forward) ==========
x = torch.full((2, 16), base_map['N'], dtype=torch.long, device=device)
mask_id = getattr(model.config, "mask_token_id", 3)
x[:, 3] = mask_id
x[:, 9] = mask_id

with torch.no_grad():
    out = model(input_ids=x)

logits = out.logits.detach().cpu()
print("logits.shape =", tuple(logits.shape))

dna = "TGATGTGACTCACATAGGCGGTGGCGTGATATGTTGTGACTCATTTCCCGGAAACGGATGACTAATGCCATATGTTATCAGTTTCCTGGAAATTTGATCACGCCATATTGTGAAATCATGCGATTCCCGGATCACGTGACGGCCGGACGTGACAAGTATGAGTCACTAAGTGGCGTGATCTTACGAATCACGTGATGGTCAATGTCACGTGATCGGCTGGTGAGTCAGCAATATCGTGTGATTCATTC"
inp = dna_to_base_ids(dna, device=device).unsqueeze(0)
with torch.no_grad():
    out2 = model(input_ids=inp)
pred2 = out2.logits.argmax(dim=-1).squeeze(0).cpu().tolist()
print("argmax base:", "".join(labels[i] for i in pred2))

# ==========  Representations (Embedding) ==========

max_len_cfg = getattr(model.config, "max_position_embeddings", 1024)
max_length = int(min(512, max_len_cfg))  # You can change it to a longer value, but do not exceed max_position_embeddings.

# 2) Construct sample DNA text with base IDs (note: not tokenizer IDs)
sequence = "ACTG" * (max_length // 4)
seq_base = dna_to_base_ids(sequence, device=device).unsqueeze(0)  # [1, L] in 0..4

# 3) Temporarily switch the backbone to “characterization mode”.
was_pretrain = getattr(model.backbone, "pretrain", False)
was_for_repr = getattr(model.backbone, "for_representation", False)
model.backbone.pretrain = False                 # Let Backbone accept a single sequence
model.backbone.for_representation = True        # Let forward return the fused representation.

with torch.inference_mode():
    embeddings, _ = model.backbone(seq_base)    # [B, L, H]
embeddings_cpu = embeddings.detach().cpu()
print("embeddings.shape =", tuple(embeddings_cpu.shape))

# 4) Restore original settings
model.backbone.pretrain = was_pretrain
model.backbone.for_representation = was_for_repr

</pre>


<h2>4 The dataset for downstream tasks.</h2>

All data used in this study were obtained from publicly available datasets.

For the Genomic Benchmarks tasks, we used datasets hosted on Hugging Face: [https://huggingface.co/katarinagresova](https://huggingface.co/katarinagresova). Data were processed following the procedures described in the associated GitHub repositories: [https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) and [https://github.com/HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna).

The Nucleotide Transformer downstream tasks were downloaded from: [https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks/tree/main](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks/tree/main), and prepared according to the data processing and loading pipeline provided in the Caduceus repository: [https://github.com/kuleshov-group/caduceus](https://github.com/kuleshov-group/caduceus).

Chromatin profile prediction data were obtained from the DeepSEA resource. Preprocessing followed the Sei framework implementation: [https://github.com/FunctionLab/sei-framework](https://github.com/FunctionLab/sei-framework), and task-specific fine-tuning was configured in accordance with the GENA-LM DeepSEA scripts: [https://github.com/AIRI-Institute/GENA_LM/blob/main/downstream_tasks/DeepSea/run_deepsea_finetuning.py](https://github.com/AIRI-Institute/GENA_LM/blob/main/downstream_tasks/DeepSea/run_deepsea_finetuning.py).

For the enhancer activity prediction task, we used the dataset available at: [https://huggingface.co/datasets/GenerTeam/DeepSTARR-enhancer-activity/tree/main](https://huggingface.co/datasets/GenerTeam/DeepSTARR-enhancer-activity/tree/main), and followed the data preprocessing and model fine-tuning procedures described in the associated study.

DNA long-range benchmark tasks were constructed from the dataset available at Harvard Dataverse: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YUP2G5](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YUP2G5), and processed following the JanusDNA repository: [https://github.com/Qihao-Duan/JanusDNA](https://github.com/Qihao-Duan/JanusDNA).

For the experiment evaluating the generalization performance of enhancers, mouse memory CD8 T cell enhancers and Drosophila E2-4 neural enhancers were obtained from the EnhancerAtlas database: [http://www.enhanceratlas.net/scenhancer/download.php](http://www.enhanceratlas.net/scenhancer/download.php). Human K562 cell-line enhancer sequences were also retrieved from EnhancerAtlas. Ten experimentally validated, highly active developmental enhancers designed in the DREAM study were downloaded from the supplementary materials of the corresponding publication: [https://academic.oup.com/nar/article/52/21/13447/7825962#supplementary-data](https://academic.oup.com/nar/article/52/21/13447/7825962#supplementary-data). You can find our processed enhancer dataset via this link: https://doi.org/10.5281/zenodo.17995482 .

To benchmark the embedding quality of DNA foundation models, we used the DNA Foundation Benchmark dataset, available at: [https://huggingface.co/datasets/hfeng3/dna_foundation_benchmark_dataset/tree/main](https://huggingface.co/datasets/hfeng3/dna_foundation_benchmark_dataset/tree/main).


## Contact  
  - **Cheng Yang**: [yangchengyjs@hnu.edu.cn](mailto:[yangchengyjs@hnu.edu.cn)   
  College of Computer Science and Electronic Engineering, Hunan University, Changsha
