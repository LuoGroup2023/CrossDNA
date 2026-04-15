# Explicit Dynamic Cross-Strand Interactions for DNA Sequence Language Modeling


---

## Plan
- [x] CrossDNAv2 Scripts for Pretraining, NT & Genomic Benchmarks.
- [ ] Paper Released.
- [x] [[HuggingFace]](https://huggingface.co/chengCCC/CrossDNA_pretrain/tree/main) includes variants of the CrossDNA model.
- [x] Source Code and Pretrained Weights on transformers.
---

<h2>1 Quick start</h2>

<h3>1.1 Clone the repo and switch to the crossdnav2 branch.</h3>
<pre>
git clone -b crossdnav2 https://github.com/LuoGroup2023/CrossDNA.git
cd CrossDNA
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
pip install transformer pytorch-lightning==2.5.0.post0 wandb hydra-core==1.3.2 omegaconf==2.3.0 datasets polars genomic_benchmarks liftover psutil kipoiseq pyBigWig timm
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

<h2>2 Reproducing the paper</h2>

<h3>2.1 Pre-training on the Human Reference Genome</h3>

<p>The recommended entry point is the pre-training script under <code>scripts/pre_train</code>. Before running, please update the environment-specific paths in the script, such as <code>conda</code>, <code>full_path_to_root</code>, and the output directory.</p>
<pre>
  bash scripts/pre_train/CrossDNAv2_2k.sh
</pre>

<p>This script launches <code>experiment=hg38-pretrain/crossdnav2</code> with the default 2k pre-training setup. You can edit variables such as <code>SEQLEN</code>, <code>BLOCK_SIZE</code>, <code>BATCH_SIZE</code>, <code>D_MODEL</code>, <code>Depth</code>, <code>LR</code>, and <code>MAX_EPOCHES</code> directly in the script.</p>

<h3>2.2 Genomic Benchmarks</h3>
<p>GenomicBenchmarks provides 8 binary- and multi-class tasks packaged as a Python library.</p>
<p>The recommended launch script is <code>scripts/benchmark/gb/gb_crossdnav2.sh</code>. Please update the checkpoint path and dataset root in the script, or pass them from the command line.</p>
<pre>
  bash scripts/benchmark/gb/gb_crossdnav2.sh human_enhancers_cohn
</pre>

<p>Optional override:</p>
<pre>
  bash scripts/benchmark/gb/gb_crossdnav2.sh human_enhancers_cohn /path/to/pretrain.ckpt /path/to/genomic_benchmark
</pre>

<p>An additional 408K/tiny-backbone variant is also provided:</p>
<pre>
  bash scripts/benchmark/gb/gb_crossdnav2_408k.sh human_enhancers_cohn /path/to/pretrain_408k.ckpt /path/to/genomic_benchmark
</pre>

<p>Task-specific <code>MAX_LENGTH</code>, <code>BATCH_SIZE</code>, and <code>LR</code> are selected automatically inside the script according to <code>DATASET_NAME</code>.</p>

<h3>2.3 Nucleotide Transformer Benchmark</h3>
<p>Datasets are hosted on the Hub as <code>InstaDeepAI/nucleotide_transformer_downstream_tasks</code>.</p>
<p>The recommended launch script is <code>scripts/benchmark/nt/nt_crossdnav2.sh</code>. Please update the checkpoint path and dataset root in the script, or pass them from the command line.</p>
<pre>
  bash scripts/benchmark/nt/nt_crossdnav2.sh H3K4me3
</pre>

<p>Optional override:</p>
<pre>
  bash scripts/benchmark/nt/nt_crossdnav2.sh H3K4me3 /path/to/pretrain.ckpt /path/to/nucleotide_transformer
</pre>

<p>Task-specific <code>BATCH_SIZE</code> and <code>LR</code> are configured inside the script for each NT dataset.</p>

---




<h2>3 The dataset for downstream tasks.</h2>

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


