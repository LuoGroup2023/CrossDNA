#!/bin/bash
set -euo pipefail

source /data/zhaol/miniconda3/etc/profile.d/conda.sh
conda activate CrossDNA

export LD_LIBRARY_PATH=/data/zhaol/miniconda3/envs/CrossDNA/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}

cd /data/zhaol/projects/yangcheng/CrossDNAv2
full_path_to_root="/data/zhaol/projects/yangcheng/CrossDNAv2"

export HYDRA_FULL_ERROR=1

# --------------------------------------------------
# Default settings
# You can run:
#   bash gb_crossdnav2.sh human_enhancers_cohn
#
# Optional override:
#   bash gb_crossdnav2.sh human_enhancers_cohn /new/pretrain.ckpt /new/data/root
# --------------------------------------------------

DEFAULT_PRETRAINED_CKPT="/data/zhaol/projects/yangcheng/CrossDNAv2/outputs/pretrain/hg38/2026-03-10/18-56-11_CrossDNAv2_len-2k_blocksize-2048_gpus-1_batchsize-60_d_model-128_depth-6_lr-3e-4_epoches-60/checkpoints/test/loss.ckpt"
DEFAULT_DATA_ROOT="/data/zhaol/projects/yangcheng/CrossDNAv2/data/genomic_benchmark"

DATASET_NAME="${1:-human_enhancers_cohn}"
PRETRAINED_CKPT="${2:-$DEFAULT_PRETRAINED_CKPT}"
DATA_ROOT="${3:-$DEFAULT_DATA_ROOT}"

if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
    echo "Checkpoint not found: ${PRETRAINED_CKPT}"
    echo "You can override it by:"
    echo "  bash gb_crossdna.sh <DATASET_NAME> <PRETRAINED_CKPT> [GB_DATA_ROOT]"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Dataset root not found: ${DATA_ROOT}"
    echo "You can override it by:"
    echo "  bash gb_crossdna.sh <DATASET_NAME> [PRETRAINED_CKPT] <GB_DATA_ROOT>"
    exit 1
fi

# --------------------------------------------------
# Task-specific settings: MAX_LENGTH / BATCH_SIZE / LR
#
# dummy_mouse_enhancers_ensembl     1210    2   median len 2381
# demo_coding_vs_intergenomic_seqs  100000  2   median len 200
# demo_human_or_worm                100000  2   median len 200
# human_enhancers_cohn              27791   2   median len 500
# human_enhancers_ensembl           154842  2   median len 269
# human_ensembl_regulatory          289061  3   median len 401
# human_nontata_promoters           36131   2   median len 251
# human_ocr_ensembl                 174756  2   median len 315
# --------------------------------------------------

case "${DATASET_NAME}" in
    dummy_mouse_enhancers_ensembl)
        MAX_LENGTH=2048
        BATCH_SIZE=8
        LR="5e-5"
        ;;
    demo_coding_vs_intergenomic_seqs)
        MAX_LENGTH=200
        BATCH_SIZE=128
        LR="8e-5"
        ;;
    demo_human_or_worm)
        MAX_LENGTH=200
        BATCH_SIZE=128
        LR="8e-5"
        ;;
    human_enhancers_cohn)
        MAX_LENGTH=500
        BATCH_SIZE=64
        LR="8e-5"
        ;;
    human_enhancers_ensembl)
        MAX_LENGTH=512
        BATCH_SIZE=96
        LR="1e-4"
        ;;
    human_ensembl_regulatory)
        MAX_LENGTH=768
        BATCH_SIZE=64
        LR="1e-4"
        ;;
    human_nontata_promoters)
        MAX_LENGTH=300
        BATCH_SIZE=96
        LR="8e-5"
        ;;
    human_ocr_ensembl)
        MAX_LENGTH=512
        BATCH_SIZE=60
        LR="2e-4"
        ;;
    *)
        echo "Unsupported DATASET_NAME: ${DATASET_NAME}"
        exit 1
        ;;
esac

NUM_DEVICES=1
GLOBAL_BATCH_SIZE=${BATCH_SIZE}
D_MODEL=128
DEPTH=6
BLOCK_SIZE=2048
MAX_EPOCHES=20
NUM_WORKERS=8
RC_AUG="false"

# 微调设置
WEIGHT_DECAY="0.01"
LABEL_SMOOTHING="0.03"
FREEZE_BACKBONE_EPOCHS=4

RUN_DATE=$(date +"%Y-%m-%d")
RUN_TIME=$(date +"%H-%M-%S")

WANDB_NAME="gb_crossdnav2_${DATASET_NAME}_len-${MAX_LENGTH}_dmodel-${D_MODEL}_depth-${DEPTH}_lr-${LR}_bs-${BATCH_SIZE}"
HYDRA_RUN_DIR="${full_path_to_root}/outputs/gb_benchmark/${RUN_DATE}/${RUN_TIME}_${WANDB_NAME}"
WATCH_DIR="${full_path_to_root}/watch_folder/gb_benchmark/${RUN_DATE}"

export WANDBID=$(python -c "import wandb; print(wandb.util.generate_id())")

mkdir -p "${HYDRA_RUN_DIR}"
mkdir -p "${WATCH_DIR}"

ARGS=(
    experiment=genomic-benchmark/crossdnav2
    dataset.dataset_name=${DATASET_NAME}
    dataset.dest_path=${DATA_ROOT}
    dataset.max_length=${MAX_LENGTH}
    dataset.batch_size=${BATCH_SIZE}
    dataset.rc_aug=${RC_AUG}
    loader.num_workers=${NUM_WORKERS}
    train.global_batch_size=${GLOBAL_BATCH_SIZE}

    model.config.d_model=${D_MODEL}
    model.config.depth=${DEPTH}
    model.config.block_size=${BLOCK_SIZE}
    model.config.use_bridge=true
    model.config.bridge_dropout=0.0
    model.config.gate_freeze_steps=0
    model.config.transformer_cfg.hidden_size=${D_MODEL}
    model.config.transformer_cfg.attn.window_size=128
    model.config.comba_cfg.hidden_size=${D_MODEL}

    optimizer.lr=${LR}
    optimizer.weight_decay=${WEIGHT_DECAY}
    task.loss.label_smoothing=${LABEL_SMOOTHING}

    trainer.max_epochs=${MAX_EPOCHES}
    trainer.precision=bf16-mixed
    trainer.devices=${NUM_DEVICES}

    train.freeze_backbone_epochs=${FREEZE_BACKBONE_EPOCHS}
    train.pretrained_model_path=${PRETRAINED_CKPT}
    train.pretrained_model_strict_load=false
    train.pretrained_model_state_hook.freeze_backbone=false

    wandb.project=CrossDNA-gb-benchmark
    wandb.group=crossdnav2_gb_finetune
    wandb.mode=online
    wandb.id=${WANDBID}
    wandb.name=${WANDB_NAME}

    hydra.run.dir=${HYDRA_RUN_DIR}
)

if [ "${NUM_DEVICES}" -gt 1 ]; then
    ARGS+=('+trainer.strategy="ddp"')
else
    ARGS+=('+trainer.strategy="auto"')
fi

python -m train "${ARGS[@]}" \
    > "${WATCH_DIR}/${RUN_TIME}_${WANDB_NAME}.log" 2>&1