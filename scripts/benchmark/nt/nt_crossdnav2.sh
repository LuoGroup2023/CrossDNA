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
#   bash nt_crossdnav2.sh H3K4me3
#
# Optional override:
#   bash nt_crossdnav2.sh H3K4me3 /new/pretrain.ckpt /new/data/root
# --------------------------------------------------

DEFAULT_PRETRAINED_CKPT="/data/zhaol/projects/yangcheng/CrossDNAv2/outputs/pretrain/hg38/2026-03-10/18-56-11_CrossDNAv2_len-2k_blocksize-2048_gpus-1_batchsize-60_d_model-128_depth-6_lr-3e-4_epoches-60/checkpoints/test/loss.ckpt"
DEFAULT_DATA_ROOT="/data/zhaol/projects/yangcheng/CrossDNAv2/data/nucleotide_transformer"

DATASET_NAME="${1:-H3K9ac_double_strand}"
PRETRAINED_CKPT="${2:-$DEFAULT_PRETRAINED_CKPT}"
DATA_ROOT="${3:-$DEFAULT_DATA_ROOT}"

if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
    echo "Checkpoint not found: ${PRETRAINED_CKPT}"
    echo "You can override it by:"
    echo "  bash nt_crossdna.sh <DATASET_NAME> <PRETRAINED_CKPT> [NT_DATA_ROOT]"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Dataset root not found: ${DATA_ROOT}"
    echo "You can override it by:"
    echo "  bash nt_crossdna.sh <DATASET_NAME> [PRETRAINED_CKPT] <NT_DATA_ROOT>"
    exit 1
fi

# -----------------------------
# enhancer                 200  2  14968  MCC
# enhancer_types           200  3  14968  MCC
# H3                       500  2  13468  MCC
# H3K4me1                  500  2  28509  MCC
# H3K4me2                  500  2  27614  MCC
# H3K4me3                  500  2  33119  MCC
# H3K9ac                   500  2  25003  MCC
# H3K14ac                  500  2  29743  MCC
# H3K36me3                 500  2  31392  MCC
# H3K79me3                 500  2  25953  MCC
# H4                       500  2  13140  MCC
# H4ac                     500  2  30685  MCC
# promoter_all             300  2  53276  F1
# promoter_non_tata        300  2  47759  F1
# promoter_tata            300  2   5517  F1
# splice_sites_all         600  3  27000  MCC
# splice_sites_acceptor    600  2  19961  F1
# splice_sites_donor       600  2  19775  MCC
# -----------------------------

case "${DATASET_NAME}" in
    enhancer) BATCH_SIZE=60; LR="1e-4" ;;
    enhancer_types) BATCH_SIZE=60; LR="1e-4" ;;
    H3) BATCH_SIZE=60; LR="1e-4" ;;
    H3K4me1) BATCH_SIZE=40; LR="1e-4" ;;
    H3K4me2) BATCH_SIZE=40; LR="1e-4" ;;
    H3K4me3) BATCH_SIZE=60; LR="2e-4" ;;
    H3K9ac) BATCH_SIZE=60; LR="2e-4" ;;
    H3K14ac) BATCH_SIZE=60; LR="2e-4" ;;
    H3K36me3) BATCH_SIZE=40; LR="1e-4" ;;
    H3K79me3) BATCH_SIZE=60; LR="2e-4" ;;
    H4) BATCH_SIZE=40; LR="1e-4" ;;
    H4ac) BATCH_SIZE=40; LR="1e-4" ;;
    promoter_all) BATCH_SIZE=40; LR="1e-4" ;;
    promoter_non_tata) BATCH_SIZE=40; LR="1e-4" ;;
    promoter_tata) BATCH_SIZE=60; LR="2e-4" ;;
    splice_sites_acceptor) BATCH_SIZE=80; LR="3e-4" ;;
    splice_sites_donor) BATCH_SIZE=40; LR="1e-4" ;;
    splice_sites_all) BATCH_SIZE=40; LR="1e-4" ;;
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
MAX_EPOCHES=10
NUM_WORKERS=8
RC_AUG="false"

# 微调设置
LABEL_SMOOTHING="0.01"
WEIGHT_DECAY="0.01"
FREEZE_BACKBONE_EPOCHS=1

RUN_DATE=$(date +"%Y-%m-%d")
RUN_TIME=$(date +"%H-%M-%S")

WANDB_NAME="nt_crossdnav2_${DATASET_NAME}_dmodel-${D_MODEL}_depth-${DEPTH}_lr-${LR}_bs-${BATCH_SIZE}"
HYDRA_RUN_DIR="${full_path_to_root}/outputs/nt_benchmark/${RUN_DATE}/${RUN_TIME}_${WANDB_NAME}"
WATCH_DIR="${full_path_to_root}/watch_folder/nt_benchmark/${RUN_DATE}"

export WANDBID=$(python -c "import wandb; print(wandb.util.generate_id())")

mkdir -p "${HYDRA_RUN_DIR}"
mkdir -p "${WATCH_DIR}"

ARGS=(
    experiment=nt-benchmark/crossdnav2

    dataset.dataset_name=${DATASET_NAME}
    dataset.dest_path=${DATA_ROOT}
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

    train.pretrained_model_path=${PRETRAINED_CKPT}
    train.pretrained_model_strict_load=false
    train.pretrained_model_state_hook.freeze_backbone=false
    train.freeze_backbone_epochs=${FREEZE_BACKBONE_EPOCHS}

    wandb.project=CrossDNA-nt-benchmark
    wandb.group=crossdnav2_nt_finetune
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