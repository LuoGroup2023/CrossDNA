#!/bin/bash
set -euo pipefail

source /data/zhaol/miniconda3/etc/profile.d/conda.sh

conda activate CrossDNA

export LD_LIBRARY_PATH=/data/zhaol/miniconda3/envs/CrossDNA/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}

cd /data/zhaol/projects/yangcheng/CrossDNAv2

full_path_to_root="/data/zhaol/projects/yangcheng/CrossDNAv2"

export HYDRA_FULL_ERROR=1

# Run script 140K 143360 100k 102400 50k 51200 10k 10240
SEQLEN=2048


BLOCK_SIZE=1024

SEQLEN_DIS="$((SEQLEN / 1024))k"

NUM_DEVICES=1

BATCH_SIZE=60

D_MODEL=128

Depth=6

LR="3e-4"

MAX_EPOCHES=60

RC_AUG="false"


WANDB_NAME="CrossDNAv2_len-${SEQLEN_DIS}_blocksize-${BLOCK_SIZE}_gpus-${NUM_DEVICES}_batchsize-${BATCH_SIZE}_d_model-${D_MODEL}_depth-${Depth}_lr-${LR}_epoches-${MAX_EPOCHES}"

# time
RUN_DATE=$(date +"%Y-%m-%d")
RUN_TIME=$(date +"%H-%M-%S")

HYDRA_RUN_DIR="${full_path_to_root}/outputs/pretrain/hg38/${RUN_DATE}/${RUN_TIME}_${WANDB_NAME}"
WATCH_DIR="${full_path_to_root}/watch_folder/pretrain_hg38/${RUN_DATE}"

export WANDBID=$(python -c "import wandb; print(wandb.util.generate_id())")

mkdir -p "${HYDRA_RUN_DIR}"
mkdir -p "${WATCH_DIR}"

if [ "${NUM_DEVICES}" -gt 1 ]; then
  TRAINER_STRATEGY_ARG='+trainer.strategy="ddp"'
else
  TRAINER_STRATEGY_ARG='+trainer.strategy="auto"'
fi

python -m train \
  experiment=hg38-pretrain/crossdnav2 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=$(( BATCH_SIZE / NUM_DEVICES )) \
  dataset.batch_size_eval=$(( BATCH_SIZE / NUM_DEVICES )) \
  dataset.rc_aug="${RC_AUG}" \
  dataset.add_eos=False \
  loader.num_workers=30 \
  model.config.depth=${Depth} \
  model.config.d_model=${D_MODEL} \
  model.config.block_size=${BLOCK_SIZE} \
  optimizer.lr="${LR}" \
  trainer.max_epochs=${MAX_EPOCHES} \
  trainer.precision=32 \
  trainer.devices=${NUM_DEVICES} \
  ${TRAINER_STRATEGY_ARG} \
  wandb.project=CrossDNA \
  wandb.group=crossdna_hg38_pretrain \
  wandb.mode=online \
  wandb.id=${WANDBID} \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  > "${WATCH_DIR}/${RUN_TIME}_${WANDB_NAME}.log" 2>&1
  



