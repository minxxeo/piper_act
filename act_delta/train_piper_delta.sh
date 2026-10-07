#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p ../logs_delta
log_file="../logs_delta/piper_act_delta_chunk50_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Delta training log: $(realpath "$log_file")"
{
date --iso-8601=seconds
python -c 'import sys, torch, detr.main; assert sys.prefix.endswith("/piper_act"), "Activate piper_act first"; assert torch.cuda.is_available(), "CUDA unavailable"; print("Model source:", detr.main.__file__); print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'
python -u imitate_episodes.py \
    --task_name piper_wrist_delta_follower \
    --ckpt_dir ../checkpoints_delta/piper_wrist_delta_follower_chunk_act \
    --policy_class ACT --kl_weight 10 --chunk_size 50 \
    --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 5000 --lr 1e-5 --seed 0 "$@"
} 2>&1 | tee "$log_file"
