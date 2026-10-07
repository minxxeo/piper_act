#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p ../logs
log_file="../logs/piper_act_chunk50_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Training log: $(realpath "$log_file")"
{
date --iso-8601=seconds
python -c 'import sys, torch; assert sys.prefix.endswith("/piper_act"), "Activate conda environment piper_act first"; assert torch.cuda.is_available(), "CUDA GPU is unavailable"; print("Python:", sys.version); print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))'
python -u imitate_episodes.py \
    --task_name piper_wrist \
    --ckpt_dir ../checkpoints/piper_wrist_act \
    --policy_class ACT --kl_weight 10 --chunk_size 50 \
    --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 5000 --lr 1e-5 --seed 0 "$@"
} 2>&1 | tee "$log_file"
