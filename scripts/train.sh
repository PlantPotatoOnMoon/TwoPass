#!/usr/bin/env bash
# Launch TwoPass training with one process per GPU.
#
#   WAN_DIR=/path/to/Wan2.2-TI2V-5B MANIFEST=/path/to/train_manifest.json OUT_DIR=/path/to/run \
#       bash scripts/train.sh configs/bridge.json [--resume_checkpoint /path/to/run/stepXXXXXXX.pt]
set -euo pipefail

CONFIG="$1"
shift
: "${WAN_DIR:?set WAN_DIR to the Wan2.2-TI2V-5B directory}"
: "${MANIFEST:?set MANIFEST to the training manifest}"
: "${OUT_DIR:?set OUT_DIR to the output directory}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29996}"

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

python -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port "${MASTER_PORT}" \
    -m twopass.train --config "${CONFIG}" --wan_dir "${WAN_DIR}" --manifest "${MANIFEST}" \
    --out_dir "${OUT_DIR}" "$@"
