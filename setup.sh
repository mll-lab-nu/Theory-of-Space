#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run: source setup.sh [--run-exp]  (keeps conda env active)"
  exit 1
fi

# Parse arguments
RUN_EXP=false
for arg in "$@"; do
  case $arg in
    --run-exp)
      RUN_EXP=true
      shift
      ;;
  esac
done

# set up conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$(dirname "${BASH_SOURCE[0]}")"
if ! conda env list | awk '{print $1}' | grep -qx "tos"; then
  conda create -n tos python=3.10 -y
  conda activate tos
  python -m pip install -e .
fi

# Add huggingface token (optional, avoid 429 rate limit)
# export HF_TOKEN=

# Download the dataset (skip only if room_data/3-room exists AND is non-empty)
if [ -d "room_data/3-room" ] && [ "$(ls -A room_data/3-room 2>/dev/null)" ]; then
  echo "room_data/3-room already exists and is not empty, skipping."
else
  hf download MLL-Lab/tos-data --repo-type dataset --local-dir room_data
  mkdir -p room_data/3-room
  unzip room_data/*.zip -d room_data/3-room

  for dir in room_data/3-room/*; do
    if [ -d "$dir" ]; then
      mv "$dir"/* room_data/3-room/
      rmdir "$dir"
    fi
  done

  rm -f room_data/*.zip
fi


# Add api keys (if you have not set them up yet)
# export OPENAI_API_KEY=
# export ANTHROPIC_API_KEY=
# export GOOGLE_API_KEY=

# run experiments
if $RUN_EXP; then
  mkdir -p logs
  python scripts/SpatialGym/spatial_run.py \
    --phase all \
    --model-name gpt-5.2 \
    --num 25 \
    --data-dir room_data/3-room/ \
    --output-root results/ \
    --render-mode text,vision \
    --exp-type passive,active \
    --inference-mode batch 2>&1 | tee logs/gpt-5.2.log
else
  echo "Skipping experiments. Use --run-exp to run them."
fi
