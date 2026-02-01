#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run: source setup.sh  (keeps conda env active)"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$(dirname "${BASH_SOURCE[0]}")"
conda create -n tos python=3.10 -y || true
conda activate tos
python -m pip install -e .

# Add api keys (optional)
# export OPENAI_API_KEY=
# export ANTHROPIC_API_KEY=
# export GOOGLE_API_KEY=

# Download the dataset
hf download yw12356/tos_3room_100runs --repo-type dataset --local-dir room_data
mkdir -p room_data/3-room
unzip room_data/*.zip -d room_data/3-room
for dir in room_data/3-room/*; do
  if [ -d "$dir" ]; then
    mv "$dir"/* room_data/3-room/
    rmdir "$dir"
  fi
done
rm room_data/*.zip

# Run the experiments
mkdir logs
python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num 25 \
  --data-dir room_data/3-room/ \
  --output-root results/ \
  --render-mode text,vision \
  --exp-type passive,active \
  --inference-mode batch \
  --false-belief-exp 2>&1 | tee logs/gpt-5.2.log