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
hf download yw12356/tos_dataset_0117_3room_100runs --repo-type dataset --local-dir room_data
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
python scripts/SpatialGym/spatial_run.py \
  --phase explore \
  --model-name gemini-3-pro-preview \
  --num 100 \
  --data-dir room_data/3-room/ \
  --output-root results_arxiv/ \
  --render-mode text,vision \
  --exp-type active \
  --false-belief-exp \
  --replay