conda create -n tos python=3.10 -y
conda activate tos
pip install -e .

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
  --phase all \
  --model-name gpt-5.2 \
  --num 10 \
  --data-dir room_data/3-room/ \
  --output-root result/ \
  --render-mode vision,text \
  --exp-type active,passive \
  --cogmap \
  --false-belief-exp \