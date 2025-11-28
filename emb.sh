BATCH_SIZE=8
NUM=8
METHOD="vine"
DATASETS="/mnt/shared/Diffusiondbsub/train"
OUTPUT_DIR="./outputs"

# generate authentic watermark data for testing the performance of defense

echo "Creating dataset for ${METHOD} on ${DATASETS}"
python -m wm.cli.emb \
    --dataset ${DATASETS} \
    --output_dir ${OUTPUT_DIR} \
    --method ${METHOD} \
    --num ${NUM} \
    --batch_size ${BATCH_SIZE} \
    --filetype .png \
    --message default
echo "Finished ${method} on ${dataset}"
