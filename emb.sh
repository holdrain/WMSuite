BATCH_SIZE=8
NUM=8
METHOD="vine"
DATASETS=""
OUTPUT_DIR="./outputs"

echo "Generating watermarked data using ${METHOD} on ${DATASETS}"

python -m wm.cli.emb \
    --dataset ${DATASETS} \
    --output_dir ${OUTPUT_DIR} \
    --method ${METHOD} \
    --num ${NUM} \
    --batch_size ${BATCH_SIZE} \
    --filetype .png \
    --message default
echo "Finished ${method} on ${dataset}"
