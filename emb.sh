BATCH_SIZE=8
NUM=1000
DEVICE="cuda:1"
METHOD="vine" # stegastamp, dwtdct, stable_signature,hidden,rivaGan, vine
DATASETS="sampled_train_5000"
OUTPUT_DIR="/home/dongziping/WMAna/data"

# generate authentic watermark data for testing the performance of defense

echo "Creating dataset for ${METHOD} on ${DATASETS}"
python -m wm.cli.emb \
    --dataset ${DATASETS} \
    --output_dir ${OUTPUT_DIR} \
    --method ${METHOD} \
    --num ${NUM} \
    --batch_size ${BATCH_SIZE} \
    --filetype .png \
    --message_mode default \
    --device ${DEVICE}
echo "Finished ${method} on ${dataset}"
