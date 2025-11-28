METHOD="hidden"
DATAPATH="/home/dongziping/WMSuite/outputs/hidden"
LOG_DIR="./outputs"
DEVICE="cuda:0"
BETA=1e-6

# generate authentic watermark data for testing the performance of defense

echo "Extracting for ${METHOD} on ${DATAPATH}"
python -m wm.cli.extract \
    --data_path ${DATAPATH} \
    --log_dir ${LOG_DIR} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --beta ${BETA}
echo "Finish!"
