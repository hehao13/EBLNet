#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
python eval.py \
	--dataset MSD \
    --arch network.EBLNet.EBLNet_resnet101_os8 \
    --inference_mode  whole \
    --single_scale \
    --scales 1.0 \
    --split test \
    --cv_split 0 \
    --resize_scale 384 \
    --mode semantic \
    --with_mae_ber \
    --num_points 96 \
    --thres_gcn 0.9 \
    --num_cascade 3 \
    --no_flip \
    --ckpt_path ${2} \
    --snapshot ${1}
