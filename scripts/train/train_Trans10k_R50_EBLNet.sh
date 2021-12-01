#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./Trans10k/eblnet
mkdir -p ${EXP_DIR}
# Example on Cityscapes by resnet50-deeplabv3+ as baseline
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset Trans10k \
  --arch network.EBLNet.EBLNet_resnet50_os8 \
  --max_cu_epoch 16 \
  --lr 0.01 \
  --lr_schedule poly \
  --poly_exp 0.9 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --crop_size 512 \
  --max_epoch 16 \
  --dice_loss \
  --edge_weight 3.0 \
  --joint_edge_loss_light_cascade \
  --apex \
  --num_points 96 \
  --thres_gcn 0.9 \
  --num_cascade 3 \
  --bs_mult 4 \
  --exp  r50_os8_16epoches \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt
