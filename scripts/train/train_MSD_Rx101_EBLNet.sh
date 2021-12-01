#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./msd/eblnet
mkdir -p ${EXP_DIR}
# Example on Cityscapes by resnet50-deeplabv3+ as baseline
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset MSD \
  --arch network.EBLNet.EBLNet_resnext101_os8 \
  --max_cu_epoch 160 \
  --lr 0.002 \
  --lr_schedule poly \
  --poly_exp 0.9 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --crop_size 384 \
  --max_epoch 160 \
  --edge_weight 3.0 \
  --dice_loss \
  --joint_edge_loss_light_cascade \
  --apex \
  --num_points 96 \
  --thres_gcn 0.9 \
  --bs_mult 2 \
  --num_cascade 3 \
  --exp  rx101_160epoches \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt

