#!/usr/bin/env bash

# Evaluate the best validation model on Scene Flow test set
CUDA_VISIBLE_DEVICES=0 python train.py \
--mode test \
--checkpoint_dir checkpoints/aanet_sceneflow \
--batch_size 64 \
--val_batch_size 1 \
--img_height 288 \
--img_width 576 \
--val_img_height 576 \
--val_img_width 960 \
--feature_type aanet \
--feature_pyramid_network \
--milestones 20,30,40,50,60 \
--max_epoch 64 \
--evaluate_only

# Evaluate a specific model on Scene Flow test set
CUDA_VISIBLE_DEVICES=0 python train.py \
--mode test \
--checkpoint_dir checkpoints/aanet_sceneflow \
--pretrained_aanet pretrained/aanet_sceneflow-5aa5a24e.pth \
--batch_size 64 \
--val_batch_size 1 \
--img_height 288 \
--img_width 576 \
--val_img_height 576 \
--val_img_width 960 \
--feature_type aanet \
--feature_pyramid_network \
--milestones 20,30,40,50,60 \
--max_epoch 64 \
--evaluate_only