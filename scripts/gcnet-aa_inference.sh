#!/usr/bin/env bash

# Inference on Scene Flow test set
CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode test \
--pretrained_aanet pretrained/gcnet-aa_sceneflow-0d6d65fa.pth \
--batch_size 1 \
--img_height 576 \
--img_width 960 \
--feature_type gcnet \
--feature_pyramid \
--num_downsample 1 \
--no_intermediate_supervision
