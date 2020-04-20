#!/usr/bin/env bash

# Inference on Scene Flow test set
CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode test \
--pretrained_aanet pretrained/ganet-aa_sceneflow-a27abfeb.pth \
--batch_size 1 \
--img_height 576 \
--img_width 960 \
--feature_type ganet \
--feature_pyramid \
--no_intermediate_supervision
