#!/usr/bin/env bash

# Inference on Scene Flow test set
CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode test \
--pretrained_aanet pretrained/stereonet-aa_sceneflow-1fbe2dea.pth \
--batch_size 1 \
--img_height 576 \
--img_width 960 \
--feature_type stereonet \
--num_scales 1 \
--num_fusions 4 \
--num_deform_blocks 4 \
--refinement_type stereonet
