#!/usr/bin/env bash

# Inference on Scene Flow test set
CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode test \
--pretrained_aanet pretrained/aanet_sceneflow-5aa5a24e.pth \
--batch_size 1 \
--img_height 576 \
--img_width 960 \
--feature_type aanet \
--feature_pyramid_network \
--no_intermediate_supervision

# Inference on KITTI 2015 test set for submission
CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode test \
--data_dir data/KITTI/kitti_2015/data_scene_flow \
--dataset_name KITTI2015 \
--pretrained_aanet pretrained/aanet_kitti15-fb2a0d23.pth \
--batch_size 1 \
--img_height 384 \
--img_width 1248 \
--feature_type aanet \
--feature_pyramid_network \
--no_intermediate_supervision \
--output_dir output/kitti15_test

# Inference on KITTI 2012 test set for submission
CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode test \
--data_dir data/KITTI/kitti_2012/data_stereo_flow \
--dataset_name KITTI2012 \
--pretrained_aanet pretrained/aanet_kitti12-e20bb24d.pth \
--batch_size 1 \
--img_height 384 \
--img_width 1248 \
--feature_type aanet \
--feature_pyramid_network \
--no_intermediate_supervision \
--output_dir output/kitti12_test
