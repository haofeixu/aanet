#!/usr/bin/env bash

# Train on Scene Flow training set
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--mode val \
--data_dir data/SceneFlow \
--checkpoint_dir checkpoints/aanet_sceneflow \
--batch_size 64 \
--val_batch_size 64 \
--img_height 288 \
--img_width 576 \
--val_img_height 576 \
--val_img_width 960 \
--feature_type aanet \
--feature_pyramid_network \
--milestones 20,30,40,50,60 \
--max_epoch 64

# Train on mixed KITTI 2012 and KITTI 2015 training set
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--data_dir data/KITTI \
--dataset_name KITTI_mix \
--checkpoint_dir checkpoints/aanet_kittimix \
--pretrained_aanet checkpoints/aanet_sceneflow/aanet_best.pth \
--batch_size 32 \
--val_batch_size 32 \
--img_height 336 \
--img_width 960 \
--val_img_height 384 \
--val_img_width 1248 \
--feature_type aanet \
--feature_pyramid_network \
--load_pseudo_gt \
--milestones 400,600,800,900 \
--max_epoch 1000 \
--save_ckpt_freq 100 \
--no_validate

# Train on KITTI 2015 training set
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--data_dir data/KITTI/kitti_2015/data_scene_flow \
--dataset_name KITTI2015 \
--mode train_all \
--checkpoint_dir checkpoints/aanet_kitti15 \
--pretrained_aanet checkpoints/aanet_kittimix/aanet_latest.pth \
--batch_size 32 \
--val_batch_size 32 \
--img_height 336 \
--img_width 960 \
--val_img_height 384 \
--val_img_width 1248 \
--feature_type aanet \
--feature_pyramid_network \
--load_pseudo_gt \
--highest_loss_only \
--learning_rate 1e-4 \
--milestones 400,600,800,900 \
--max_epoch 1000 \
--save_ckpt_freq 100 \
--no_validate

# Train on KITTI 2012 training set
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--data_dir data/KITTI/kitti_2012/data_stereo_flow \
--dataset_name KITTI2012 \
--mode train_all \
--checkpoint_dir checkpoints/aanet_kitti12 \
--pretrained_aanet checkpoints/aanet_kittimix/aanet_latest.pth \
--batch_size 32 \
--val_batch_size 32 \
--img_height 336 \
--img_width 960 \
--val_img_height 384 \
--val_img_width 1248 \
--feature_type aanet \
--feature_pyramid_network \
--load_pseudo_gt \
--highest_loss_only \
--learning_rate 1e-4 \
--milestones 400,600,800,900 \
--max_epoch 1000 \
--save_ckpt_freq 100 \
--no_validate
