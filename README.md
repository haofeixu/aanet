# AANet

PyTorch implementation of our paper: 

**[AANet: Adaptive Aggregation Network for Efficient Stereo Matching](https://arxiv.org/abs/2004.09548)**, [CVPR 2020](http://cvpr2020.thecvf.com/)

Authors: Haofei Xu and [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/)

We propose a sparse points based intra-scale cost aggregation (ISA) module and a cross-scale cost aggregation (CSA) module for efficient and accurate stereo matching. 

The implementation of improved version **AANet+ (stronger performance & slightly faster speed)** is also included in this repo.

<p align="center"><img width=80% src="assets/overview.png"></p>

## Highlights

- **Modular design**

  We decompose the end-to-end stereo matching framework into five components: 

  **feature extraction**, **cost volume construction**, **cost aggregation**, **disparity computation** and **disparity refinement.** 

  One can easily construct a customized stereo matching model by combining different components.

- **High efficiency**

  Our method can run at **60ms** for a KITTI stereo pair (384x1248 resolution)!

- **Full framework**

  All codes for training, validating, evaluating, inferencing and predicting on any stereo pair are provided!

## Installation

Our code is based on PyTorch 1.2.0, CUDA 10.0 and python 3.7. 

We recommend using [conda](https://www.anaconda.com/distribution/) for installation: 

```shell
conda env create -f environment.yml
```

After installing dependencies, build deformable convolution:

```shell
cd nets/deform_conv && bash build.sh
```

## Dataset Preparation

Download [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) datasets. 

Our folder structure is as follows:

```
data
├── KITTI
│   ├── kitti_2012
│   │   └── data_stereo_flow
│   ├── kitti_2015
│   │   └── data_scene_flow
└── SceneFlow
    ├── Driving
    │   ├── disparity
    │   └── frames_finalpass
    ├── FlyingThings3D
    │   ├── disparity
    │   └── frames_finalpass
    └── Monkaa
        ├── disparity
        └── frames_finalpass
```

If you would like to use the pseudo ground truth supervision introduced in our paper, you can download the pre-computed disparity on KITTI 2012 and KITTI 2015 training set here: [KITTI 2012](https://drive.google.com/open?id=1ZJhraqgY1sL4UfHBrVojttCbvNAXfdj0), [KITTI 2015](https://drive.google.com/open?id=14NGQp9CwIVNAK8ZQ6GSNeGraFGtVGOce). 

For KITTI 2012, you should place the unzipped file `disp_occ_pseudo_gt` under `kitti_2012/data_stereo_flow/training` directory. 

For KITTI 2015, you should place `disp_occ_0_pseudo_gt` under `kitti_2015/data_scene_flow/training`.

It is recommended to symlink your dataset root to `$AANET/data`:

```shell
ln -s $YOUR_DATASET_ROOT data
```

Otherwise, you may need to change the corresponding paths in the scripts.

## Model Zoo

All pretrained models are available in the [model zoo](MODEL_ZOO.md).

We assume the downloaded weights are located under the `pretrained` directory. 

Otherwise, you may need to change the corresponding paths in the scripts.

## Inference

To generate prediction results on the test set of Scene Flow and KITTI dataset, you can run [scripts/aanet_inference.sh](scripts/aanet_inference.sh). 

The inference results on KITTI dataset can be directly submitted to the online evaluation server for benchmarking.

## Prediction

We also support predicting on any rectified stereo pairs. [scripts/aanet_predict.sh](scripts/aanet_predict.sh) provides an example usage.

## Training

All training scripts on Scene Flow and KITTI datasets are provided in [scripts/aanet_train.sh](scripts/aanet_train.sh). 

Note that we use 4 NVIDIA V100 GPUs (32G) with batch size 64 for training, you may need to tune the batch size according to your hardware. 

We support using tensorboard to monitor and visualize the training process. You can first start a tensorboard session with

```shell
tensorboard --logdir checkpoints
```

and then access [http://localhost:6006](http://localhost:6006) in your browser.

- **How to train on my own data?**

  You can first generate a filename list by creating a data reading function in [filenames/generate_filenames.py](filenames/generate_filenames.py) (an example on KITTI dataset is provided), and then create a new dataset dictionary in [dataloader/dataloader.py](dataloader/dataloader.py).

- **How to develop new components?**

  Our framework is flexible to develop new components, e.g., new feature extractor, cost aggregation module or refinement architecture. You can 1) create a new file (e.g., `my_aggregation.py`) under `nets` directory, 2) import the module in `nets/aanet.py` and 3) use it in the model definition.

## Evaluation

To enable fast experimenting, evaluation runs on-the-fly without saving the intermediate results. 

We provide two types of evaluation setting:

- After training, evaluate the model with best validation results
- Evaluate a pretrained model

Check [scripts/aanet_evaluate.sh](scripts/aanet_evaluate.sh) for an example usage.

## Citation

If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{xu2020aanet,
  title={AANet: Adaptive Aggregation Network for Efficient Stereo Matching},
  author={Xu, Haofei and Zhang, Juyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1959--1968},
  year={2020}
}
```


## Acknowledgements

Part of the code is adopted from previous works: [PSMNet](https://github.com/JiaRenChang/PSMNet), [GwcNet](https://github.com/xy-guo/GwcNet) and [GA-Net](https://github.com/feihuzhang/GANet). We thank the original authors for their awesome repos. The deformable convolution op is taken from [mmdetection](https://github.com/open-mmlab/mmdetection). The FLOPs counting code is modified from [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter). The code structure is partially inspired by [mmdetection](https://github.com/open-mmlab/mmdetection) and our previous work [rdn4depth](https://github.com/haofeixu/rdn4depth).





