# P3SNet: Parallel Pyramid Pooling Stereo Network

This repository contains the code (in PyTorch) for "[P3SNet: Parallel Pyramid Pooling Stereo Network]" paper   

### Reference
```

```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Acknowledgements](#Acknowledgements)

## Introduction

In autonomous driving and advanced driver assistance systems (ADAS), stereo matching is a challenging research topic. Recent work has shown that high accuracy disparity maps can be obtained with end-to-end training with the help of deep convolutional neural networks from stereo images. However, many of these methods suffer from long run-time for real-time studies. Therefore, in this paper, we introduce P3SNet, which can generate both real-time results and  competitive disparity maps to the state-of-the-art. P3SNet architecture consists of two main modules: parallel pyramid pooling and hierarchical disparity aggregation. The parallel pyramid pooling structure makes it possible to obtain local and global information intensively from its multi-scale features. The hierarchical disparity aggregation provides multi-scale disparity maps by using a coarse-to-fine training strategy with the help of the costs obtained from multi-scale features. The proposed approach was evaluated on several benchmark datasets. The results on all datasets showed that the proposed P3SNet achieved better or competitive results while having lower runtime.

## Usage

Our code is based on PyTorch 1.6.0, CUDA 10.2 and python 3.8.

We recommend using conda for installation:
```
conda env create -f environment.yml
```
[Optional] For time comparison, we used [Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension) by modifying it.
```
cd model/ && python setup_cpu.py install
```
After installation change the correlation1D function in submodule.py


#### Datasets

Download [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) and [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) datasets. 

Dataset folder structure is as follows:

```
dataset
├── KITTI
│   ├── kitti2012
│   │   └── training
│   │   └── testing
│   └── kitti2015
│       └── training
│       └── testing
└── SceneFlow
    ├── driving_frames_cleanpass
    ├── driving_disparity
    ├── frames_cleanpass
    ├── frames_disparity
    ├── monkaa_frames_cleanpass
    └── monkaa_disparity
```

#### Train

Use the following command to train on Scene Flow.

```
python main.py --model P3SNET \
               --mode train \
               --dataset SceneFlow \
               --datapath (path to the dataset folder) \
               --maxdisp 192 \
               --epochs 200 \
               --train_batch_size 16 \
               --test_batch_size 16 \
               --save_dir saved_models 
```

Use the following command to finetune on KITTI 2012/KITTI 2015.
```
python main.py --model P3SNET \
               --mode train \
               --dataset KITTI2012(or KITTI2015) \
               --datapath (path to the dataset folder) \
               --loadmodel (path to the pre-training model)
               --maxdisp 192 \
               --epochs 700 \
               --train_batch_size 16 \
               --test_batch_size 16 \
               --save_dir saved_models/finetune2012
```


#### Evaluation

Use the following command to evaluate the trained model.
```
python main.py --model P3SNET \
               --mode eval \
               --dataset SceneFlow (SceneFlow, KITTI2012 or KITTI2015) \
               --datapath (path to the dataset folder) \
               --loadmodel (path to the pre-training model)
               --maxdisp 192
```

Replace P3SNet expressions with P3SNET_plus to use the P3SNET+ model instead of the P3SNet model.

## Results

#### Evaluations on SceneFlow Dataset

![Image](https://github.com/aemlek/P3SNet/blob/main/figure/table-4.png "KITTI20215_results")



#### Evaluations on KITTI 2012 and KITTI 2015 benchmarks

![Image](https://github.com/aemlek/P3SNet/blob/main/figure/table-5.png "KITTI20215_results")

![Image](https://github.com/aemlek/P3SNet/blob/main/figure/KITTI20212_results.png "KITTI20212_results")

![Image](https://github.com/aemlek/P3SNet/blob/main/figure/KITTI20215_results.png "KITTI20215_results")

## Acknowledgements

Part of the code is adopted from previous works: PSMNet, .... The 1D correlation op is taken from [Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension).  We thank the original authors for their awesome repos.

