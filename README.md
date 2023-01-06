# Spatial Pruned Sparse Convolution for Efficient 3D Object Detection

<p align="center">
    <a href="https://arxiv.org/abs/2209.14201"><img src="https://img.shields.io/badge/arXiv-2210.05593-b31b1b"></a>
    <a href="https://github.com/CVMI-Lab/SlotCon/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>
<p align="center">
	Spatial Pruned Sparse Convolution for Efficient 3D Object Detection (NeurIPS 2022)<br>
  By
  Jianhui Liu,
  Yukang Chen,
  Xiaoqing Ye,
  Zhuotao Tian,
  Xiao Tan,
  and 
  Xiaojuan Qi.
</p>

## Introduction

3D scenes are dominated by a large number of background points, which is redundant for the detection task that mainly needs to focus on foreground objects.
In this paper, we analyze major components of existing sparse 3D CNNs and
find that 3D CNNs ignore the redundancy of data and further amplify it in the
down-sampling process, which brings a huge amount of extra and unnecessary
computational overhead. Inspired by this, we propose a new convolution operator
named spatial pruned sparse convolution (SPS-Conv), which includes two variants,
spatial pruned submanifold sparse convolution (SPSS-Conv) and spatial pruned
regular sparse convolution (SPRS-Conv), both of which are based on the idea of
dynamically determining crucial areas for redundancy reduction. We validate that
the magnitude can serve as important cues to determine crucial areas which get rid
of the extra computations of learning-based methods. The proposed modules can
easily be incorporated into existing sparse 3D CNNs without extra architectural
modifications. Extensive experiments on the KITTI, Waymo and nuScenes datasets
demonstrate that our method can achieve more than 50% reduction in GFLOPs
without compromising the performance.


<p align="center">
	<img src=pic/method.png width=80% />
<p align="center">


### Experimental results

#### nuScenes dataset

|                                             | mAP | NDS | download | 
|---------------------------------------------|----------:|:-------:|:---------:|
| [CenterPoint + SPSS ratio0.5 + SPRS ratio0.5](cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_spss_ratio0.5_sprs_ratio0.5.yaml) | 58.68	| 66.26	 | [Google](https://drive.google.com/file/d/1mgHP61bZ2iqwdISBtIrKRcSt7YBoloVr/view?usp=sharing) | 
| [CenterPoint + SPSS ratio0.3 + SPRS ratio0.5](cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_spss_ratio0.3_sprs_ratio0.5.yaml) | 59.09	| 66.63	 | [Google](https://drive.google.com/file/d/1C8Dm3yZZjuplYzkqPzXWgCyEUcIREcbX/view?usp=sharing)| 


## Getting Started
### Installation

#### a. Clone this repository
```shell
https://github.com/CVMI-Lab/SPS-Conv.git && cd SPS-Conv
```
#### b. Install the environment
Following the install documents for [OpenPCdet](docs/INSTALL.md)

*spconv 2.x is highly recommended instead of spconv 1.x version.

#### c. Prepare the datasets. 

Download and organize the official [KITTI](docs/GETTING_STARTED.md) and [Waymo](docs/GETTING_STARTED.md) and [nuScenes](GETTING_STARTED.md)  following the document in OpenPCdet.



### Training

For KITTI
```shell
cd tools 
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file /cfgs/kitti_models/voxel_rcnn_car_spss_ratio0.5_sprs_ratio0.5.yaml
```

For nuScenes
```shell
cd tools 
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_spss_ratio0.3_sprs_ratio0.5.yaml
```

For Waymo
```shell
cd tools 
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/waymo_models/centerpoint_spss_ratio0.3_sprs_ratio0.5.yaml
```

### Evaluation
We provide the trained weight file so you can just run with that. You can also use the model you trained.

```shell
NUM_GPUS=4
cd tools 
bash scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${PATH_TO_CONFIG_FILE} --ckpt ${PATH_TO_CHECKPOINT}
```


## TODO List
- - [ ] CUDA code of SPS-Conv
- - [ ] KITTI and Waymo pre-trained weights (will be released within a week)

## Citation 
If you find this project useful in your research, please consider citing:

```
@inproceedings{liu2022spatial,
  title={Spatial Pruned Sparse Convolution for Efficient 3D Object Detection},
  author={Liu, Jianhui and Chen, Yukang and Ye, Xiaoqing and Tian, Zhuotao and Tan, Xiao and Qi, Xiaojuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Acknowledgement
-  This work is built upon the `OpenPCDet`. Please refer to the official github repositories, [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for more information.

