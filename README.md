# StereoMatchingTFG

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)

## Introduction
TO DO

## Usage

### Setup

Install conda and import virtual environment from stereo.yml with:
```
conda env create -f stereo.yml
```
Activate the virtual environment with:
```
conda activate stereo
```

Download the following datasets:
- [KITTI Stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
Usage of KITTI 2015 dataset:
Download stereo 2015/flow 2015/scene flow 2015 data set (2 GB)

Usage of Scene Flow dataset:
Download RGB finalpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa.
Create the folders frames_finalpass, disparity and camera_data. In each one create the folders TRAIN and TEST.
Locate Driving and Monkaa intern folders in TRAIN and FlyingThings3D has TRAIN and TEST folders.
```

Change the datasets paths in [dataloaders/\_\_init\_\_.py](dataloaders/__init__.py)

### Train
To train on Sceneflow use the following command:
```python
sh train.sh
```
That contains the following code:
```python
CUDA_VISIBLE_DEVICES=0 python train.py \
                       --dataset 'sceneflow' \
                       --batchsize 32 \
                       --total_epochs 40 \
                       --start_epoch 0 \
                       --workers_train 8 \
                       --workers_test 8 \
                       --learnrate  0.0001 \
                       --model DispNet \
                       --logdir 'tensorboard/finetune' \
                       --pth_name 'DispNetSceneFlow' \
                       --maxdisp 192
```
Tensorboard will save the loss, metrics and outputs on [tensorboard/train](tensorboard/train).

The pth will be saved on /Vitis/build/float_model/DispNetSceneFlow.pth.

The old format pth will be saved on /Vitis/build/float_model/DispNetSceneFlow_old.pth.

### Finetune
To finetune on KITTI 2015 use the following command:
```python
sh finetune.sh
```
That contains the following code:
```python
CUDA_VISIBLE_DEVICES=0 python train.py \
                       --dataset 'kitti' \
                       --batchsize 32 \
                       --total_epochs 300 \
                       --start_epoch 0 \
                       --workers_train 8 \
                       --workers_test 8 \
                       --learnrate 0.0001 \
                       --model DispNet \
                       --logdir 'tensorboard/finetune' \
                       --pth_name 'DispNetKITTI.pth' \
                       --maxdisp 192 \
                       --load_model './Vitis/build/float_model/DispNetSceneFlow.pth'
```

Tensorboard will save the loss, metrics and outputs on [tensorboard/finetune](tensorboard/finetune).

The pth will be saved on /Vitis/build/float_model/DispNetKITTI.pth.

The old format pth will be saved on /Vitis/build/float_model/DispNetKITTI_old.pth.

### Prediction
TO DO

### Pretrained Model
TO DO

### Tensorboard
To check loss, metrics and outputs of execution use the following command in [tensorboard](tensorboard) directory:
```
tensorboard --logdir (train or finetune) --port (desired port)
```

If you are connecting via ssh the following option can be added to the ssh command to redirect the remote port to your local machine:
```
ssh ... -L (local port):127.0.0.1:(remote port) ...
```

### Creación xmodel de la red para ejecutar con Alveo U50 (Vitis Ai)
Se genera el modelo cuantizado en el directorio ./Vitis/build/quant_model y se compila para poder ejecutar en Alveo U50
```python
./docker_run.sh xilinx/vitis-ai-cpu:latest
conda activate vitis-ai-pytorch
cd Vitis
sh quantize.sh
source compile.sh u50
```
### Ejecutar en Alveo U50
(TO DO)

## Results
TO FO

