# StereoMatchingTFG

## Setup

Install conda and import virtual environment from stereo.yml with:
```python
conda env create -f stereo.yml
```

Download the following datasets:
- [KITTI Stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
Usage of KITTI 2015 dataset
Download stereo 2015/flow 2015/scene flow 2015 data set (2 GB)

Usage of Scene Flow dataset
Download RGB finalpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa.
Create the folders frames_finalpass, disparity and camera_data. In each one create the folders TRAIN and TEST.
Locate Driving and Monkaa intern folders in TRAIN and FlyingThings3D has TRAIN and TEST folders.
```

Change the datasets paths in [dataloaders/\_\_init\_\_.py](dataloaders/__init__.py)

## Entrenar red y generación pth
Se genera model.pth en el directorio ./Vitis/build/float_model
```python
sh main.sh
```

## Creación xmodel de la red para ejecutar con Alveo U50 (Vitis Ai)
Se genera el modelo cuantizado en el directorio ./Vitis/build/quant_model y se compila para poder ejecutar en Alveo U50
```python
./docker_run.sh xilinx/vitis-ai-cpu:latest
conda activate vitis-ai-pytorch
cd Vitis
sh quantize.sh
source compile.sh u50
```

## Ejecutar en Alveo U50
(TO DO)
