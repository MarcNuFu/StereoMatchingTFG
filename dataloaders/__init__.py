import dataloaders.KITTIDataset as kitti
import dataloaders.SceneFlowDataset as sceneflow

__datasets__ = {
    "sceneflow": sceneflow.SceneFlowDataset,
    "kitti": kitti.KITTIDataset
}

__datapath__ = {
    "sceneflow": "./dataset/SceneFlow/",
    "kitti": "./dataset/kitti2015v2/"
}

__filenames__ = {
    "sceneflow_train": "./filenames/sceneflow_train.txt",
    "sceneflow_test": "./filenames/sceneflow_test.txt",
    "kitti_train": "./filenames/kitti15_val.txt",
    "kitti_test": "./filenames/kitti15_train.txt"
}