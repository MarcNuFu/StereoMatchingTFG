import dataloaders.KITTIDataset as kitti
import dataloaders.SceneFlowDataset as sceneflow
from pathlib import Path

path = str(Path(__file__).parent.absolute())

__datasets__ = {
    "sceneflow": sceneflow.SceneFlowDataset,
    "kitti": kitti.KITTIDataset
}

__datapath__ = {
    "sceneflow": path + "/dataset/SceneFlow/",
    "kitti": path + "/dataset/kitti2015v2/"
}

__filenames__ = {
    "sceneflow_train": path + "/filenames/sceneflow_train.txt",
    "sceneflow_test": path + "/filenames/sceneflow_test.txt",
    "kitti_train": path + "/filenames/kitti15_val.txt",
    "kitti_test": path + "/filenames/kitti15_train.txt"
}