from . import KITTIloader2015 as lt
from . import KITTILoader as DA
import torch


def get_dataloaders(path, batchsize, workers_train, workers_test):
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(path)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
        batch_size=batchsize, shuffle=True, num_workers=workers_train, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=batchsize, shuffle=False, num_workers=workers_test, drop_last=False)

    return TrainImgLoader, TestImgLoader
