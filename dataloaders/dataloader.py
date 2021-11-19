from . import KITTIloader2015 as lt
from . import KITTILoader as DA
import torch
import matplotlib.pyplot as plt


def get_dataloaders(path, batchsize, workers_train, workers_test):
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(path)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
        batch_size=batchsize, shuffle=True, num_workers=workers_train, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=batchsize, shuffle=False, num_workers=workers_test, drop_last=False)

    return TrainImgLoader, TestImgLoader


def get_dataloader_values_range(dataloader):
    imgL, imgR, disp_L = get_dataloader_first_images(dataloader)
    return torch.min(imgL), torch.max(imgL)


def get_dataloader_images_size(dataloader):
    imgL, imgR, disp_L = get_dataloader_first_images(dataloader)
    return imgL.size(), imgR.size(), disp_L.size()


def get_dataloader_first_images(dataloader):
    dataiter = iter(dataloader)
    imgL, imgR, disp_L = dataiter.next()
    return imgL, imgR, disp_L


def show_images(dataloader):
    imgL, imgR, disp_L = get_dataloader_first_images(dataloader)
    plt.figure()
    print(imgL[0].size())
    plt.imshow(disp_L[0])
    plt.figure()
    plt.imshow(imgR[0].permute(1, 2, 0))
    plt.figure()
    plt.imshow(imgL[0].permute(1, 2, 0))
    plt.show()
