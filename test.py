import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

import dataloaders.dataloader as dataloader


def clip_img(img):
    max = torch.max(img).item()
    min = torch.min(img).item()
    img = img/(max-min) + (1-max/(max-min))
    return img


def generate_images(dest_dir):                                           
  test_loader = dataloader.get_test_dataloader('kitti', 1, 0)  

  for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
    imgL = sample['left'].squeeze()
    imgR = sample['right'].squeeze()


    fn = sample["left_filename"][0]          
    name = fn.split('/')
            
    fn = os.path.join("testfotos/", '_'.join(name[2:])).replace(".png", "") + "/"
    os.mkdir(fn)
            
    imgL_np = (clip_img(imgL.permute(1, 2, 0))).data.cpu().numpy()
    io.imsave(fn+"imgL.png", img_as_ubyte(imgL_np), check_contrast=False)
            
    imgR_np = (clip_img(imgR.permute(1, 2, 0))).data.cpu().numpy()
    io.imsave(fn+"imgR.png", img_as_ubyte(imgR_np), check_contrast=False)        

  return
  
if __name__ == '__main__':
    generate_images('')