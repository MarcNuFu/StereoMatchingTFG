'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Make the target folder
Copies images, application code and compiled xmodel to 'target'
'''

'''
Author: Mark Harvey
'''

import torch
import torchvision

import argparse
import os
import shutil
import sys
import cv2
import numpy as np
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte

sys.path.append('../')
import dataloaders.dataloader as dataloader


DIVIDER = '-----------------------------------------'


def clip_img(img):
    max = torch.max(img).item()
    min = torch.min(img).item()
    img = img/(max-min) + (1-max/(max-min))
    return img


def generate_images(dest_dir):                                           
  test_loader = dataloader.get_test_dataloader('kitti', 1, 0)  
  
  os.mkdir(dest_dir+"/imgL")
  os.mkdir(dest_dir+"/imgR")
  
  for batch_idx, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
    imgL = sample['left'].squeeze()
    imgR = sample['right'].squeeze()


    fn = sample["left_filename"][0]          
    name = fn.split('/')
            
    fnL = os.path.join(dest_dir+"/imgL", '_'.join(name[2:])).replace(".png", "") 
    fnR = os.path.join(dest_dir+"/imgR", '_'.join(name[2:])).replace(".png", "") 

    imgL_np = (clip_img(imgL.permute(1, 2, 0))).data.cpu().numpy()
    io.imsave(fnL+".png", img_as_ubyte(imgL_np), check_contrast=False)       
    
    imgR_np = (clip_img(imgR.permute(1, 2, 0))).data.cpu().numpy()
    io.imsave(fnR+".png", img_as_ubyte(imgR_np), check_contrast=False)  

  return


def make_target(build_dir,target,app_dir):

    dset_dir = build_dir + '/dataset'
    comp_dir = build_dir + '/compiled_model'
    target_dir = build_dir + '/target_' + target

    # remove any previous data
    shutil.rmtree(target_dir, ignore_errors=True)    
    os.makedirs(target_dir)

    # copy application code
    print('Copying application code from',app_dir,'...')
    shutil.copy(os.path.join(app_dir, 'app_mt.py'), target_dir)

    # copy compiled model
    model_path = comp_dir + '/Dispnet_' + target + '.xmodel'
    print('Copying compiled model from',model_path,'...')
    shutil.copy(model_path, target_dir)

    # create images
    dest_dir = target_dir + '/images'
    shutil.rmtree(dest_dir, ignore_errors=True)  
    os.makedirs(dest_dir)
    generate_images(dest_dir)


    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',  type=str,  default='build', help='Path to build folder. Default is build')
    ap.add_argument('-t', '--target',     type=str,  default='zcu102', choices=['zcu102','zcu104','u50','vck190'], help='Target board type (zcu102,zcu104,u50,vck190). Default is zcu102')
    ap.add_argument('-a', '--app_dir',    type=str,  default='application', help='Full path of application code folder. Default is application')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --build_dir    : ', args.build_dir)
    print (' --target       : ', args.target)
    print (' --app_dir      : ', args.app_dir)
    print('------------------------------------\n')


    make_target(args.build_dir, args.target, args.app_dir)


if __name__ ==  "__main__":
    main()
