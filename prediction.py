import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

import dataloaders.dataloader as dataloader
import utils.device_manager as device_manager
import utils.argument_parser as argument_parser
import utils.color_map as printer
from models import __models__
from utils.train_utils import *
from utils.tensorboard_printer import disp_error_image_func

def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper
    
    
@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")
        
            
def clip_img(img):
    max = torch.max(img).item()
    min = torch.min(img).item()
    img = img/(max-min) + (1-max/(max-min))
    return img

'''
def get_sample_images(sample, device):
    imgL, imgR = sample['left'], sample['right']
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    
    return imgL, imgR
'''       
    
def test(args, device):
    print('Generating disparity maps\n')
                                            
    #PredImgLoader = dataloader.get_pred_dataloader(args.dataset, args.batchsize, args.workers_test)  
    PredImgLoader = dataloader.get_test_dataloader(args.dataset, args.batchsize, args.workers_test)
                                           
    model = __models__[args.model]
    model = model.to(device)
    
    print("Loading model {}".format(args.load_model))
    if torch.cuda.is_available():
      model.load_state_dict(torch.load(args.load_model))
    else:
      model.load_state_dict(torch.load(args.load_model, map_location=torch.device('cpu')))
    
    model.eval()
    
    print('\nStarting predictions\n')
    with torch.no_grad():
      for batch_idx, sample in tqdm(enumerate(PredImgLoader), total=len(PredImgLoader)):
          imgL, imgR, disp_gt = get_sample_images(sample, device)   
          recon, _ = model(imgL, imgR)
                   
          recon = torch.squeeze(recon,1)
          disp_gt = torch.squeeze(disp_gt,1)
          
          error_map = disp_error_image_func(recon, disp_gt)

          #Save output         
          disp_est_np = tensor2numpy(recon)  
          disp_gt_np = tensor2numpy(disp_gt)            
          top_pad_np = tensor2numpy(sample["top_pad"])
          right_pad_np = tensor2numpy(sample["right_pad"])
          left_filenames = sample["left_filename"]           
          
          for disp_est, top_pad, right_pad, fn, imgL2, recon2, error_map2, disp_gt2 in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames, imgL, recon, error_map, disp_gt_np):
            assert len(disp_est.shape) == 2           

            
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            name = fn.split('/')
            
            fn = os.path.join("predictions/", '_'.join(name[2:])).replace(".png", "") + "/"
            os.mkdir(fn)
            
            imgL_np = tensor2numpy(clip_img(imgL2.permute(1, 2, 0)))
            io.imsave(fn+"imgL.png", img_as_ubyte(imgL_np), check_contrast=False)
            
            error_map_np = tensor2numpy(clip_img(error_map2.permute(1, 2, 0)))
            io.imsave(fn+"d1.png", img_as_ubyte(error_map_np), check_contrast=False)
            
            disp_est1 = printer.kitti_colormap(disp_est, 192)           
            cv2.imwrite(fn+"colormap.png", disp_est1)
            
            disp_est2 = np.round(disp_est * 256).astype(np.uint16)
            io.imsave(fn+"prediction.png", disp_est2, check_contrast=False)
            
            disp_est3 = np.round(disp_gt2 * 256).astype(np.uint16)
            io.imsave(fn+"dispgt.png", disp_est3, check_contrast=False)

    print('\nDone')
       
    
if __name__ == '__main__':
    args = argument_parser.get_arguments()
    device = device_manager.select_device()
    test(args, device)