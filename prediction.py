import utils.device_manager as device_manager
import utils.argument_parser as argument_parser
import utils.color_map as printer
import dataloaders.dataloader as dataloader
from tqdm import tqdm
from models import __models__
import torch
import numpy as np
import os
import cv2
from skimage import io
from skimage import img_as_ubyte
import matplotlib.pyplot as plt


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
        
        
def get_sample_images(sample, device):
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity'] 
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    disp_gt = disp_gt.to(device)
    
    return imgL, imgR, disp_gt    
 
    
def clip_img(img):
    max = torch.max(img).item()
    min = torch.min(img).item()
    img = img/(max-min) + (1-max/(max-min))
    return img
    
    
def test(args, device):
    print('Generating disparity maps\n')
                                            
    TestImgLoader = dataloader.get_test_dataloader(args.dataset, args.batchsize, args.workers_test)  
                                       
    model = __models__[args.model](args.maxdisp)
    model = model.to(device)
   
    
    print("Loading model {}".format(args.load_model))
    model.load_state_dict(torch.load(args.load_model))
    
    with torch.no_grad():
      for batch_idx, sample in tqdm(enumerate(TestImgLoader), total=len(TestImgLoader)):
          model.eval()
          
          #Prepare data
          imgL, imgR, disp_gt = get_sample_images(sample, device)
          
          #Test          
          recon = model(imgL, imgR)[-1]
          
          #Save output         
          disp_est_np = tensor2numpy(recon)      
          top_pad_np = tensor2numpy(sample["top_pad"])
          right_pad_np = tensor2numpy(sample["right_pad"])
          left_filenames = sample["left_filename"]           
          
          for disp_est, top_pad, right_pad, fn, imgL2, recon2 in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames, imgL, recon):

            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            name = fn.split('/')
            
            fn = os.path.join("predictions/", '_'.join(name[2:])).replace(".png", "") + "/"
            os.mkdir(fn)
            
            imgL_np = tensor2numpy(clip_img(imgL2.permute(1, 2, 0)))
            io.imsave(fn+"imgL.png", img_as_ubyte(imgL_np), check_contrast=False)
            
            if args.dataset == 'kitti':
              disp_est1 = printer.kitti_colormap(disp_est)           
              cv2.imwrite(fn+"colormap.png", disp_est1)
              disp_est2 = np.round(disp_est * 256).astype(np.uint16)
              io.imsave(fn+"prediction.png", disp_est2, check_contrast=False)
            
            else:
              recon_np = tensor2numpy(recon2.unsqueeze(0).permute(1, 2, 0))
              recon_np = np.round(recon_np * 256).astype(np.uint16)
              io.imsave(fn+"recon.png", recon_np, check_contrast=False)
            
            


    print('\nDone')
       
    
if __name__ == '__main__':
    args = argument_parser.get_arguments()
    device = device_manager.select_device()
    test(args, device)