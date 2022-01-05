import torch
import torchvision.utils as vutils
import utils.color_map as color
from skimage import io
import numpy as np        
import cv2
import os



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
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")
        
        
def save_metrics(logger, mode_tag, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for tag, values in scalar_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            scalar_name = '{}/{}'.format(mode_tag, tag)
            # if len(values) > 1:
            scalar_name = scalar_name + "_" + str(idx)
            logger.add_scalar(scalar_name, value, global_step)
            
            
def print_info_epoch(logger, mode_tag, loss, imgL, imgR, recon, disp_gt, epoch):
    save_loss(logger, mode_tag, loss, epoch)
    save_images(logger, mode_tag, imgL, imgR, recon, disp_gt, epoch)


def save_loss(logger, mode_tag, loss, step):
    logger.add_scalar(mode_tag, loss, step) 
                               

def save_color_map(logger, mode_tag, recon, sample, step):
    #recon = recon[-1]

   
    if isinstance(recon, torch.Tensor):
      recon_np = recon[-1].data.cpu().numpy()
      
    if isinstance(sample["top_pad"], torch.Tensor):
      top_np = sample["top_pad"][-1].data.cpu().numpy()  
      
    if isinstance(sample["right_pad"], torch.Tensor):
      right_np = sample["right_pad"][-1].data.cpu().numpy()  
      
    left_filename = sample["left_filename"][-1]
  
    recon_np = np.array(recon_np[top_np:, :-right_np], dtype=np.float32) 
    color_map = color.kitti_colormap(recon_np)  
    cv2.imwrite('/home/marcnf/StereoMatchingTFG/outputs/test'+ str(step)+'_color.png', color_map)

    rounded = (recon_np * 256).astype(np.uint16)
    io.imsave('/home/marcnf/StereoMatchingTFG/outputs/test' + str(step)+'_round.png', rounded)
      
    #logger.add_image(mode_tag + "/" + color_map, vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True), step)
      
                             
def save_image(logger, mode_tag, img_name, img, step):
    if isinstance(img, torch.Tensor):
      img = img.data.cpu().numpy()
      
    if len(img.shape) == 3:
      img = img[:, np.newaxis, :, :]
      
    img = torch.from_numpy(img)[-1]
    logger.add_image(mode_tag + "/" + img_name, vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True), step)


def save_images(logger, mode_tag, imgL, imgR, recon, disp_gt, step):   
    recon = torch.squeeze(recon,1)
    disp_gt = torch.squeeze(disp_gt,1)
    
    save_image(logger, mode_tag, "imgL", imgL, step)
    save_image(logger, mode_tag, "imgR", imgR, step)
    save_image(logger, mode_tag, "disp_gt", disp_gt, step)

    if isinstance(recon, list):
      recon = recon[-1]
      
    save_image(logger, mode_tag, "recon", recon, step)
    
    error_map = disp_error_image_func(recon, disp_gt)
    save_image(logger, mode_tag, "error_map", error_map, step)
   
   
def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols

    
def disp_error_image_func(D_est_tensor, D_gt_tensor, abs_thres=3., rel_thres=0.05, dilate_radius=1):
    error_colormap = gen_error_colormap()
    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    B, H, W = D_gt_np.shape

    # valid mask
    mask = D_gt_np > 0

    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)

    # get colormap
    cols = error_colormap

    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    error_image[np.logical_not(mask)] = 0.

    # show color tag in the top-left corner of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2])))