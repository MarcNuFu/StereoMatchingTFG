import torch
import torch.nn.functional as F
import utils.tensorboard_printer as tensor_printer 
from utils.metrics import *
 
weights = ()   


def lossEPE(output, target, mask):
  b, _, h, w = target.size()
  upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
  return F.smooth_l1_loss(upsampled_output[mask], target[mask], reduction='mean')
  

def calculate_loss(disps, target, mask):
  total_loss = 0
  
  if torch.is_tensor(disps):
    total_loss = lossEPE(disps, target, mask)
  else:
    for disp, weight in zip(disps, weights):
      total_loss += weight * lossEPE(disp, target, mask)

  return total_loss
  
  
def set_weight_per_epoch(epoch, total_epochs):
  global weights
  
  if epoch == 0:
    weights = (0, 0, 0, 0, 0.2, 1)
    
  elif epoch >= 0.1*total_epochs and epoch < 0.15*total_epochs:
    weights = (0, 0, 0, 0.2, 1, 0.4)  
    
  elif epoch >= 0.15*total_epochs and epoch < 0.2*total_epochs:
    weights = (0, 0, 0.2, 1, 0.4, 0)  
    
  elif epoch >= 0.2*total_epochs and epoch < 0.25*total_epochs:
    weights = (0, 0.2, 1, 0.4, 0, 0)  
    
  elif epoch >= 0.25*total_epochs and epoch < 0.3*total_epochs:
    weights = (0.2, 1, 0.4, 0, 0, 0)  
    
  elif epoch >= 0.3*total_epochs and epoch < 0.4*total_epochs:
    weights = (1, 0.4, 0, 0, 0, 0)  
     
  elif epoch >= 0.4*total_epochs:
    weights = (1, 0, 0, 0, 0, 0)     
  
  return weights
  
def get_sample_images(sample, device):
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity'] 
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    
    disp_gt = torch.unsqueeze(disp_gt, 1)
    disp_gt = disp_gt.to(device)
    
    return imgL, imgR, disp_gt
    
    
def save_outputs(loss, batch_idx, epoch_idx, logger, imgL, imgR, recon, disp_gt, lenLoader, mask, maxdisp, mode):    
    if (batch_idx % 25) == 0: 
      tensor_printer.save_images(logger, mode, imgL, imgR, recon, disp_gt, batch_idx)
                    
    global_step = lenLoader * epoch_idx + batch_idx  
    scalar_outputs = {"loss": loss}
    scalar_outputs["D1"] = D1_metric(recon.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1))
    scalar_outputs["EPE"] = EPE_metric(recon.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1))
    scalar_outputs["Thres1"] = Thres_metric(recon.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 1.0) 
    scalar_outputs["Thres2"] = Thres_metric(recon.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 2.0) 
    scalar_outputs["Thres3"] = Thres_metric(recon.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 3.0)
    
    tensor_printer.save_metrics(logger, mode, scalar_outputs, global_step)
    del scalar_outputs