import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

sys.path.append('../')
import utils.device_manager as device_manager
import utils.argument_parser_vitis as argument_parser
import dataloaders.dataloader as dataloader
from models import __models__
from tqdm import tqdm
from models.DispNet import DispNet

DIVIDER = '-----------------------------------------'


def get_sample_images(sample, device):
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity'] 
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    
    disp_gt = torch.unsqueeze(disp_gt, 1)
    disp_gt = disp_gt.to(device)
    
    return imgL, imgR, disp_gt
    
      
def test_model(model, device, TestImgLoader, maxdisp):
    model.eval()
    
    with torch.no_grad():
      for batch_idx, sample in tqdm(enumerate(TestImgLoader), total=len(TestImgLoader)):
          #Prepare data      
          imgL, imgR, disp_gt = get_sample_images(sample, device)
          
          mask = (disp_gt < maxdisp)
          mask.detach_()
          
          #Test         
          recon, disps = model(imgL, imgR)
                 
          
          #PSMNET
          
          loss = lossEPE(recon, disp_gt, mask)
          
          #loss = model_loss(recon, disp_gt, mask)  
          
          break  
          
          
def lossEPE(output, target, mask):
  b, _, h, w = target.size()
  upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
  return F.smooth_l1_loss(upsampled_output[mask], target[mask], reduction='mean')
  


def quantize(build_dir, quant_mode, batchsize, selected_model, workers_test, dataset, device, maxdisp, pth_name, args):

  dset_dir = build_dir + '/dataset'
  float_model = build_dir + '/float_model'
  quant_model = build_dir + '/quant_model' 

  # load trained model
  
  model = __models__[args.model].to(device)
  
  path_pth = os.path.join(float_model, pth_name + '.pth')
  
  print("Loadin model\n")
  if torch.cuda.is_available():
    model.load_state_dict(torch.load(path_pth))
  else:
    model.load_state_dict(torch.load(path_pth, map_location=torch.device('cpu')))
  
  print("Model loaded\n")
    
  # force to merge BN with CONV for better quantization accuracy
  optimize = 1

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  imgL = torch.randn([batchsize, 3, 384, 1280])
  imgR = torch.randn([batchsize, 3, 384, 1280])

  print("Quantizer")
  
  quantizer = torch_quantizer(quant_mode, model, (imgL, imgR), output_dir=quant_model) 
  quantized_model = quantizer.quant_model
  print("\nThe quantized model is:\n"+str(quantized_model))
   
  # data loader
  TestImgLoader = dataloader.get_test_dataloader(dataset, batchsize, workers_test)  
  
  # evaluate 
  print("Forwarding with converted model")
  test_model(quantized_model, device, TestImgLoader, maxdisp)

  # export config
  if quant_mode == 'calib':
    print("Exporting quantized configuration")
    quantizer.export_quant_config()
  if quant_mode == 'test':
    print("Exporting xmodel at " + str(quant_model))
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
  return



def run_main():
  args = argument_parser.get_arguments()
  
  device = device_manager.select_device()

  quantize(args.build_dir, args.quant_mode, args.batchsize, args.model, args.workers_test, args.dataset, device, args.maxdisp, args.pth_name, args)

  return



if __name__ == '__main__':
    run_main()
