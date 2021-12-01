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
import train as train
from models import __models__

DIVIDER = '-----------------------------------------'




def quantize(build_dir, quant_mode, batchsize, selected_model, workers_test, dataset, device):

  dset_dir = build_dir + '/dataset'
  float_model = build_dir + '/float_model'
  quant_model = build_dir + '/quant_model' 

  # load trained model
  model = __models__[selected_model]
  model = model.to(device)
  
  path_pth = os.path.join(float_model,'f_model.pth')  
  model.load_state_dict(torch.load(path_pth, map_location=torch.device('cpu')))


  # force to merge BN with CONV for better quantization accuracy
  optimize = 1

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  rand_in = torch.randn([batchsize, 6, 384, 1248])
  quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
  quantized_model = quantizer.quant_model
  print("\nThe quantized model is:\n"+str(quantized_model))
  
  # data loader
  TestImgLoader = dataloader.get_test_dataloader(dataset, batchsize, workers_test)  

  # evaluate 
  print("Forwarding with converted model")
  criterion = nn.MSELoss()
  train.test_epoch(quantized_model, device, TestImgLoader, criterion)

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
  
  quantize(args.build_dir, args.quant_mode, args.batchsize, args.model, args.workers_test, args.dataset, device)

  return



if __name__ == '__main__':
    run_main()
