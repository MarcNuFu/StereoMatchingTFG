import dataloaders.dataloader as dataloader
import utils.tensorboard_printer as tensor_printer 
from tqdm import tqdm
from models import __models__
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import torch.nn.functional as F
import numpy as np


def get_sample_images(sample, device):
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity'] 
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    disp_gt = disp_gt.to(device)
    
    return imgL, imgR, disp_gt
    

def train_epoch(model, device, TrainImgLoader, optimizer, logger, epoch_idx, criterion, maxdisp, batchsize):
    model.train()
    total_train_loss = 0
    
    for batch_idx, sample in tqdm(enumerate(TrainImgLoader), total=len(TrainImgLoader)):
        #Prepare data
        global_step = len(TrainImgLoader) * epoch_idx + batch_idx   
        imgL, imgR, disp_gt = get_sample_images(sample, device)
        mask = (disp_gt < maxdisp) & (disp_gt > 0)
        mask.detach_()  
             
        #Train  
        optimizer.zero_grad()             
        recon = model(imgL,imgR)
        loss = criterion(recon[mask], disp_gt[mask])
        loss.backward()      
        optimizer.step()
        
        #Print results
        total_train_loss += loss
        tensor_printer.save_loss(logger, 'train', loss, global_step)
        if (batch_idx % 150) == 0: 
          tensor_printer.save_images(logger, 'train', imgL, imgR, recon, disp_gt, batch_idx)
    
    
    total_train_loss = total_train_loss/len(TrainImgLoader)    
    return imgL, imgR, recon, disp_gt, total_train_loss
    
    
def test_epoch(model, device, TestImgLoader, logger, epoch_idx, criterion, maxdisp, batchsize):
    model.eval()
    total_test_loss = 0
    
    with torch.no_grad():
      for batch_idx, sample in tqdm(enumerate(TestImgLoader), total=len(TestImgLoader)):
          #Prepare data
          global_step = len(TestImgLoader) * epoch_idx + batch_idx        
          imgL, imgR, disp_gt = get_sample_images(sample, device)
          mask = (disp_gt < maxdisp) & (disp_gt > 0)
          mask.detach_()
          
          #Test          
          recon = model(imgL, imgR)                 
          loss = criterion(recon[mask], disp_gt[mask]) 
          
          #Print results
          total_test_loss += loss        
          tensor_printer.save_loss(logger, 'test', loss, global_step)
          if (batch_idx % 50) == 0: 
            tensor_printer.save_images(logger, 'test', imgL, imgR, recon, disp_gt, batch_idx)
      
            
      total_test_loss = total_test_loss/len(TestImgLoader)
      return imgL, imgR, recon, disp_gt, sample, total_test_loss


def train(args, device):
    print('Preparing training\n')
    logger = SummaryWriter(args.logdir)
                                            
    TrainImgLoader = dataloader.get_train_dataloader(args.dataset, args.batchsize, args.workers_train)
    TestImgLoader = dataloader.get_test_dataloader(args.dataset, args.batchsize, args.workers_test)  
                                       
    model = __models__[args.model]
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
   
    if args.load_model != '':
      print("Loading model {}".format(args.load_model))
      model.load_state_dict(torch.load(args.load_model))
    
    print('\nStarting training\n')   
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch}')    
    
        #Train
        imgL, imgR, recon, disp_gt, loss = train_epoch(model, device, TrainImgLoader, optimizer, logger, epoch, criterion, args.maxdisp, args.batchsize)
        tensor_printer.print_info_epoch(logger, 'trainEpoch', loss, imgL, imgR, recon, disp_gt, epoch)
        print(f'Train - Epoch:{epoch}, Loss:{loss.item():.4f}')
        gc.collect()
        
        #Test
        imgL, imgR, recon, disp_gt, sample, loss = test_epoch(model, device, TestImgLoader, logger, epoch, criterion, args.maxdisp, args.batchsize)
        tensor_printer.print_info_epoch(logger, 'testEpoch', loss, imgL, imgR, recon, disp_gt, epoch)
        print(f'Test  - Epoch:{epoch}, Loss:{loss.item():.4f}')
        gc.collect()
        
        #Scheduler step to decrease lr
        scheduler.step()
        
    logger.flush()
    logger.close() 
    print('\nTraining ended')
    
    print('\nSaving model to ./Vitis/build/float_model/'+ args.pth_name + '.pth')
    torch.save(model.state_dict(), './Vitis/build/float_model/' + args.pth_name + '.pth')
    print('\nModel saved')
    
    
if __name__ == '__main__':
    train()