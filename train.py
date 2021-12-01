import dataloaders.dataloader as dataloader
import utils.print_outputs as printer 
from models import __models__

import torch
import torch.nn as nn
import torch.optim as optim
import gc

def get_sample_images(sample, device):
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity'] 
    
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    disp_gt = disp_gt.to(device)
    
    img = torch.cat([imgL, imgR], dim=1)
    img = img.to(device)
    
    disp_true = disp_gt.unsqueeze(1)
    disp_true = disp_true.to(device)
    
    return imgL, imgR, img, disp_gt, disp_true
    

def train_epoch(model, device, TrainImgLoader, optimizer, criterion):
    for batch_idx, sample in enumerate(TrainImgLoader):
        model.train()
        
        imgL, imgR, img, disp_gt, disp_true = get_sample_images(sample, device)
        
        optimizer.zero_grad()
        recon = model(img)
        loss = criterion(recon, disp_true)  
        loss.backward()
        optimizer.step()
    
    loss = loss / len(TrainImgLoader)
    return imgL, imgR, recon, disp_gt, loss
    
    
def test_epoch(model, device, TestImgLoader, criterion):
    model.eval()

    with torch.no_grad():
      for batch_idx, sample in enumerate(TestImgLoader):        
          imgL, imgR, img, disp_gt, disp_true = get_sample_images(sample, device)
          
          recon = model(img)
          loss = criterion(recon, disp_true) 
          
      
      loss = loss / len(TestImgLoader)
      return imgL, imgR, recon, disp_gt, loss


def train(args, device):                                                             
    TrainImgLoader = dataloader.get_train_dataloader(args.dataset, args.batchsize, args.workers_train)
    TestImgLoader = dataloader.get_test_dataloader(args.dataset, args.batchsize, args.workers_test)  
                                       
    model = __models__[args.model]
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
     
    num_epochs = args.epochs
    outputs_train = []
    outputs_test = []
    loss_train = []
    loss_test = []
    
    print(f'Starting training - Iteration per epoch: {len(TrainImgLoader)}\n')
    
    for epoch in range(num_epochs):
        #Train
        imgL, imgR, recon, disp_gt, loss = train_epoch(model, device, TrainImgLoader, optimizer, criterion)
        loss_train.append(loss.item())
        outputs_train.append((epoch, imgL, imgR, recon, disp_gt))  
        print(f'Train - Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        gc.collect()
        
        #Test
        imgL, imgR, recon, disp_gt, loss = test_epoch(model, device, TestImgLoader, criterion)
        loss_test.append(loss.item())
        outputs_test.append((epoch, imgL, imgR, recon, disp_gt))  
        print(f'Test  - Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        gc.collect()
    
    print('\nTraining ended')
    torch.save(model.state_dict(), "./Vitis/build/float_model/model.pth")
    
    #Print train outputs
    printer.print_loss_curve(loss_train, args.output_filename + '_train', "Train")
    printer.print_output(outputs_train, args.epochs_printed, num_epochs, args.output_filename + '_train', "Train")
    
    #Print test outputs
    printer.print_loss_curve(loss_test, args.output_filename + 'test', "Test")
    printer.print_output(outputs_test, args.epochs_printed, num_epochs, args.output_filename + '_test', "Test")
    
if __name__ == '__main__':
    train()