import dataloaders.dataloader as dataloader
import utils.print_outputs as printer 
from models import __models__

import torch
import torch.nn as nn
import torch.optim as optim


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
    

def train(args, device):                                                             
    TrainImgLoader = dataloader.get_train_dataloader(args.dataset, args.batchsize, args.workers_train)
                                     
    model = __models__[args.model]
    #if device.type == 'cuda':
      #model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
     
    num_epochs = args.epochs
    outputs = []
    trainLoss=[]

    print(f'Starting training - Iteration per epoch: {len(TrainImgLoader)}\n')
    
    for epoch in range(num_epochs):
        for batch_idx, sample in enumerate(TrainImgLoader):
            model.train()
              
            imgL, imgR, img, disp_gt, disp_true = get_sample_images(sample, device)
            
            optimizer.zero_grad()
            
            recon = model(img)
            
            loss = criterion(recon, disp_true)
                       
            loss.backward()
            optimizer.step()
    
        loss = loss / len(TrainImgLoader)
        trainLoss.append(loss.item())
        outputs.append((epoch, imgL, imgR, recon, disp_gt))  
        
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    
    print('\nTraining ended')
    torch.save(model, "model.pth")
    
    printer.print_loss_curve(trainLoss, args.output_filename)
    printer.print_output(outputs, args.epochs_printed, num_epochs, args.output_filename)