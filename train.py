import dataloaders2.dataloader as dataloader
import utils.print_outputs as printer 
from models import __models__

import torch
import torch.nn as nn
import torch.optim as optim


def train(args, device):                                                             
    TrainImgLoader = dataloader.get_train_dataloader(args.path_dataset, args.batchsize, args.workers_train)
                                          
    model = __models__[args.model].to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
     
    num_epochs = args.epochs
    outputs = []
    trainLoss=[]
    
    for epoch in range(num_epochs):
        for batch_idx, sample in enumerate(TrainImgLoader):
            imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
                
            model.train()
            img = torch.cat([imgL, imgR], dim=1)
            img = img.to(device)
            disp_true = disp_gt.unsqueeze(1)
            disp_true = disp_true.to(device)
            
            optimizer.zero_grad()
            
            recon = model(img)
            
            loss = criterion(recon, disp_true)
                       
            loss.backward()
            optimizer.step()
    
        loss = loss / len(TrainImgLoader)
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        trainLoss.append(loss.item())
        outputs.append((epoch, imgL, imgR, recon, disp_gt))  
    

    torch.save(model, "model.pth")
    
    printer.print_loss_curve(trainLoss, args.output_filename)
    printer.print_output(outputs, args.epochs_printed, num_epochs, args.output_filename)