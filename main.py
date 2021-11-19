import dataloaders.dataloader as dataloader
import utils.device_manager as device_manager
import utils.argument_parser as argument_parser

import models.autoencoder as autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def main():
    args = argument_parser.get_arguments()

    device = device_manager.select_device()

    #160 imagenes de la carpeta de training para cada loader
    TrainImgLoader, TestImgLoader = dataloader.get_dataloaders(args.path_dataset,
                                                               args.batchsize,
                                                               args.workers_train,
                                                               args.workers_test)

    model = autoencoder.Autoencoder().to(device)
    #dataloader.show_images(TrainImgLoader)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3, 
                                 weight_decay=1e-5)
                                 
    num_epochs = 3
    outputs = []
    trainLoss=[]
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (imgL, imgR, disp_true) in enumerate(TrainImgLoader):
            imgL = imgL.to(device)
            recon = model(imgL)
            loss = criterion(recon, imgL)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, imgL, recon))                            

    #dataloader.show_images(TestImgLoader)

    imgs = outputs[num_epochs-1][1].cpu()
    recon = outputs[num_epochs-1][2].cpu().detach().numpy()
    plt.figure()
    plt.imshow(imgs[0].permute(1, 2, 0))
    plt.figure()
    plt.imshow(recon[0].transpose(1, 2, 0))

    
    plt.show()


if __name__ == '__main__':
    main()
