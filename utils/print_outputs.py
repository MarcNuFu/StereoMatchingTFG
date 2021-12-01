import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def print_figures(outputs, epoch, base_name, title):
    fig = plt.figure(figsize=(15, 15))
    plt.title(title + ' Outputs - Epoch: ' + str(epoch))
    plt.axis('off')
    
    imgsL = outputs[epoch][1].cpu()
    imgsR = outputs[epoch][2].cpu()
    recon = outputs[epoch][3].cpu().detach().numpy()
    disp_true = outputs[epoch][4].cpu()
    
    fig.add_subplot(2, 2, 1).title.set_text('Left image')
    imgL = clip_img(imgsL[0].permute(1, 2, 0))
    plt.imshow(imgL)
    
    fig.add_subplot(2, 2, 3).title.set_text('Right image')
    imgR = clip_img(imgsR[0].permute(1, 2, 0))
    plt.imshow(imgR)
    
    fig.add_subplot(2, 2, 2).title.set_text('Prediction')
    plt.imshow(recon[0].transpose(1, 2, 0))
    
    fig.add_subplot(2, 2, 4).title.set_text('Ground truth')
    plt.imshow(disp_true[0])
    
    plt.savefig('outputs/' + base_name + str(epoch) + '.png')
    plt.show()
    
    
def print_output(outputs, num_epochs_printed, total_epochs, base_name, title):
    for i in range(1, num_epochs_printed+1):
      print_figures(outputs, total_epochs-i, base_name, title)
      
      
def clip_img(img):
    max = torch.max(img).item()
    min = torch.min(img).item()
    img = img/(max-min) + (1-max/(max-min))
    return img
    
    
def print_loss_curve(trainLoss, base_name, title):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')    
    plt.title(title + ' Loss per Epoch')                        
    plt.plot(trainLoss)
    
    plt.savefig('outputs/' + base_name + '_loss.png')
    plt.show()