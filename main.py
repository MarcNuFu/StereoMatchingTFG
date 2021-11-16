import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import dataloader.KITTIloader2015 as lt
import dataloader.KITTILoader as DA


def select_device():
    # use GPU if available   
    if (torch.cuda.device_count() > 0):
        print('You have',torch.cuda.device_count(),'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print('    Device',str(i),': ',torch.cuda.get_device_name(i))
        device = torch.device('cuda:0')
        print('Device 0 selected')
    else:
        device = torch.device('cpu')
        print('CPU selected')
        
    return device


def load_data(batchsize, numWorkersTrain, numWorkersTest):
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader("./dataset/kitti2015v2/training/")

    TrainImgLoader = torch.utils.data.DataLoader(
             DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
             batch_size= batchsize, shuffle= True, num_workers= numWorkersTrain, drop_last=False)
    
    TestImgLoader = torch.utils.data.DataLoader(
             DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
             batch_size= batchsize, shuffle= False, num_workers= numWorkersTest, drop_last=False)
                   
    return(TrainImgLoader, TestImgLoader)


def main():
    #Parametros
    batchsize = 1
    num_epochs = 3
    learnrate = 0.001
    numWorkersTrain = 1
    numWorkersTest = 1
    betaOptim=(0.9, 0.999)
    cudaAv = torch.cuda.is_available()

    
    #device = select_device()
    #TrainImgLoader, TestImgLoader = load_data(batchsize, numWorkersTrain, numWorkersTest)

      
    return

if __name__ == '__main__':
    main()
