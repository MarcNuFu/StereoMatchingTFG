import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import dataloader.KITTIloader2015 as lt
import dataloader.KITTILoader as DA

def main():
    batchsize = 1
    num_epochs = 3
    learnrate = 0.001
    numWorkersTrain = 1
    numWorkersTest = 1
    betaOptim=(0.9, 0.999)
    cudaAv = torch.cuda.is_available()

    print(cudaAv)
    return

if __name__ == '__main__':
    main()
