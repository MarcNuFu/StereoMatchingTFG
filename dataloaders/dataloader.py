from torch.utils.data import DataLoader
from dataloaders import __datasets__, __datapath__, __filenames__

def get_dataloaders(dataset, batchsize, workers_train, workers_test):   
    TrainImgLoader = get_train_dataloader(batchsize, workers_train, dataset)
    
    TestImgLoader = get_test_dataloader(batchsize, workers_test, dataset)
    
    return TrainImgLoader, TestImgLoader
    
    
def get_train_dataloader(dataset, batchsize, workers_train):
    StereoDataset = __datasets__[dataset]
    filename = __filenames__[dataset + "_train"]
    path = __datapath__[dataset]
    
    train_dataset = StereoDataset(path, filename, True)
    
    TrainImgLoader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=workers_train, drop_last=True)
    
    return TrainImgLoader
    
    
def get_test_dataloader(dataset, batchsize, workers_test):
    StereoDataset = __datasets__[dataset]
    filename = __filenames__[dataset + "_test"]
    path = __datapath__[dataset]
    
    test_dataset = StereoDataset(path, filename, False)
    
    TestImgLoader = DataLoader(test_dataset, batchsize, shuffle=True, num_workers=workers_test, drop_last=False)
    
    return TestImgLoader