from torch.utils.data import DataLoader
from dataloaders2.dataset import KITTIDataset as KITTIDataset

def get_dataloaders(path, batchsize, workers_train, workers_test):   
    TrainImgLoader = get_train_dataloader(path, batchsize, workers_train)
    
    TestImgLoader = get_test_dataloader(path, batchsize, workers_test)
    
    return TrainImgLoader, TestImgLoader
    
    
def get_train_dataloader(path, batchsize, workers_train):
    train_dataset = KITTIDataset(path, "./dataloaders2/kitti15_train.txt", True)
    
    TrainImgLoader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=workers_train, drop_last=True)
    
    return TrainImgLoader
    
    
def get_test_dataloader(path, batchsize, workers_test):
    test_dataset = KITTIDataset(path, "./dataloaders2/kitti15_val.txt", False)
    
    TestImgLoader = DataLoader(test_dataset, batchsize, shuffle=False, num_workers=workers_test, drop_last=False)
    
    return TestImgLoader