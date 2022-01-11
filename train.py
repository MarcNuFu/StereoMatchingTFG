import gc
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

import dataloaders.dataloader as dataloader
import utils.tensorboard_printer as tensor_printer
import utils.device_manager as device_manager
import utils.argument_parser as argument_parser
from models import __models__
from utils.train_utils import *


def process_sample(model, maxdisp, sample, device, selected_model):
    imgL, imgR, disp_gt = get_sample_images(sample, device)
    
    mask = (disp_gt < maxdisp) & (disp_gt > 0)
    mask.detach_()

    if selected_model == "DispNetV2test":
      recon = model(imgL, imgR) 
      disps = recon.clone().detach()
    else:
      recon, disps = model(imgL, imgR)

    loss = calculate_loss(disps, disp_gt, mask)

    del disps

    return recon, loss, imgL, imgR, disp_gt, mask


def train_epoch(model, device, TrainImgLoader, optimizer, logger, epoch_idx, maxdisp, selected_model):
    model.train()
    total_train_loss = 0

    for batch_idx, sample in tqdm(enumerate(TrainImgLoader), total=len(TrainImgLoader)):
        optimizer.zero_grad()

        recon, loss, imgL, imgR, disp_gt, mask = process_sample(model, maxdisp, sample, device, selected_model+"train")

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if logger is not None:
            save_outputs(loss.item(), batch_idx, epoch_idx, logger, imgL, imgR, recon, disp_gt, len(TrainImgLoader), mask, maxdisp, 'train')

    total_train_loss = total_train_loss / len(TrainImgLoader)
    if logger is not None:
        tensor_printer.print_info_epoch(logger, 'trainEpoch', total_train_loss, imgL, imgR, recon, disp_gt, epoch_idx)

    return total_train_loss


def test_epoch(model, device, TestImgLoader, logger, epoch_idx, maxdisp, selected_model):
    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for batch_idx, sample in tqdm(enumerate(TestImgLoader), total=len(TestImgLoader)):
            recon, loss, imgL, imgR, disp_gt, mask = process_sample(model, maxdisp, sample, device, selected_model+"test")
            total_test_loss += loss.item()

            if logger is not None:
                save_outputs(loss.item(), batch_idx, epoch_idx, logger, imgL, imgR, recon, disp_gt, len(TestImgLoader), mask, maxdisp, 'test')

        total_test_loss = total_test_loss / len(TestImgLoader)
        if logger is not None:
            tensor_printer.print_info_epoch(logger, 'testEpoch', total_test_loss, imgL, imgR, recon, disp_gt, epoch_idx)

        return total_test_loss


def train():
    print('Preparing training\n')
    args = argument_parser.get_arguments()
    device = device_manager.select_device()
    logger = SummaryWriter(args.logdir)

    TrainImgLoader = dataloader.get_train_dataloader(args.dataset, args.batchsize, args.workers_train)
    TestImgLoader = dataloader.get_test_dataloader(args.dataset, args.batchsize, args.workers_test)

    model = __models__[args.model]
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    
    best_loss = args.checkloss

    if args.load_model != '':
        print("Loading model {}".format(args.load_model))
        if torch.cuda.is_available():
          model.load_state_dict(torch.load(args.load_model))
        else:
          model.load_state_dict(torch.load(args.load_model, map_location=torch.device('cpu')))
          
    print('\nStarting training\n')
    for epoch in range(args.start_epoch, args.total_epochs):
        #if  epoch == 600 or epoch == 1200 or epoch == 1800: #To DispNetV2
          #optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
          #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
          #for g in optimizer.param_groups:
            #g['lr'] = args.learnrate

        weights = set_weight_per_epoch(epoch, args.total_epochs, args.model, args.dataset)
        print(f'\nEpoch {epoch} - Weights = ', weights)
        
        # Train
        loss_train = train_epoch(model, device, TrainImgLoader, optimizer, logger, epoch, args.maxdisp, args.model)
        print(f'Train - Epoch:{epoch}, Loss:{loss_train:.4f}')
        gc.collect()

        # Test
        loss_test = test_epoch(model, device, TestImgLoader, logger, epoch, args.maxdisp, args.model)
        print(f'Test  - Epoch:{epoch}, Loss:{loss_test:.4f}')
        gc.collect()
        
        if loss_test < best_loss:
          best_loss = loss_test
          print('\nSaving best model to ./Vitis/build/float_model/' + args.pth_name + '_best.pth')
          print('\nSaving best model for old Pytorch versions to ./Vitis/build/float_model/' + args.pth_name + '_best_old.pth')
          torch.save(model.state_dict(), './Vitis/build/float_model/' + args.pth_name + '_best.pth')
          torch.save(model.state_dict(), './Vitis/build/float_model/' + args.pth_name + '_best_old.pth', _use_new_zipfile_serialization=False)

        # Scheduler step to decrease lr
        #scheduler.step()

    logger.flush()
    logger.close()
    print('\nTraining ended')

    print('\nSaving model to ./Vitis/build/float_model/' + args.pth_name + '.pth')
    print('\nSaving model for old Pytorch versions to ./Vitis/build/float_model/' + args.pth_name + '_old.pth')
    torch.save(model.state_dict(), './Vitis/build/float_model/' + args.pth_name + '.pth')
    torch.save(model.state_dict(), './Vitis/build/float_model/' + args.pth_name + '_old.pth', _use_new_zipfile_serialization=False)
    print('\nModel saved')


if __name__ == '__main__':
    train()
