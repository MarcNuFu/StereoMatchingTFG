import argparse
from models import __models__
from dataloaders import __datasets__

DIVIDER = '-----------------------------------------'


def get_arguments():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-d',  '--build_dir',  type=str, default='build',                              help='Path to build folder. Default is build')
    ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('--dataset',           type=str, default='kitti',                              help='Used dataset. Default is KITTI', choices=__datasets__.keys())
    ap.add_argument('-b',  '--batchsize',  type=int, default=8,                                    help='Testing batchsize - must be an integer. Default is 8')
    ap.add_argument('--workers_test',      type=int, default=1,                                    help='Number of workers in test DataLoader. Must be an integer. Default is 1')
    ap.add_argument('--maxdisp',           type=int, default=192,                                  help='Maximum disparity')
    ap.add_argument('--model',                       default='DispNet',                            help='Select a model structure', choices=__models__.keys())
    ap.add_argument('--pth_name',          type=str, default='model',                              help='Name of saved pth model. Default is model')
    args = ap.parse_args()
    
    print_arguments(args)
    
    return args


def print_arguments(args):
    print('\n'+DIVIDER)
    print(' Command line options:')
    print('--build_dir     : ', args.build_dir)
    print('--quant_mode    : ', args.quant_mode)
    print('--dataset       : ', args.dataset)
    print('--batchsize     : ', args.batchsize)
    print('--workers_train : ', args.workers_test)
    print('--model         : ', args.model)
    print('--pth_name      : ', args.pth_name)
    print('--maxdisp       : ', args.maxdisp)
    print(DIVIDER + '\n')