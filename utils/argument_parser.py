import argparse
from models import __models__
from dataloaders import __datasets__

DIVIDER = '-----------------------------------------'


def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument('--output_filename', type=str,   default='backbone',               help='Filename to save outputs. Default is backbone')
    ap.add_argument('--dataset',         type=str,   default='sceneflow',              help='Used dataset. Default is sceneflow', choices=__datasets__.keys())
    ap.add_argument('--batchsize',       type=int,   default=1,                        help='Training batchsize. Must be an integer. Default is 1')
    ap.add_argument('--epochs',          type=int,   default=3,                        help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('--workers_train',   type=int,   default=1,                        help='Number of workers in train DataLoader. Must be an integer. Default is 1')
    ap.add_argument('--workers_test',    type=int,   default=1,                        help='Number of workers in test DataLoader. Must be an integer. Default is 1')
    ap.add_argument('--epochs_printed',  type=int,   default=1,                        help='Output epoch printed/saved in plot. Must be an integer. Default is 1')
    ap.add_argument('--learnrate',       type=float, default=0.001,                    help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    ap.add_argument('--model',                       default='AutoencoderBackbone',    help='Select a model structure', choices=__models__.keys())
    args = ap.parse_args()

    print_arguments(args)

    return args


def print_arguments(args):
    print('\n' + DIVIDER)
    print('Command line options:')
    print('--dataset            : ', args.dataset)
    print('--batchsize          : ', args.batchsize)
    print('--learnrate          : ', args.learnrate)
    print('--epochs             : ', args.epochs)
    print('--workers_train      : ', args.workers_train)
    print('--workers_test       : ', args.workers_test)
    print('--epochs_printed     : ', args.epochs_printed)
    print('--output_filename    : ', args.output_filename)
    print('--model              : ', args.model)
    print(DIVIDER + '\n')
