import argparse
from models import __models__
from dataloaders import __datasets__

DIVIDER = '-----------------------------------------'


def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument('--dataset',            type=str,   default='sceneflow',              help='Used dataset. Default is sceneflow', choices=__datasets__.keys())
    ap.add_argument('--pth_name',           type=str,   default='model',                  help='Name of saved pth model. Default is model')
    ap.add_argument('--load_model',         type=str,   default='',                       help='Path of model to load, if empty model is not loaded. Default is empty')
    ap.add_argument('--batchsize',          type=int,   default=1,                        help='Training batchsize. Must be an integer. Default is 1')
    ap.add_argument('--total_epochs',       type=int,   default=3,                        help='Number of TOTAL training epochs. Must be an integer. Default is 3')
    ap.add_argument('--start_epoch',        type=int,   default=0 ,                       help='Strarting epoch. Must be an integer. Default is 0')
    ap.add_argument('--workers_train',      type=int,   default=1,                        help='Number of workers in train DataLoader. Must be an integer. Default is 1')
    ap.add_argument('--workers_test',       type=int,   default=1,                        help='Number of workers in test DataLoader. Must be an integer. Default is 1')
    ap.add_argument('--maxdisp',            type=int,   default=192,                      help='Maximum disparity')
    ap.add_argument('--learnrate',          type=float, default=0.0001,                   help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    ap.add_argument('--model',                          default='AutoencoderBackbone',    help='Select a model structure', choices=__models__.keys())
    ap.add_argument('--logdir',                         default='stereo',                 help='The directory to save logs and checkpoints')
    ap.add_argument('--milestones',                     default=[200,220,240,260],        help='Epochs at which learning rate is divided by 2')  
    args = ap.parse_args()

    print_arguments(args)

    return args


def print_arguments(args):
    print('\n' + DIVIDER)
    print('Command line options:')
    print('--dataset            : ', args.dataset)
    print('--pth_name           : ', args.pth_name)
    print('--load_model         : ', args.load_model)
    print('--batchsize          : ', args.batchsize)
    print('--learnrate          : ', args.learnrate)
    print('--total_epochs       : ', args.total_epochs)
    print('--start_epoch        : ', args.start_epoch)
    print('--workers_train      : ', args.workers_train)
    print('--workers_test       : ', args.workers_test)
    print('--model              : ', args.model)
    print('--logdir             : ', args.logdir)
    print('--maxdisp            : ', args.maxdisp)
    print('--milestones         : ', args.milestones)
    print(DIVIDER + '\n')
