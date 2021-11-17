import argparse

DIVIDER = '-----------------------------------------'


def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument('--path_dataset',  type=str,   default='./dataset/kitti2015v2/training/', help='Path to dataset. Default is KITTI2015')
    ap.add_argument('--batchsize',     type=int,   default=1,                               help='Training batchsize. Must be an integer. Default is 1')
    ap.add_argument('--epochs',        type=int,   default=3,                                 help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('--workers_train', type=int,   default=1,                                 help='Number of workers in train DataLoader. Must be an integer. Default is 1')
    ap.add_argument('--workers_test',  type=int,   default=1,                                 help='Number of workers in test DataLoader. Must be an integer. Default is 1')
    ap.add_argument('--learnrate',     type=float, default=0.001,                             help='Optimizer learning rate. Must be floating-point value. Default is 0.001')

    args = ap.parse_args()

    print_arguments(args)

    return args


def print_arguments(args):
    print('\n' + DIVIDER)
    print('Command line options:')
    print('--build_dir        : ', args.path_dataset)
    print('--batchsize        : ', args.batchsize)
    print('--learnrate        : ', args.learnrate)
    print('--epochs           : ', args.epochs)
    print('--workers_train    : ', args.workers_train)
    print('--workers_test     : ', args.workers_test)
    print(DIVIDER + '\n')
