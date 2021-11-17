import dataloaders.dataloader as dataloader
import utils.device_manager as device_manager
import utils.argument_parser as argument_parser


def main():
    args = argument_parser.get_arguments()

    device = device_manager.select_device()

    TrainImgLoader, TestImgLoader = dataloader.get_dataloaders(args.path_dataset,
                                                               args.batchsize,
                                                               args.workers_train,
                                                               args.workers_test)

    return


if __name__ == '__main__':
    main()
