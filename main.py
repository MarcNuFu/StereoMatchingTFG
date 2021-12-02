import utils.device_manager as device_manager
import utils.argument_parser as argument_parser
from train import train

    
def main():
    
    args = argument_parser.get_arguments()

    device = device_manager.select_device()

    train(args, device)



if __name__ == '__main__':
    main()
