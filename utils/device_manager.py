import torch

DIVIDER = '-----------------------------------------'


def select_device():
    print('\n' + DIVIDER)
    
    if torch.cuda.device_count() > 0:
        print('You have', torch.cuda.device_count(), 'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print('    Device', str(i), ': ', torch.cuda.get_device_name(i))
        device = torch.device('cuda:0')
        print('Device 0 selected')
    else:
        device = torch.device('cpu')
        print('CPU selected')
        
    print(DIVIDER + '\n')
    
    return device


def is_cuda_available():
    return torch.cuda.is_available()
