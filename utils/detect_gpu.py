# External libraries
import torch

def detect_gpu() -> None:
    device_num = torch.cuda.current_device()
    print(
        torch.cuda.device(device_num),
        torch.cuda.device_count(),
        torch.cuda.get_device_name(device_num),
        torch.cuda.is_available()
    )


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Running on GPU')
    else:
        device = torch.device('cpu')
        print('Running on CPU')
    
    return device
