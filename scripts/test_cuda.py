import torch

if __name__ == '__main__':
    print('torch version: ', torch.__version__)
    print('torch file:    ', torch.__file__)
    print('cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print('device name:   ', torch.cuda.get_device_name(0))
