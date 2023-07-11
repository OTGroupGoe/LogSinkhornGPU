import os
import sys
import torch


def check():
    print(f'which python3:   {sys.executable}')
    print(f'torch version:   {torch.__version__}')
    print(f'torch file:      {torch.__file__}')
    print(f'cuda available:  {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print('device name:    ', torch.cuda.get_device_name(i))
    print(f'LD_LIBRARY_PATH: {os.environ["LD_LIBRARY_PATH"]}\n')
    print('')


if __name__ == '__main__':
    check()
