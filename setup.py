from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Install for several CUDA architectures
# The +PTX should make the code work also for newer architectures, but
# this has not been tested (no newer architectures in GWDG cluster yet) 
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0 7.5+PTX"

setup(
    name='LogSinkhornGPU',
    version='0.3.0',
    install_requires=['torch', 'pykeops'],
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='LogSinkhornGPU.backend',
            sources=[
                'LogSinkhornGPU/backend/pybind.cpp',
                'LogSinkhornGPU/backend/logsumexp.cu',
                'LogSinkhornGPU/backend/domdec.cu'
            ],
            extra_compile_args={'nvcc': ['-O2', "--use_fast_math"]},
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Ismael Medina',
    author_email='ismael.medina@cs.uni-goettingen.de',
    description='LogSinkhorn routines in the GPU',
    keywords='Optimal transport logsinkhorn GPU',
    url='https://github.com/OTGroupGoe/LogSinkhornGPU'
)
