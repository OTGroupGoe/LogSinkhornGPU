from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='LogSinkhornGPU',
    version='0.3.0',
    install_requires=['torch', 'pykeops'],
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='LogSinkhornGPU.backend',
            sources=[
                'LogSinkhornGPU/backend/logsumexp.cpp',
                'LogSinkhornGPU/backend/logsumexp_kernel.cu'
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Ismael Medina',
    author_email='ismael.medina@cs.uni-goettingen.de',
    description='LogSinkhorn routines in the GPU',
    keywords='Optimal transport logsinkhorn GPU',
    url='https://github.com/OTGroupGoe/LogSinkhornGPU'
)
