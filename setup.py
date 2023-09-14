import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.system('make -j%d' % os.cpu_count())

# Python interface
setup(
    name='LogSinkhornGPU_accurate',
    version='0.2.0',
    install_requires=['torch'],
    packages=['LogSinkhornGPU'],
    package_dir={'LogSinkhornGPU': './'},
    ext_modules=[
        CUDAExtension(
            name='LogSinkhornGPUBackend_accurate',
            include_dirs=['./'],
            sources=[
                'pybind/bind.cpp',
            ],
            libraries=['make_pytorch'],
            library_dirs=['objs'],
            # extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Ismael Medina',
    author_email='ismael.medina@cs.uni-goettingen.de',
    description='LogSinkhorn routines in the GPU',
    keywords='Optimal transport logsinkhorn GPU',
    url='https://github.com/OTGroupGoe/LogSinkhornGPU',
    zip_safe=False,
)
