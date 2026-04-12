from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Performance-oriented compiler flags
# NVCC flags
nvcc_extra_args = [
    '-O3',
    '--use_fast_math',
    '-std=c++17',
    # Ampere
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_86,code=sm_86',
    # Hopper
    '-gencode=arch=compute_90a,code=sm_90a',
]

# CXX flags
cxx_extra_args = [
    '-O3',
    '-std=c++17',
]

setup(name='spade_hy',
      ext_modules=[
          CUDAExtension(name='spade_hy',
                        sources=['summarize_mmc.cu'],
                        extra_compile_args={
                            'cxx': cxx_extra_args,
                            'nvcc': nvcc_extra_args
                        })
      ],
      cmdclass={'build_ext': BuildExtension})
