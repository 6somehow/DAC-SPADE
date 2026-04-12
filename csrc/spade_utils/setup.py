import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.cuda import get_device_capability

python_include = subprocess.check_output([
    'python', '-c', "import sysconfig; print(sysconfig.get_path('include'))"
]).decode().strip()
torch_include = subprocess.check_output([
    'python', '-c',
    "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))"
]).decode().strip()
print('Python include:', python_include)
print('Torch include directories:', torch_include)

cuda_flags = [
    '-DNDEBUG', '-Xcompiler=-Wno-psabi', '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda', '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler', '--use_fast_math', '-std=c++20',
    '-O3', f'-I{python_include}',
    '-DTORCH_COMPILE'
] + torch_include.split()
cpp_flags = ['-std=c++20', '-O3']

cuda_flags += [
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_89,code=sm_89',
]

device_capability = get_device_capability()
device_capability_str = f'{device_capability[0]}{device_capability[1]}'
cuda_flags += [
    f'-gencode=arch=compute_{device_capability_str},code=sm_{device_capability_str}'
]
if device_capability == (9, 0):
    cuda_flags += [
        '-gencode=arch=compute_90a,code=sm_90a'
    ]

source_files = [
    'spade_utils.cpp',
    'utils/cossim.cu',
    'utils/mask_to_bsr.cu',
    'utils/scatter.cu',
    'utils/static_sink_diag_set.cu',
]

setup(name='spade_utils',
      ext_modules=[
          CUDAExtension('spade_utils',
                        sources=source_files,
                        extra_compile_args={
                            'cxx': cpp_flags,
                            'nvcc': cuda_flags
                        },
                        libraries=['cuda'])
      ],
      cmdclass={'build_ext': BuildExtension})
