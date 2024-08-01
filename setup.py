from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="block_quant",
    ext_modules=[
        CUDAExtension(
            "block_quant_cuda",
            ["quant_block.cu"],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)