# Modified from
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py

import glob
import os
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension

version = "0.1.0"
package_name = "onevision_cpp_ops"


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "onevision_cpp_ops._CPP",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    setup(
        name=package_name,
        version=version,
        ext_modules=get_extensions(),
        packages=["onevision_cpp_ops"],
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )
