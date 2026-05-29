from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qwen_rms_scale_swiglu_cuda",
    ext_modules=[
        CUDAExtension(
            name="qwen_rms_scale_swiglu_cuda",
            sources=[
                "csrc/qwen_rms_scale_swiglu/qwen_rms_scale_swiglu_torch.cpp",
                "csrc/qwen_rms_scale_swiglu/kernel/qwenRmsScaleSwigluKernel.cu",
            ],
            include_dirs=[
                "csrc/qwen_rms_scale_swiglu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-std=c++17",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
