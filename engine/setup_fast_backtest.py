"""
Build script for the fast_backtest C++ extension.

Usage:
    python setup_fast_backtest.py build_ext --inplace

Prerequisites:
    pip install pybind11

This produces fast_backtest.<platform>.pyd (Windows) or .so (Linux/Mac)
in the same directory, importable as `import fast_backtest`.
"""

from setuptools import setup, Extension
import pybind11
import sys

extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17"]
else:
    extra_compile_args = ["-O3", "-std=c++17", "-ffast-math", "-march=native"]

ext = Extension(
    name="fast_backtest",
    sources=["fast_backtest.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args,
)

setup(
    name="fast_backtest",
    version="1.0.0",
    description="C++ accelerator for IG_Backtest daily compounding loop",
    ext_modules=[ext],
)
