import setuptools
import numpy

ext_module = setuptools.Extension('mpoints.hybrid_hawkes_exp_cython',
                                  sources=['mpoints/hybrid_hawkes_exp_cython.c'],
                                  extra_compile_args=["-ffast-math"])

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    ext_modules= [ext_module],
    include_dirs=[numpy.get_include()],
    name="mpoints",
    version="0.1",
    author="Maxime Morariu-Patrichi, Mikko Pakkanen",
    author_email="maximemorariu@me.com",
    description="Simulate and estimate state-dependent Hawkes processes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.maximemorariu.com",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)