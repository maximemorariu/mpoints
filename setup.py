import setuptools
import numpy

ext_modules = [
    setuptools.Extension('mpoints.hybrid_hawkes_exp_cython',
                         sources=['mpoints/hybrid_hawkes_exp_cython.c'],
                         extra_compile_args=["-ffast-math"],
                         )
]

setuptools.setup(
    name="mpoints",
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
