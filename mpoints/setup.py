from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

# Run in the console (under Anaconda version): python setup.py build_ext --inplace

# setup(
#     ext_modules=cythonize("hybrid_hawkes_exp_likelihood.pyx"),
#     include_dirs=[numpy.get_include()]
# )

ext_modules = [
    Extension(name="hybrid_hawkes_exp_cython",
              sources=["hybrid_hawkes_exp_cython.pyx"],
              # comment this line when compiling on Windows
              libraries=["m"],
              extra_compile_args=["-ffast-math"]
              )
]

setup(
    name="hybrid_hawkes_exp_cython",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()])
