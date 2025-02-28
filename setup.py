from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# List of .pyx files
extensions = [
    Extension("featureExtractor", ["featureExtractor.pyx"]),
    Extension("NoiseFilter", ["NoiseFilter.pyx"]),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
    include_dirs=[np.get_include()],
)