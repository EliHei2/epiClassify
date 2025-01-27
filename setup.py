from pathlib import Path

from setuptools import setup, find_packages

long_description = Path('README.rst').read_text('utf-8')

try:
    from GNNProject import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''

setup(name='GNNProject',
      version='0.1.0',
      description='Graph-Structured Inductive Bias in Deep Learning with Application to Cell Type Classification',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/e-sollier/DL2020/',
      author=__author__,
      author_email=__email__,
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
          l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
      ],
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          "License :: OSI Approved :: MIT License",
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
      ],
      doc=[
          'sphinx',
          'sphinx_rtd_theme',
          'sphinx_autodoc_typehints',
          'typing_extensions; python_version < "3.8"',
      ],
      )
