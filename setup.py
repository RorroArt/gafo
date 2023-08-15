import os
from setuptools import setup

readme = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(readme, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(
    name='gafo',
    version='0.0.1',
    description='Geometric Algebras Fast Operations just for you :)',
    author='Rodrigo Caridad',
    license='MIT',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=['gafo', 'gafo.utils', 'gafo.backends'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
    install_requires=['numpy'],
    python_requires='>=3.8',
    extras_require={
        'cuda': ["pycuda"],
        'linting': [
            "flake8",
            "pylint",
            "mypy",
            "pre-commit",
        ],
        'testing': [
            "torch",
            "pytest",
            "pytest-xdist",
            "types-PyYAML",
        ],
      },
    include_package_data=True
)