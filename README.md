# Geometric Algebras Fast Operations (GAFO)

This is a small library to implement fast and flexible and differentiable geometric algebra operations. Initially it will be built on top of jax and numpy. However, it will eventually support backend such as C, CUDA, Triton, Metal, or OpenCL. 

## Goal

The principal purpose of the library is to be use for the implementations of neural networks that operate in geometriic algebras. Mainly the goal is to have an easy of implementing the ideas described in the following papers:

- Algebra Nets (Quaternion and Complex valued nets): https://arxiv.org/abs/2006.07360
- Clifford layers: https://arxiv.org/abs/2209.04934
- Geometric Algebra Transformers: https://arxiv.org/pdf/2305.18415.pdf

Then, the second main goal is to optimize the primitive operations of geometric algebras such as the geometric and exterior products.

# Usage 

## Instalation

1. Clone the repo
```
git clone <repo-link>
```
2. Install Requ


# References 

Citing your references is cool, this is why I wanted to provide the list of wonderful frameworks, libraries, papers, and repos that inspired this project.

- Efficient Implementation of Geometric Algebra: https://pure.uva.nl/ws/files/4375498/52687_fontijne.pdf
- Algorithmic structure for geometric algebra operators and application to quadric surface: https://pastel.hal.science/tel-02085820/document
- A Geometric Algebra Implementation using Binary Tree: https://hal.science/hal-01510078/document
- TFGA: https://github.com/RobinKa/tfga/tree/31350b9159626083c6f9ddfa8ac67a6c36cf5089
- Numga: https://github.com/EelcoHoogendoorn/numga
- Alphafold repo: https://github.com/deepmind/alphafold
- Tinygrad: https://github.com/tinygrad/tinygrad/tree/master

# TODO

- Initial Commit
   - Implement autograd using the GRAD wrapper 

- Add axis control and write good tests

- Compress and handle shapes commit
    - Projective GA 
    

- Mini Equi-MLP Commit (milestone)
    - Trainable MLP
    - Test Equivariant

- NN module Commit 
    - Equi-MLP layer
    - Haiku wrapper
    - Torch wrapper
    - Attention, LayerNorm, Clifford Convolution

- C Commit (Let the optimization Begin!!) (Beat numpy )
    - Test operations in C
    - Add C backend (hard)

- Metal Commit (beat C in Mac)
    - Operations in Metal 
    - Add Metal Backend

- Triton commit (beat jax and torch in Nvidia GPU)
    - Operations in Triton
    - Add Triton Backend
