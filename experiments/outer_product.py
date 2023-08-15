# Outer product is broken with the cayley matrix 
# This is a different implementation of the outer product

# The goal is that given multivectors mv1, mv2
# mv1 ^ mv2 = - mv2 ^ mv1

# Imports
from gafo import Algebra, MultiVector
from gafo.utils import count_swaps    
import numpy as np

from typing import NamedTuple

class MVData(NamedTuple): 
    data: np.ndarray
    bitmaps: np.ndarray

def simplify(bases, result):
    simple_result = np.zeros(len(bases))
    for basis, value in result:
        idx = np.where(bases == basis)[0]
        simple_result[idx] += value
    return simple_result
    

# Computes per blade outer product
def blade_op(a, b): 
    a_scale, a_bitmap = a
    b_scale, b_bitmap = b

    if (a_bitmap & b_bitmap) != 0:
        return 0, 0

    bitmap = a_bitmap ^ b_bitmap
    swaps = count_swaps(a_bitmap, b_bitmap)
    sign =  1 if (swaps) & 1 == 0 else -1

    return (bitmap, sign * a_scale * b_scale)


# Computes the full outer product of two multivectors
def outer_product(mv1: MVData, mv2: MVData) -> MultiVector:
    result = []

    for a in zip(mv1.data, mv1.bitmaps):
        for b in zip(mv2.data, mv2.bitmaps):
            result.append(blade_op(a, b))
    
    return simplify(mv1.bitmaps, result)

if __name__ == '__main__':
    a = MVData(
        np.array([7,3,4,4,0,0,0,0]),
        np.array([0,1,2,4,3,5,6,7]) 
    )
    b = MVData(
        np.array([8,3,9,1,0,0,0,0]),
        np.array([0,1,2,4,3,5,6,7]) 
    )

    print(outer_product(a,b))
    print(outer_product(b,a))
