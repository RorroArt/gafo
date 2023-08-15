# Cayley is broken for inner and exterior products :() 
# So I created this file to explore the algorithm until we get it right :)

import numpy as np
from typing import NamedTuple, List

# Important bit utils that 
from gafo.utils import count_swaps, get_elements, ones_positions, count_ones

# Product operation. we will use it to test the product
def numpy_product( x: np.array, y: np.array, cayley: np.array):
        '''
        Generic product function given by a cayley tensor. 
        Taken from Alphafolds's quaternion product
        '''
        
        return np.sum(cayley * x[..., :, None, None] * y[..., None, :, None], axis=(-3, -2))


# Class to store the cayley information
class Cayley(NamedTuple):
    geometric: List
    exterior: List
    inner: List

def compute_cayley(signature, dim):
    '''
    Computes the cayley tensor for all products (Geometric, inner and exterior).
    '''
    cayley = np.zeros((dim,dim,dim))
    exterior_cayley = np.zeros((dim,dim,dim))
    inner_cayley = np.zeros((dim,dim,dim))
    for i in range(dim):
        for j in range(dim):
            k, mulbits= i ^ j, i & j
            swaps, sigs = count_swaps(i,j), get_elements(signature, ones_positions(mulbits)).count('-') 
            sign =  1 if (swaps + sigs) & 1 == 0 else -1
            cayley[i][j][k] = sign
            if mulbits == 0:
                exterior_cayley[i][j][k] = 1 if (swaps) & 1 == 0 else -1
            else:
                inner_cayley[i][j][k] = sign
    exterior_cayley = np.array(exterior_cayley)

    return Cayley(cayley, exterior_cayley, inner_cayley)


# Lest implememt a different operation for the outer product ;)

def outer_product(mv1, mv2) -> MultiVector:



if __name__ == '__main__':
    cayley = compute_cayley('+++', 8)
    r1, r2 = np.array([4, 5, 1, 5, 2, 3, 1, 3]), np.array([3, 6, 4, 6, 2, 1, 4, 3])
    print(numpy_product(r1, r2, cayley.exterior))
    print(numpy_product(r2, r1, cayley.exterior))