from collections import defaultdict
import numpy as np
import math


# This file is a miminimal and pedagogical implementation of the library
# We will be using primitive Numpy operations as a backend for the moment


# Bitwise operations for computing basis products and signs

# This function counts the number of 1-bits in a number's binary string 
# Very useful for determining grade
def count_ones(n):
    return bin(n).count('1')

# As it sounds, it will help determing the sign 
def count_swaps(a, b):
    a = a >> 1
    _sum = 0
    while a != 0:
        _sum += count_ones(a & b)
        a = a >> 1
    return _sum

def ones_positions(n):
    pos = []
    index = 0
    while n:
        if n & 1:
            pos.append(index)
        n >>= 1
        index += 1
    return pos

def bit_len(n):
    return len(bin(n))-2

# Array helpers
# This is intended to be replaced with backend operations

def get_elements(A, indeces):
    return [A[i] for i in indeces]

# This function computes the cayley matrix between two basis
# TODO: Handle 0 in signature (projective)
def cayley(dim, signature):
    cayley = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    exterior_cayley = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)] 
    inner_cayley = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            k, mulbits= i ^ j, i & j
            swaps, sigs = count_swaps(i,j) + get_elements(signature, ones_positions(mulbits)).count('-') 
            sign = 1 if sigs & 1 == 0 else -1
            cayley[i][j][k] = sign
            
            if i != 0 :
                exterior_cayley[i][j][k] = sign
            else:
                inner_cayley[i][j][k] = sign

    return cayley, exterior_cayley, inner_cayley

# Implementation of geometric product by blades

# Simple multivector class where values get stored by grade
class MultiVector:
    def __init__(self, values, grades):
        self.blades = defaultdict(list)
        for grade, value in zip(grades, values):
            self.blades[grade].append(value)
        self.blades = dict(self.blades)

    def numpy(self):
        flat_values = []
        for grade, values in self.blades.items():
            flat_values = flat_values + values
        return np.array(flat_values)

def geo_product(a, b, cayley):
    x = np.tensordot(a, cayley, axes=[-1, 0])
    x = np.expand_dims(b, axis=len(b.shape) -1 ) @ x

    return x.squeeze(-2)

# Simple algebra class which stores signature, cayley and operations.

class Algebra():
    def __init__(self, p,q,r=0):


        self.N = p+q+r
        self.dim = 2**self.N

        self.grades = [count_ones(e) for e in range(self.dim)]

        self.signature = "+"*p + "-"*q + "0"*0
        self.cayley, self.exterior_cayley, self.inner_cayley  = cayley(self.dim, self.signature)
    
    def mv(self, values):
        '''
        Creates new MultiVector from array of values
        '''
        # Specifiy all values
        assert len(values) == self.dim, 'Length of values not matching with dimension'
        # TODO: Patch with zeros 
        return MultiVector(values, self.grades)

    # TODO: Implement this without numpy first ? 
    def product(self, mv1, mv2, cayley):
        '''
        Generic product function given by a cayley tensor.
        '''
        cayley = np.array(cayley)
        a, b = mv1.numpy(), mv2.numpy()        
        result = np.sum(cayley * a[..., :, None, None] * b[..., None, :, None], axis=(-3, -2))
        
        return self.mv(result.tolist())

    def gp(self, a, b):
        '''
        Geometric product between two MultiVectors using precomputed cayley.
        '''
        return self.product(a, b, self.cayley)

    def wp(self, a, b):
        '''
        Wedge (exterior) product between two MultiVectors using precomputed cayley.
        '''
        return self.product(a, b, self.exterior_cayley)
    
    def ip(self, a, b):
        '''
        Inner product between two MultiVectors using precomputed cayley.
        '''
        return self.product(a, b, self.inner_cayley)
# Test quaternion product 
def test_quaternion(result, a, b):
    expected_result = [
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] +- a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0], 
    ]

    assert np.array_equal(result, expected_result), f'The answer is incorrect. Result: {result}, Expected result{expected_result}'

    print(' \n Quaternion product test passed: The answer is correct \n')

def test_complex(result, a, b):
    expected_result = np.array([
        a[0]*b[0] - a[1]*b[1],
        a[0]*b[1] + a[1]*b[0] 
    ])

    assert np.array_equal(result, expected_result), f'The answer is incorrect. Result: {result}, Expected result{expected_result}'

    print(' \n Complex product test passed: The answer is correct \n')

def test_wedge_product(result, a, b, G):
    ab, ba = G.gp(a,b).numpy(), G.gp(b,a).numpy()
    expected_result = 0.5 * (ab - ba)

    assert np.array_equal(result, expected_result), f'The answer is incorrect. Result: {result}, Expected result{expected_result}'

    print(' \n Wedge product test passed: The answer is correct \n')

def test_inner_product(result, a, b, G):
    ab, ba = G.gp(a,b).numpy(), G.gp(b,a).numpy()
    expected_result = 0.5 * (ab + ba)

    assert np.array_equal(result, expected_result),f'The answer is incorrect. Result: {result}, Expected result{expected_result}'

    print(' \n Inner product test passed: The answer is correct \n')

def test_products(geometric, inner, exterior):
    assert np.array_equal(inner + exterior, geometric), f'Product test failed: The results are not consistent \n Geometric:{geometric}, Inner + Exterior {inner + exterior }'
    print(' \n Product test passed: The products match \n')
if __name__ == '__main__':

    # Expected functionality
    
    # Create Algebra Homeomorphic to complex numbers
    C = Algebra(0,1)

    # Initialize complex numbers
    a = C.mv([2,3])
    b = C.mv([5,7])

    # Perform geometric product
    result = C.gp(a, b)

    # Test
    test_complex(result.numpy(), a.numpy(), b.numpy())


    # Create Algebra Homeomorphic to quaterions
    H = Algebra(0,2)

    # Intitialize quaternion
    quat1 = H.mv([1,4,5,6])
    quat2 = H.mv([4,5,6,3])

    # Perform geometric product
    result = H.gp(quat1, quat2)
    exterior = H.wp(quat1, quat2)
    exterior2 = H.wp(quat2, quat1).numpy()
    inner = H.ip(quat1, quat2)

    ab, ba = H.gp(quat1,quat2).numpy(), H.gp(quat2,quat1).numpy()
    expected_result = 0.5 * (ab + ba)
    print(np.array(H.exterior_cayley))
    # Test
    #test_quaternion(result.numpy(), quat1.numpy(), quat2.numpy())
    test_products(result.numpy(), inner.numpy(), exterior.numpy())
    #test_inner_product(exterior.numpy(), quat1, quat2, H)
    #test_wedge_product(exterior.numpy(), quat1, quat2, H)
    #print(inner.numpy())
    #print(exterior.numpy())

