import math
from typing import Union, List, NamedTuple
from gafo.utils import count_ones, count_swaps, ones_positions, bit_len
from gafo.utils import get_elements, zeros_array, ones_array

from gafo.multivector import MultiVector

import numpy as np

import gafo.diff_ops as ops

pi = math.pi
scalar = Union[int, float]

# Here we store the cayley tensors for the products
# Cayley tensors are list now but we should exploit their sparsity in the future
class Cayley(NamedTuple):
    geometric: List
    exterior: List
    inner: List
    

class Algebra:
    __slots__ = 'N', 'dim', 'grades', 'basis', 'signature', 'cayley', 'exterior_cayley', 'inner_cayley', 'device'
    
    def __init__(self, p: int, q: int, r: int = 0, device='Numpy'):
        self.N = p+q+r
        self.dim = 2**self.N

        self.basis = sorted([e for e in range(self.dim)], key=count_ones)
        self.grades = [count_ones(e) for e in range(self.dim)]
        self.grades.sort()

        self.signature = "+"*p + "-"*q + "0"*0
        self.cayley = self.compute_cayley(self.signature)

        self.device = device
    
    def to(self, device):
        self.device = device

    def mv(self, values: Union[List[float], List[int], np.ndarray]):
        '''
        Creates new MultiVector from array of values
        '''
        # Specifiy all values
        assert len(values) == self.dim, 'Length of values not matching with dimension'
        # TODO: Patch with zeros 
        return MultiVector(values, self.grades, algebra=self, device=self.device)

    def ones(self) -> MultiVector: return MultiVector(ones_array(self.dim), self.grades, algebra=self, device=self.device)
    def zeros(self) -> MultiVector: return MultiVector(zeros_array(self.dim), self.grades, algebra=self, device=self.device) 

    def blades_init( self, 
        values: Union[List[float], List[int], np.ndarray],
        blades: Union[List[int], np.ndarray] ) -> MultiVector:
        
        full_values = []

        idx = 0
        for grade in self.grades:
            if grade in blades:
                full_values.append(values[idx])
                idx += 1
            else: 
                full_values.append(0)

        return MultiVector(full_values, self.grades, algebra=self, device=self.device )
    
    def compute_cayley(self, signature):
        '''
        Computes the cayley tensor for all products (Geometric, inner and exterior).
        '''
        cayley = np.zeros((self.dim, self.dim, self.dim))
        exterior_cayley = np.zeros((self.dim, self.dim, self.dim))
        inner_cayley =  np.zeros((self.dim, self.dim, self.dim))
        for i, a in enumerate(self.basis):
            for j, b in enumerate(self.basis):
                c, mulbits= a ^ b, a & b
                k = self.basis.index(c)
                sigs, swaps = count_swaps(a,b), get_elements(signature, ones_positions(mulbits)).count('-') 
                sign =  1 if (swaps + sigs) & 1 == 0 else -1
                cayley[i,j,k] = sign
                if (mulbits) == 0:
                    exterior_cayley[i,j,k] = sign
                else:
                    inner_cayley[i,j,k] = sign

        return Cayley(cayley, exterior_cayley, inner_cayley)

    # Unary ops

    def exp(self, x: MultiVector) -> MultiVector: return ops.Exp.apply(x)
    def log(self, x: MultiVector) -> MultiVector: return ops.Log.apply(x)
    def relu(self, x: MultiVector) -> MultiVector: return ops.Relu.apply(x)
    def sin(self, x: MultiVector) -> MultiVector: return ops.Sin.apply(x)
    def sqrt(self, x: MultiVector) -> MultiVector: return ops.Sqrt.apply(x)
    def sigmoid(self, x: MultiVector) -> MultiVector: return ops.Sigmoid.apply(x)
    def cos(self, x: MultiVector) -> MultiVector: return self.sin(pi/2 - self)
    def tan(self, x:MultiVector) -> MultiVector: return self.sin(x)/self.cos(x)

    # Binary ops

    def add(self, x: Union[float, int, MultiVector] , y:Union[float, int, MultiVector]) -> MultiVector: 
        if isinstance(x, scalar) and isinstance(y, scalar): raise 'At least on input has to be a Multivector'
        elif isinstance(x, scalar): n = x; x = y; y = Scalar.mv([n]) 
        elif isinstance(y, scalar): y = Scalar.mv([y])
        
        return ops.Add.apply(x, y)
    
    def mul(self, x:Union[float, int, MultiVector], y:Union[float, int, MultiVector]) -> MultiVector: 
        if isinstance(x, scalar) and isinstance(y, scalar): raise 'At least on input has to be a Multivector'
        elif isinstance(x, scalar): n = x; x = y; y = Scalar.mv([n]) 
        elif isinstance(y, scalar): y = Scalar.mv([y])
        
        return ops.Mul.apply(x, y)

    def pow(self, x:MultiVector, y: scalar) -> MultiVector:
        assert isinstance(y, scalar) and isinstance(y, scalar),'Pow is only supported for types MultiVector ** Scalar'
        return ops.Pow.apply(x, Scalar.mv([y]))

    def geo_prod(self, x:MultiVector, y:MultiVector) -> MultiVector:
        assert isinstance(x, MultiVector) and isinstance(y, MultiVector), 'Inputs have to be multivectors'
        
        return ops.Product.apply(x, y, self.cayley.geometric)
    
    def ext_prod(self, x:MultiVector, y:MultiVector) -> MultiVector: 
        assert isinstance(x, MultiVector) and isinstance(y, MultiVector), 'Inputs have to be multivectors'

        return ops.Product.apply(x, y, self.cayley.exterior)
    
    def inner_prod(self, x:MultiVector, y:MultiVector) -> MultiVector: 
        assert isinstance(x, MultiVector) and isinstance(y, MultiVector), 'Inputs have to be multivectors'
        
        return ops.Product.apply(x, y, self.cayley.inner)
    
    wedge_prod = outer_prod = ext_prod 
    dot_prod = inner_prod
    
    
# Common algebras

Scalar = Algebra(0,0)
Complex = Algebra(0,1)
Quaternions = Algebra(0, 2)
R3 = Algebra(3,0)