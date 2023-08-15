import numpy as np
from typing import TYPE_CHECKING, Union, List, NamedTuple

from collections import defaultdict

from gafo.operator import Operator

if TYPE_CHECKING: from gafo.algebra import Algebra

scalar = Union[int, float]

class MultiVector:
    __slots__ = 'values', 'blades', 'algebra', 'op', 'device'

    def __init__(self, values: Union[List, np.ndarray], grades:Union[List, np.ndarray], algebra: 'Algebra', device: str='Numpy', dtype: type=float):
        self.algebra = algebra

        # There should a better way of doing this 
        self.blades = defaultdict(list)
        for i, grade in enumerate(grades):
            self.blades[grade].append(i)
        self.blades = dict(self.blades)

        self.values = values.astype(np.float32) if isinstance(values, np.ndarray) else np.array(values).astype(np.float32)
        self.device = device
        self.op = Operator(self, device)

    def numpy(self): return self.values
    def to(self, device: str): self.device = device; self.op = Operator(self, device)
    def blade(self, *grades): idxs = [idx for grade in grades for idx in self.blades[grade]]; return self.values[idxs]
    def blade_select(self, *grades): return self.algebra.blades_init(self.blade(*grades), [*grades])

    def extract(self):
        return self.op.lazy_device_data()
    
    def __repr__(self):
        return f'MV<values={self.values}, device={self.device}>'
    __str__ = __repr__
    
    # Unary ops
    def exp(self) -> 'MultiVector': return self.algebra.exp(self)
    def log(self) -> 'MultiVector': return self.algebra.log(self) 
    def relu(self) -> 'MultiVector': return self.algebra.relu(self)
    def sin(self) -> 'MultiVector': return self.algebra.sin(self) 
    def sqrt(self) -> 'MultiVector': return self.algebra.sqrt(self)
    def sigmoid(self) -> 'MultiVector': return self.algebra.sigmoid(self)
    def cos(self) -> 'MultiVector': return self.algebra.cos(self) 
    def tan(self) -> 'MultiVector': return self.algebra.tan(self) 
    def __neg__(self):
        return self * -1  

    # Binary ops
    def __add__(self, other: Union[float, int, 'MultiVector']) -> 'MultiVector': return self.algebra.add(self, other)
    
    def __radd__(self, other):
        if isinstance(other, scalar): return self + other
        else: raise f'Operation not supported with {type(other)} object'

    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        if isinstance(other, scalar): return -self + other
        else: raise f'Operation not supported with {type(other)} object'
    add = __add__

    def __mul__(self, other: Union[float, int, 'MultiVector']) -> 'MultiVector':
        if isinstance(other, MultiVector):
            return self.algebra.geo_prod(self, other)
        else:
            return self.algebra.mul(self, other)
    
    def __pow__(self, other: scalar): return self.algebra.pow(self, other)
    
    pow = __pow__

    def __rmul__(self, other):
        if isinstance(other, scalar): return self * other
        else: raise f'Operation not supported with {type(other)} object'

    def mul(self, other): return self.algebra.mul(self, other)

    def __xor__(self, other: 'MultiVector') -> 'MultiVector': return self.algebra.ext_prod(self, other)
    

    def __matmul__(self, other: 'MultiVector') -> 'MultiVector': return self.algebra.inner_prod(self, other)

    def __truediv__(self, other: Union[float, int]):
        assert other != 0, "Division by 0 is not allowed"
        return self.algebra.mul(self, 1 / other)

    def __rtruediv__(self, other: Union[float, int]):
        if isinstance(other, scalar): return self.algebra.mul( self.pow(-1), other)
        else: raise f'Operation not supported with {type(other)} object'
