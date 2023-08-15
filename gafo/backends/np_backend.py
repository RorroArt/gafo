import numpy as np
from typing import List

from gafo.ops import Ops, Interpreter

# Define the product function
def numpy_product( x: np.array, y: np.array, cayley: np.array):
        '''
        Generic product function given by a cayley tensor. 
        Taken from Alphafolds's quaternion product
        '''
        
        return np.sum(cayley * x[..., :, None, None] * y[..., None, :, None], axis=(-3, -2))

class NumpyInterpreter(Interpreter):
    lookup_ops = {
        Ops.EXP: np.exp,
        Ops.LOG: np.log,
        Ops.RELU: lambda x: np.maximum(x, 0),
        Ops.SIN: np.sin,
        Ops.SQRT: np.sqrt,
        Ops.SIGMOID: lambda x:  1/(1 + np.exp(-x)),
        Ops.ADD: np.add,
        Ops.MUL: np.multiply,
        Ops.PRODUCT: numpy_product,
        Ops.POW: np.float_power
    }

    def to_device(self, data: np.ndarray) -> np.ndarray: return np.array(data)
    def from_device(self, array: np.ndarray) -> np.ndarray : return array.tolist()