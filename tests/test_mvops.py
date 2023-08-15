import unittest

from typing import Union
from gafo import Algebra, MultiVector

from testing_env import TestGafo, value_test, array_test, dict_test, fn_test, np_fn_test
from testing_env import geometric_test, inner_wedge_test

import numpy as np
import torch
import torch.nn.functional as F

scalar = Union[int, float]

class TestMultivector(TestGafo):

    def test_init(self):
        # From Algebra 
        a, b = self.C.mv([4,6]), self.H.mv([-6,8,7,6])

        # test grades
        array_test(list(a.blades), [0,1])
        array_test(list(b.blades), [0,1,2])

        dict_test(a.blades, {0: [4], 1: [6]})
        dict_test(b.blades, {0: [-6], 1: [8,7], 2: [6]})

        # Custom
        custom_a, custom_b = MultiVector([4,6], [0,1], self.C), MultiVector([-6,8,7,6], [0,1,1,2], self.H)

        #test grades
        array_test(list(a.blades), list(custom_a.blades))
        array_test(list(b.blades), list(custom_b.blades))

        dict_test(a.blades, custom_a.blades)
        dict_test(b.blades, custom_b.blades)

    def test_flatten(self):
        a, b = self.C.mv([4,6]), self.H.mv([-6,8,7.2,6])
        ex_a, ex_b = np.array([4,6]), np.array([-6,8,7.2,6])

        array_test(b.flatten(), ex_b.tolist())
        array_test(a.flatten(), ex_a.tolist())
        array_test(a.numpy(), ex_a)
        array_test(b.numpy(), ex_b)
        

    def test_device_change(self):
        a, b = self.C.mv([4,6]), MultiVector([-6,8,.7,6], [0,1,1,2], self.H, device='Jax')
        # Numpy -> Torch
        a.to('Torch')
        value_test('Device', a.device, 'Torch')
        value_test('Device', a.op.device, 'Torch')

        # Torch -> Jax
        a.to('Jax')
        value_test('Device', a.device, 'Jax')
        value_test('Device', a.op.device, 'Jax')

        # From init (Jax)
        value_test('Device', b.device, 'Jax')
        value_test('Device', b.op.device, 'Jax')

        # Jax -> Numpy
        b.to('Numpy')
        value_test('Device', b.device, 'Numpy')
        value_test('Device', b.op.device, 'Numpy')

    def test_unary_ops(self):
        a, b = np.random.radn(2), np.random.randn(4) 
        
        def init_op(op, G):
            return lambda x: op(G.mv(x)).numpy()

        def torch_fn(fn):
            return lambda x: fn(torch.Tensor(x)).numpy()

        # Exp
        np_fn_test(init_op(lambda x: x.exp(), self.C), np.exp, (a))
        np_fn_test(init_op(lambda x: x.exp(), self.H), np.exp, (b))
        
        # Log
        np_fn_test(init_op(lambda x: x.log(), self.C), np.log, (a))
        np_fn_test(init_op(lambda x: x.log(), self.H), np.log, (b))
        # Relu
        np_fn_test(init_op(lambda x: x.relu(), self.C), torch_fn(F.relu), (a))
        np_fn_test(init_op(lambda x: x.relu(), self.H), torch_fn(F.relu), (b))
        # Sin
        np_fn_test(init_op(lambda x: x.sin(), self.C), np.sin, (a))
        np_fn_test(init_op(lambda x: x.sin(), self.H), np.sin, (b))
        # Cos
        np_fn_test(init_op(lambda x: x.cos(), self.C), np.cos, (a))
        np_fn_test(init_op(lambda x: x.cos(), self.H), np.cos, (b))
        # Tan
        np_fn_test(init_op(lambda x: x.tan(), self.C), np.tan, (a))
        np_fn_test(init_op(lambda x: x.tan(), self.H), np.tan, (b))
        # Sqrt
        np_fn_test(init_op(lambda x: x.sqrt(), self.C), np.sqrt, (a))
        np_fn_test(init_op(lambda x: x.sqrt(), self.H), np.sqrt, (b))
        # Sigmoid
        np_fn_test(init_op(lambda x: x.sigmoid(), self.C), torch_fn(torch.sigmoid), (a))
        np_fn_test(init_op(lambda x: x.sigmoid(), self.H), torch_fn(torch.sigmoid), (b))
        # Inv
        np_fn_test(init_op(lambda x: -x, self.C), lambda x: -x, (a))
        np_fn_test(init_op(lambda x: -y, self.H), lambda x: -x, (b))

    def test_unary_ops(self):
        a, b, c, d = np.random.randn(2),np.random.randn(2), np.random.randn(4), np.random.randn(4)
        
        def init_op(op, G):
            def convert(x): return x if isinstance(x, scalar) else G.mv(x)
            return lambda x, y: op(convert(x),convert(y)).numpy()

        # Add
        np_fn_test(init_op(lambda x, y: x+y, self.C), np.add, (a, b))
        np_fn_test(init_op(lambda x, y: x+y, self.H), np.add, (c, d))
        
        # Mul
        np_fn_test(init_op(lambda x, y: x.mul(y), self.C), np.multiply, (a, b))
        np_fn_test(init_op(lambda x, y: x.mul(y), self.H), np.multiply, (c, d))
        
        # Scalar Add
        np_fn_test(init_op(lambda x, y: x+y, self.C), np.add, (3, b))
        np_fn_test(init_op(lambda x, y: x+y, self.H), np.add, (c, 5))
        
        # Scalar Mul
        np_fn_test(init_op(lambda x, y: x*y, self.C), np.multiply, (4, b))
        np_fn_test(init_op(lambda x, y: x*y, self.H), np.multiply, (c, 7))
        
        # Div
        np_fn_test(init_op(lambda x, y: x/y, self.C), np.divide, (4, b))
        np_fn_test(init_op(lambda x, y: x/y, self.H), np.divide, (c, 7))
        
        # Subtraction
        np_fn_test(init_op(lambda x, y: x-y, self.C), np.subtract, (3, b))
        np_fn_test(init_op(lambda x, y: x-y, self.H), np.subtract, (c, 5))

    def test_product(self):
        def quat_prod(x, y):
            x, y = self.H.mv(x), self.H.mv(y) 
            return (x * y).numpy()

        def complex_prod(x, y):
            x, y = self.C.mv(x), self.C.mv(y) 
            return (x * y).numpy()

        def cross_product(x, y):
            x, y = self.R.blades_init(x, [1]), self.R.blades_init(y, [1])
            print(x * y)
            return (x * y).blade(1)

        def inner_prod(x, y):
            if x.shape[0] == 4: G = self.H
            elif x.shape[0] == 2: G = self.C
            x, y = G.mv(x), G.mv(y)
            return (x @ y).numpy()
        
        def wedge_prod(x, y):
            if x.shape[0] == 4: G = self.H
            elif x.shape[0] == 2: G = self.C
            x, y = G.mv(x), G.mv(y)
            return (x ^ y).numpy()

        def geo_prod(x, y):
            if x.shape[0] == 4: return quat_prod(x,y)
            elif x.shape[0] == 2: return complex_prod(x,y)

        geometric_test(quat_prod, complex_prod, cross_product)
        inner_wedge_test(inner_prod, wedge_prod, inner_prod, geo_prod)

if __name__ == '__main__':
    unittest.main()
