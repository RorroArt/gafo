import unittest
import numpy as np
from testing_env import GafoTest, value_test, array_test 

from gafo import Algebra

def value_test(value_name, value, expected_value): assert value == expected_value, f'{value_name} does not match. {value} != {expected_value}'

class TestAlgebra(GafoTest):

    def test_init(self):
        G1, G2, G3 = Algebra(3,0), Algebra(0,2), Algebra(2,2)

        # Test dimensionality is correct
        value_test('Dimension', G1.dim, 8)
        value_test('Dimension', G2.dim, 4)
        value_test('Dimension', G3.dim, 16)

        # Test grades
        array_test(np.array(G1.grades), np.array([0,1,1,1,2,2,2,3]).astype(float))
        array_test(G2.grades, [0,1,1,2])
        array_test(G3.grades, [0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4])
        
        #Test signature
        value_test('Signature', G1.signature, '+++')
        value_test('Signature', G2.signature, '--')
        value_test('Siganture', G3.signature, '++--')

    def test_cayley(self):
        # Expected cayley for quaternions
        quat_cayley = np.zeros((4, 4, 4), dtype=np.float32) 
        quat_cayley[:, :, 0] = [[ 1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0,-1, 0],
                                [ 0, 0, 0,-1]]

        quat_cayley[:, :, 1] = [[ 0, 1, 0, 0],
                                [ 1, 0, 0, 0],
                                [ 0, 0, 0, 1],
                                [ 0, 0,-1, 0]]

        quat_cayley[:, :, 2] = [[ 0, 0, 1, 0],
                                [ 0, 0, 0,-1],
                                [ 1, 0, 0, 0],
                                [ 0, 1, 0, 0]]

        quat_cayley[:, :, 3] = [[ 0, 0, 0, 1],
                                [ 0, 0, 1, 0],
                                [ 0,-1, 0, 0],
                                [ 1, 0, 0, 0]]

        # Expected cayley for complex numbers
        complex_cayley = np.zeros((2, 2, 2), dtype=np.float32)
        complex_cayley[:, :, 0] = [ [1, 0],
                                    [0, -1] ]

        complex_cayley[:, :, 1] = [ [0, 1],
                                    [1, 0] ]

        # test geometric cayley
        array_test(self.H.cayley.geometric, quat_cayley)
        array_test(self.C.cayley.geometric, complex_cayley)

    def test_mv_init(self):

        # ones
        array_test(self.C.ones().numpy(), np.ones(self.C.dim))
        array_test(self.H.ones().numpy(), np.ones(self.H.dim))
        array_test(self.R.ones().numpy(), np.ones(self.R.dim))

        # zeros
        array_test(self.C.zeros().numpy(), np.zeros(self.C.dim))
        array_test(self.H.zeros().numpy(), np.zeros(self.H.dim))
        array_test(self.R.zeros().numpy(), np.zeros(self.R.dim))

        # By blade
        array1 = self.C.blades_init([5], [0]).numpy(), np.array([5,0]); array_test(*array1)
        array2 = self.H.blades_init([1,2], [1]).numpy(), np.array([0,1,2,0]); array_test(*array2)
        array3 = self.R.blades_init([3,6,7,8], [1,3]).numpy(), np.array([0,3,6,7,0,0,0,8]); array_test(*array3)
    
    def test_device_change(self):

        # Numpy -> Torch
        self.C.to('Torch')
        value_test('Device', self.C.device, 'Torch')
        value_test('Device', self.C.ones().device, 'Torch')
        value_test('Device', self.C.zeros().device, 'Torch')
        value_test('Device', self.C.blades_init([1],[1]).device, 'Torch')
        value_test('Device', self.C.mv([3,4]).device, 'Torch')
        value_test('Device', self.C.mv([3,4]).op.device, 'Torch')

        # Torch -> Jax
        self.C.to('Jax')
        value_test('Device', self.C.device, 'Jax')
        value_test('Device', self.C.ones().device, 'Jax')
        value_test('Device', self.C.zeros().device, 'Jax')
        value_test('Device', self.C.blades_init([1],[1]).device, 'Jax')
        value_test('Device', self.C.mv([3,4]).device, 'Jax')
        value_test('Device', self.C.mv([3,4]).op.device, 'Jax')

        # From init (Jax)
        G = Algebra(2,0, device='Jax')
        value_test('Device', G.device, 'Jax')
        value_test('Device', G.ones().device, 'Jax')
        value_test('Device', G.zeros().device, 'Jax')
        value_test('Device', G.blades_init([4,1],[1]).device, 'Jax')
        value_test('Device', G.mv([3,4,5,6]).device, 'Jax')
        value_test('Device', G.mv([3,4,3,4]).op.device, 'Jax')

        # Jax -> Numpy
        G.to('Numpy')
        value_test('Device', G.device, 'Numpy')
        value_test('Device', G.ones().device, 'Numpy')
        value_test('Device', G.zeros().device, 'Numpy')
        value_test('Device', G.blades_init([1,3],[1]).device, 'Numpy')
        value_test('Device', G.mv([3,4,6,7]).device, 'Numpy')
        value_test('Device', G.mv([3,4,5,7]).op.device, 'Numpy')

if __name__ == '__main__':
    unittest.main()