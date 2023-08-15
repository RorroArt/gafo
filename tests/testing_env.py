import unittest 
import json 

import numpy as np

from gafo import Algebra


# Generic Test functions
def value_test(value_name, value, expected_value): assert value == expected_value, f'{value_name} does not match. {value} != {expected_value}'
def array_test(array, expected_array): np.testing.assert_allclose(array, expected_array)
def dict_test(dic, expected_dic): assert json.dumps(dic, sort_keys=True) == json.dumps(expected_dic, sort_keys=True), f'Dicts do not match. {dic} != {expected_dic}'
def fn_test(fn1, fn2, inputs): _fn1 = fn1(*inputs); _fn2 = fn2(*inputs); assert _fn1 == _fn2, f'Results from both functions does not match. {_fn1} != {_fn2}'
def np_fn_test(fn1, fn2, inputs): _fn1 = fn1(*inputs); _fn2= fn2(*inputs); np.testing.assert_allclose(_fn1, _fn2) 


# Test Products 

# Tests the geometric product by evaluating expected product of common algebras
def geometric_test(quat_prod, complex_prod,  cross_product):
    quat1, quat2 = np.random.randn(4), np.random.randn(4)
    complex1, complex2 = np.random.randn(2), np.random.randn(2) 
    r1, r2 = np.array([5.6, 41, .03]), np.array([5.632, 6.1, 34])

    expected_quat = np.array([
        quat1[0]*quat2[0] - quat1[1]*quat2[1] - quat1[2]*quat2[2] - quat1[3]*quat2[3],
        quat1[0]*quat2[1] + quat1[1]*quat2[0] + quat1[2]*quat2[3] - quat1[3]*quat2[2],
        quat1[0]*quat2[2] - quat1[1]*quat2[3] + quat1[2]*quat2[0] + quat1[3]*quat2[1],
        quat1[0]*quat2[3] + quat1[1]*quat2[2] - quat1[2]*quat2[1] + quat1[3]*quat2[0], 
    ])

    expected_complex = np.array([
        complex1[0]*complex2[0] - complex1[1]*complex2[1],
        complex1[0]*complex2[1] + complex1[1]*complex2[0] 
    ])
    
    expected_cross = np.array([
        r1[0]*r2[1] - r1[1]*r2[0],
        r1[1]*r2[2] - r1[2]*r2[1],
        r1[2]*r2[0] - r1[0]*r2[2]  
    ]) 

    np_fn_test(quat_prod, lambda x, y: expected_quat, (quat1,quat2))
    np_fn_test(complex_prod, lambda x, y: expected_complex, (complex1,complex2))
    
    # test cross product 
    #np_fn_test(cross_product, lambda x, y: expected_cross, (r1,r2))

   

# Tests inner and wedge products by checking their properties
def inner_wedge_test(inner_prod, wedge_prod, dot_prod, geo_prod):
    quat1, quat2 = np.random.randn(4), np.random.randn(4)
    complex1, complex2 = np.random.randn(2), np.random.randn(2) 
    r1, r2 = np.array([5.6, 41, .03]), np.array([5.632, 6.1, 34])
    # Compute geometric
    quat_geo = geo_prod(quat1, quat2)
    anti_quat_geo = geo_prod(quat2, quat1)
    complex_geo = geo_prod(complex1, complex2)
    anti_complex_geo = geo_prod(complex2, complex1)

    # Compute inner
    quat_inner = inner_prod(quat1, quat2)
    complex_inner = inner_prod(complex1, complex2)

    #array_test(quat_inner, .5*(quat_geo+anti_quat_geo))
    #array_text(complex_inner, .5*(complex_geo + anti_complex_geo))
    # Wedge test

    # Compute wedge
    quat_wedge = wedge_prod(quat1, quat2)
    complex_wedge = wedge_prod(complex1, complex2)

    #array_test(quat_wedge, .5*(quat_geo-anti_quat_geo))
    #array_text(complex_wedge, .5*(complex_geo - anti_complex_geo))
    # Antisymmetry test
    anti_quat_wedge =  -wedge_prod(quat2, quat1)
    anti_complex_wedge = -wedge_prod(complex2, complex1)
    #array_test(quat_wedge, anti_quat_wedge)
    #array_test(complex_wedge, anti_complex_wedge)

   
    # Sum test
    #array_test(quat_inner + quat_wedge, quat_geo)
    #array_test(complex_inner + complex_wedge, complex_geo)


    
  

# Common set up for tests
# Inits tree common algebras (Complex, quaternions, and cross product) to test expected results
class TestGafo(unittest.TestCase):
    def setUp(self):
        self.C = Algebra(0,1) # Complex
        self.H = Algebra(0,2) # Quaternions
        self.R = Algebra(3,0) # R^3 with cross product


