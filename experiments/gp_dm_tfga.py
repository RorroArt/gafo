import numpy as np
import time 

# Generate cayley table
'''
def sig_to_cayley(signature, grades):
    N = len(signature)
    
    cayley = np.zeros(N,N,N)

    for i, a in enumerate(grades):
        for j, b in enumerate(grades):
            if i == j:
                result_idx = 0
                idx = i
            elif
            cayley[i, j, result_idx] = sign[idx]

    return ''
'''

def reduce(basis_a, basis_b)

# TFGA product

def geo_product(a, b, cayley):
    x = np.tensordot(a, cayley, axes=[-1, 0])
    x = np.expand_dims(b, axis=len(b.shape) -1 ) @ x

    return x.squeeze(0)

# DeepMind quaternion product

def quat_product(a,b, cayley):
    return np.sum(
      cayley *
      a[..., :, None, None] *
      b[..., None, :, None],
      axis=(-3, -2))



def test_quaternion(result, a, b):
    expected_result = np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] +- a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0], 
    ])

    assert np.array_equal(result, expected_result), f'The answer is incorrect. Result: {result}, Expected result{expected_result}'

    print(' \n Quaternion product test passed: The answer is correct \n')


def test_complex(result, a, b):
    expected_result = np.array([
        a[0]*b[0] - a[1]*b[1],
        a[0]*b[1] + a[1]*b[0] 
    ])

    assert np.array_equal(result, expected_result), f'The answer is incorrect. Result: {result}, Expected result{expected_result}'

    print(' \n Complex product test passed: The answer is correct \n')

if __name__ == '__main__':

    # Test complex product

    a = np.array([4, 5])
    b = np.array([9,7])
    CAYLEY_COMPLEX = np.zeros((2, 2, 2), dtype=np.float32)

    CAYLEY_COMPLEX[:, :, 0] = [[1, 0],
                            [0, -1]]

    CAYLEY_COMPLEX[:, :, 1] = [[0, 1],
                            [1, 0]]

    print(' \n ------ Test for tfga complex product ----- \n')

    exec_time = time.monotonic()
    c = geo_product(a, b, CAYLEY_COMPLEX)
    exec_time = time.monotonic() - exec_time
    print(f'Time: {exec_time} ')
    test_complex(c, a, b)

    print(' \n ------ Test for dm complex product ----- \n')
    exec_time = time.monotonic() 
    C = quat_product(a, b, CAYLEY_COMPLEX)
    exec_time = time.monotonic() - exec_time
    print(f'Time: {exec_time} ') 
    test_complex(c, a, b)
    # Test quaternion product

    A  = np.array([3,4, 5, 10])
    B  = np.array([1,5, 7, 2])

    CAYLEY = np.zeros((4, 4, 4), dtype=np.float32)
    CAYLEY[:, :, 0] = [[ 1, 0, 0, 0],
                            [ 0,-1, 0, 0],
                            [ 0, 0,-1, 0],
                            [ 0, 0, 0,-1]]

    CAYLEY[:, :, 1] = [     [ 0, 1, 0, 0],
                            [ 1, 0, 0, 0],
                            [ 0, 0, 0, 1],
                            [ 0, 0,-1, 0]]

    CAYLEY[:, :, 2] = [ [ 0, 0, 1, 0],
                        [ 0, 0, 0,-1],
                        [ 1, 0, 0, 0],
                        [ 0, 1, 0, 0]]

    CAYLEY[:, :, 3] = [[ 0, 0, 0, 1],
                            [ 0, 0, 1, 0],
                            [ 0,-1, 0, 0],
                            [ 1, 0, 0, 0]]

                            

    # Test quaternion product
    print(' \n ------ Test for tfga quaternion product ----- \n')

    exec_time = time.monotonic()
    C = geo_product(A, B, CAYLEY)
    exec_time = time.monotonic() - exec_time
    print(f'Time: {exec_time} ')
    test_quaternion(C, A, B)

    print(' \n ------ Test for dm quaternion product ----- \n')
    exec_time = time.monotonic() 
    C = quat_product(A, B, CAYLEY)
    exec_time = time.monotonic() - exec_time
    print(f'Time: {exec_time} ') 
    test_quaternion(C, A, B)

    print(1/2 * (quat_product(A,A, CAYLEY)+quat_product(A,A, CAYLEY)))
