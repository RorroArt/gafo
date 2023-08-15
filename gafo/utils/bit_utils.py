# Bit utils
# This are used for generating the cayley tensors

def count_ones(n: int):
    return bin(n).count('1')

def count_swaps(a: int, b: int):
    a = a >> 1
    _sum = 0
    while a != 0:
        _sum += count_ones(a & b)
        a = a >> 1
    return _sum

def ones_positions(n: int):
    pos = []
    index = 0
    while n:
        if n & 1:
            pos.append(index)
        n >>= 1
        index += 1
    return pos

def bit_len(n: int):
    return len(bin(n))-2
