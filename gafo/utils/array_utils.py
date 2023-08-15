from typing import NamedTuple, List

def get_elements(A: List, indeces: List[int]):
    return [A[i] for i in indeces]

def ones_array(length: int):
    return [1 for _ in range(length)]

# This is likely storing usless zeros in memomry
def zeros_array(length: int):
    return [0 for _ in range(length)]