from typing import TYPE_CHECKING, Dict, Callable, List
from enum import Enum, auto 

if TYPE_CHECKING:
        from gafo.multivector import MultiVector

# List of Operations 
class Ops(Enum):
        # Unary ops
        EXP = auto()
        LOG = auto()
        RELU = auto()
        SIN = auto()
        SQRT = auto()
        SIGMOID = auto()
        
        # Binary ops
        ADD = auto()
        MUL = auto()
        PRODUCT = auto()
        POW = auto()


# Backend base class
class Interpreter:
        __slots__ = 'lookup_ops'
        lookup_ops: Dict[Ops, Callable]
        def to_device(self): raise 'Not implemented'
        def from_device(self): raise 'Not implemented'

# Compiler class will be implemented in the future to optimize products 

# In charge of interpreting the multivector data
# Load ops from correct device 
# Apply ops


                
