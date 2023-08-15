from gafo.ops import Ops

from typing import TYPE_CHECKING, List, Union
import numpy as np
if TYPE_CHECKING: from gafo.algebra import Cayley; from gafo import MultiVector
# Here is where we implement all the possible operations and their derivatives
# Autograd is a major TODO so at the moment only forward pass is suported

class Function:

    def forward(self): raise 'Not implemented'
    def backward(self): raise 'Not implemented'

    @classmethod
    def apply(_fn: 'Function', *x):
        fn = _fn()
        return fn.forward(*x)


# Unary ops

class Exp(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector') -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.EXP, x.extract())

class Log(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector') -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.LOG, x.extract())

class Relu(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector') -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.RELU, x.extract())


class Sin(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector') -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.SIN, x.extract())

class Sqrt(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector') -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.SQRT, x.extract())

class Sigmoid(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector') -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.SIGMOID, x.extract())

# Binary Ops 

class Add(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector', y: 'MultiVector') -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.ADD, x.extract(), y.extract())

class Mul(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector', y: 'MultiVector') -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.MUL, x.extract(), y.extract())

# Cayley should be sparse array in the future
class Product(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector', y: 'MultiVector', cayley: np.ndarray) -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.PRODUCT, x.extract(), y.extract(), cayley)

class Pow(Function):
    __slots__ = 'x'

    def forward(self, x: 'MultiVector', y: 'MultiVector') -> 'MultiVector':
        self.x = x
        return x.op.apply_op(Ops.POW, x.extract(), y.extract())