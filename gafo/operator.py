from typing import TYPE_CHECKING

from gafo.backends import JaxInterpreter, NumpyInterpreter, TorchInterpreter

from gafo.ops import Ops

if TYPE_CHECKING:
    from gafo.multivector import MultiVector

BACKENDS = {'Jax': JaxInterpreter, 'Numpy': NumpyInterpreter, 'Torch': TorchInterpreter}

class Operator:
        __slots__ = 'mv', 'device', 'device_data', 'backend'

        def __init__(self, mv: 'MultiVector', device: str):
                self.device = device
                self.mv = mv
                self.device_data = None
                self.backend = BACKENDS[device]()

        
        def lazy_device_data(self):
                if self.device_data is None:
                        self.device_data = self.backend.to_device(self.mv.values)
                return self.device_data
        
        def apply_op(self, op, *inputs):
                out = self.backend.lookup_ops[op](*inputs)
                return self.mv.algebra.mv(self.backend.from_device(out))
                