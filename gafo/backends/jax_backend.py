import jax
import jax.numpy as jnp
import numpy as np
from gafo.ops import Ops, Interpreter

# Define the product function
def jax_product(x: np.array, y: np.array, cayley: np.array):
        '''
        Generic product function given by a cayley tensor. 
        Taken from Alphafolds's quaternion product
        '''
        cayley = jnp.array(cayley) 
        return jnp.sum(cayley * x[..., :, None, None] * y[..., None, :, None], axis=(-3, -2))

class JaxInterpreter(Interpreter):
    lookup_ops = {
        Ops.EXP: jnp.exp,
        Ops.LOG: jnp.log,
        Ops.RELU: jax.nn.relu,
        Ops.SIN: jnp.sin,
        Ops.SQRT: jnp.sqrt,
        Ops.SIGMOID: jax.nn.sigmoid,
        Ops.ADD: jnp.add,
        Ops.MUL: jnp.multiply,
        Ops.PRODUCT: jax_product,
        Ops.POW: jnp.float_power
    }

    def to_device(self, data: np.ndarray) -> jnp.ndarray:
        return jnp.array(data)

    def from_device(self, array: jnp.ndarray) -> np.ndarray:
        return np.array(array)

    