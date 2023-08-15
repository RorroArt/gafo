import torch
import torch.nn.functional as F 
import numpy as np
from gafo.ops import Ops, Interpreter

# Define the product function
def torch_product( x: torch.Tensor, y: torch.Tensor, cayley: np.array):
        '''
        Generic product function given by a cayley tensor. 
        Taken from Alphafolds's quaternion product
        '''   
        cayley = torch.Tensor(cayley)
        return torch.sum(cayley * x[..., :, None, None] * y[..., None, :, None], axis=(-3, -2))

class TorchInterpreter(Interpreter):
    lookup_ops = {
        Ops.EXP: torch.exp,
        Ops.LOG: torch.log,
        Ops.RELU: F.relu,
        Ops.SIN: torch.sin,
        Ops.SQRT: torch.sqrt,
        Ops.SIGMOID: torch.sigmoid,
        Ops.ADD: torch.add,
        Ops.MUL: torch.multiply,
        Ops.PRODUCT: torch_product,
        Ops.POW: torch.pow
    }

    def to_device(self, data: np.ndarray) -> torch.Tensor:
        TORCH_DEVICE = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.Tensor(data).to(TORCH_DEVICE)

    def from_device(self, tensor: torch.Tensor) -> np.ndarray: return tensor.detach().numpy()