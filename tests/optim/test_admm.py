import torch
import numpy as np
import pypose as pp
from torch import nn
import numpy as np
from pypose.optim import ADMMOptim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ADMMModel(nn.Module):
    def __init__(self, *dim) -> None:
        super().__init__()
        init = torch.randn(*dim)
        self.x = torch.nn.Parameter(init.clone())
        self.z = torch.nn.Parameter(init.clone())
        self.u = torch.zeros_like(init).to(device)

    def obj(self, inputs):
        # Define your specific objective functions f(x) and g(z)
        result = -self.x.prod() - self.z.prod()
        return result
    
    def cnst(self, inputs):
        # Define your specific constraints, Ax + Bz - c
        violation = torch.square(torch.norm(self.x + self.z, p=2)) - 2
        return violation.unsqueeze(0)
    
    def forward(self, inputs):
        return self.obj(inputs), self.cnst(inputs)

if __name__ == "__main__":
    inputs = None
    admm_model = ADMMModel(1).to(device)
    inner_optimizer_x = torch.optim.SGD([admm_model.x], lr=1e-2, momentum=0.9)
    inner_optimizer_z = torch.optim.SGD([admm_model.z], lr=1e-2, momentum=0.9)
    optimizer = ADMMOptim(admm_model, inner_optimizer_x, inner_optimizer_z)
    loss, u, primal_residual, dual_residual = optimizer.step(inputs)
    print('-----------Optimized Result----------------')
    print("u*:", u)
    print("x*:", admm_model.x)
    print("z*:", admm_model.z)
    print("Primal Residual:", primal_residual)
    print("Dual Residual:", dual_residual)