import torch
import numpy as np
import pypose as pp
from torch import nn
import numpy as np
from pypose.optim import ADMMOptim
from pypose.optim.scheduler import CnstOptSchduler

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
        result = torch.square(self.x) + torch.square(self.z)
        return result

    def cnst(self, inputs):
        # Define your specific constraints, Ax + Bz - c
        violation = self.x +self.z- 2
        return violation

    def forward(self, inputs):
        return self.obj(inputs), self.cnst(inputs)

# class ComplexADMMModel(nn.Module):
#     def __init__(self, dim_x, dim_z) -> None:
#         super().__init__()
#         self.dim_x = dim_x
#         self.dim_z = dim_z

#         init_x = torch.randn(dim_x).to(device)
#         init_z = torch.randn(dim_z).to(device)

#         self.x = torch.nn.Parameter(init_x.clone())
#         self.z = torch.nn.Parameter(init_z.clone())

#         self.u = torch.zeros(dim_x).to(device)

#         # Creating some random matrices A and B, and vector c for the constraint
#         self.A = torch.randn(dim_x, dim_x).to(device)
#         self.B = torch.randn(dim_x, dim_z).to(device)
#         self.c = torch.randn(dim_x).to(device)

#     def obj(self):
#         f_x = torch.norm(self.x, p=1)
#         g_z = 0.5 * torch.norm(self.z, p=2) ** 2
#         return f_x + g_z

#     def cnst(self):
#         violation = torch.matmul(self.A, self.x) + torch.matmul(self.B, self.z) - self.c
#         return violation

#     def forward(self):
#         return self.obj(), self.cnst()

if __name__ == "__main__":
    input = None
    admm_model = ADMMModel(1).to(device)
    #admm_model = ComplexADMMModel(5, 3).to(device)
    inner_optimizer_x = torch.optim.SGD([admm_model.x], lr=1e-2, momentum=0.9)
    inner_schd_x = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_x, step_size=10, gamma=0.5)
    inner_optimizer_z = torch.optim.SGD([admm_model.z], lr=1e-2, momentum=0.9)
    inner_schd_z = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_z, step_size=10, gamma=0.5)
    optimizer = ADMMOptim(admm_model, inner_optimizer_x, inner_optimizer_z,inner_schd_x,inner_schd_z)

    # apply scheduler for ADMM
    scheduler = CnstOptSchduler(optimizer, steps=100, inner_iter=300, \
                                    object_decrease_tolerance=1e-6, violation_tolerance=1e-6, \
                                    verbose=True)
    while scheduler.continual():
            loss = optimizer.step(input)
            scheduler.step(loss)
    # for idx in range(300):
    #     loss, u, primal_residual, dual_residual = optimizer.step(inputs)
    #     if optimizer.terminate:
    #             break
    print('-----------Optimized Result----------------')
    # print("u*:", u)
    print("x*:", admm_model.x)
    print("z*:", admm_model.z)
    # print("Primal Residual:", primal_residual)
    # print("Dual Residual:", dual_residual)
