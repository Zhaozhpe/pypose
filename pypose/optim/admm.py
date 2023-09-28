import torch
from torch import nn
from .scndopt import _Optimizer

class ADMMOptim(_Optimizer):
    def __init__(self, model, inner_optimizer_x, inner_optimizer_z,inner_schd_x,inner_schd_z, rho=1, tolerance=3e-4,min=1e-6, max=1e32):
        defaults = {**{'min':min, 'max':max}}
        super().__init__(model.parameters(), defaults=defaults)
        self.terminate = False
        self.model = model
        self.inner_optimizer_x = inner_optimizer_x
        self.inner_optimizer_z = inner_optimizer_z
        self.inner_schd_x = inner_schd_x
        self.inner_schd_z = inner_schd_z
        self.rho = rho
        #self.max_iter = max_iter
        self.tolerance = tolerance
        self.best_violation = torch.norm(self.model.cnst())

    def step(self, inputs):

        z_old = self.model.z.clone()
        for i in range(300):
            self.inner_optimizer_x.zero_grad()
            loss_x = self.model.obj() + self.rho/2 * (self.model.cnst() + self.model.u) ** 2
            scalar_loss_x = loss_x.sum()
            scalar_loss_x.backward(retain_graph=True)
            self.inner_optimizer_x.step()

        self.inner_schd_x.step()

        # z-update step
        for j in range(300):
            self.inner_optimizer_z.zero_grad()
            loss_z = self.model.obj() + self.rho/2 * (self.model.cnst() + self.model.u) ** 2
            scalar_loss_z = loss_z.sum()
            scalar_loss_z.backward(retain_graph=True)
            self.inner_optimizer_z.step()


        self.inner_schd_z.step()

        # u-update step
        self.model.u += self.rho * self.model.cnst()

        primal_residual = torch.norm(self.model.cnst())
        dual_residual = self.rho * torch.norm(self.model.z - z_old)

        # primal_residual = torch.norm(self.model.x + self.model.z - 2)
        # dual_residual = self.rho * torch.norm(self.model.z - z_old)

        # Check convergence
        if primal_residual < self.tolerance and dual_residual < self.tolerance:
            print("Converged!")
            self.terminate = True
            return self.model.obj(), self.model.u,primal_residual, dual_residual

        z_old = self.model.z.clone()

        # Check the results
        with torch.no_grad():
            object_value, violation = self.model()
            print('--------------------NEW-ADMM-EPOCH-------------------')
            print('u:', self.model.u)
            print('object_loss:', object_value)
            print('absolute violation:', torch.norm(violation))
            print("x",self.model.x)
            print("z",self.model.z)

            self.best_violation = torch.norm(violation)

        return self.model.obj(), self.model.u,primal_residual, dual_residual
