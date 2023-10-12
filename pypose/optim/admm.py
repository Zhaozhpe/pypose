import torch
from torch import nn
from .scndopt import _Optimizer

class ADMMOptim(_Optimizer):
    def __init__(self, model, inner_optimizer_x, inner_optimizer_z, rho=1, tolerance=3e-4,min=1e-6, max=1e32):
        defaults = {**{'min':min, 'max':max}}
        super().__init__(model.parameters(), defaults=defaults)
        self.terminate = False
        self.model = model
        self.inner_optimizer_x = inner_optimizer_x
        self.inner_optimizer_z = inner_optimizer_z
        self.rho = rho
        self.inner_iter = 0
        self.tolerance = tolerance

    def step(self, inputs):

        z_old = self.model.z.clone()
        ## add last value
        with torch.no_grad():
            obj, cnst = self.model(inputs)
            self.last_object_value = obj.sum()
            self.last_violation = torch.norm(cnst)

        for i in range(self.inner_iter):
            self.inner_optimizer_x.zero_grad()
            loss_x = self.model.obj(inputs) + self.rho/2 * (self.model.cnst(inputs) + self.model.u) ** 2
            scalar_loss_x = loss_x.sum()
            scalar_loss_x.backward()
            self.inner_optimizer_x.step()

        # self.inner_schd_x.step()

        # z-update step
        for j in range(self.inner_iter):
            self.inner_optimizer_z.zero_grad()
            loss_z = self.model.obj(inputs) + self.rho/2 * (self.model.cnst(inputs) + self.model.u) ** 2
            scalar_loss_z = loss_z.sum()
            scalar_loss_z.backward()
            self.inner_optimizer_z.step()

        # self.inner_schd_z.step()
        self.loss = loss_x + loss_z
        # fix sgd optimization issue and improve accuracy
        with torch.no_grad():
            # u-update step
            self.model.u += self.rho * self.model.cnst(inputs)

            object_value, violation = self.model(inputs)
            self.object_value = object_value.sum()
            self.violation_norm = torch.norm(violation)

            # primal_residual = torch.norm(violation)
            dual_residual = self.rho * torch.norm(self.model.z - z_old)

            # primal_residual = torch.norm(self.model.x + self.model.z - 2)
            # dual_residual = self.rho * torch.norm(self.model.z - z_old)

            # Check convergence

            # if self.violation_norm < self.tolerance and dual_residual < self.tolerance:
            #     print("Converged!")
            #     self.terminate = True
            #     return self.model.obj(inputs), self.model.u,self.violation_norm, dual_residual

            z_old = self.model.z.clone()

            # Check the results
        
            
            # print('--------------------NEW-ADMM-EPOCH-------------------')
            # print('u:', self.model.u)
            # print('object_loss:', self.object_value)
            # print('absolute violation:', violation)
            # print("x",self.model.x)
            # print("z",self.model.z)

            ## why best_vlolation
            self.best_violation = torch.norm(violation)

        return self.model.obj(inputs), self.model.u,self.violation_norm, dual_residual
