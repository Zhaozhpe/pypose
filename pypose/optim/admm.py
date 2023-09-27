import torch
from torch import nn
from .scndopt import _Optimizer

class ADMMOptim(_Optimizer):
    def __init__(self, model, inner_optimizer_x, inner_optimizer_z, rho=1.0, max_iter=20, tolerance=1e-6,min=1e-6, max=1e32):
        defaults = {**{'min':min, 'max':max}}
        super().__init__(model.parameters(), defaults=defaults)
        self.model = model
        self.inner_optimizer_x = inner_optimizer_x
        self.inner_optimizer_z = inner_optimizer_z
        self.rho = rho
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.best_violation = torch.norm(self.model.cnst(None))
    
    def step(self, inputs):
        z_old = self.model.z.clone()
        for _ in range(self.max_iter):
        #for _ in range(self.max_iter):
            # x-update step
            self.inner_optimizer_x.zero_grad()
            loss_x = self.model.obj(inputs) + self.rho/2 * (self.model.cnst(inputs) + self.model.u) ** 2
            loss_x.backward()
            self.inner_optimizer_x.step()
            
            # z-update step
            self.inner_optimizer_z.zero_grad()
            loss_z = self.model.obj(inputs) + self.rho/2 * (self.model.cnst(inputs) + self.model.u) ** 2
            loss_z.backward()
            self.inner_optimizer_z.step()
            
            # u-update step
            self.model.u += self.rho * self.model.cnst(inputs)
        
            # Check the results
            with torch.no_grad():
                object_value, violation = self.model(inputs)
                print('--------------------NEW-ADMM-EPOCH-------------------')
                print('u:', self.model.u)
                print('object_loss:', object_value)
                print('absolute violation:', torch.norm(violation))
                
                # if torch.norm(violation) <= self.best_violation * self.tolerance:
                #     print("Converged!")
                #     break
                # self.best_violation = torch.norm(violation)
                primal_residual = torch.norm(self.model.x - self.model.z)
                dual_residual = self.rho * torch.norm(self.model.z - z_old)
                
                # Check convergence
                if primal_residual < self.tolerance and dual_residual < self.tolerance:
                    print("Converged!")
                    break
                
                z_old = self.model.z.clone()

        return self.model.obj(inputs), self.model.u
