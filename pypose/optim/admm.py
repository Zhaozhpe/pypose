import torch
from torch import nn

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

class ADMMOptim:
    def __init__(self, model, inner_optimizer_x, inner_optimizer_z, rho=1.0, max_iter=20, tolerance=1e-6):
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
