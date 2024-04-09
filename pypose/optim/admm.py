import torch
from torch import nn
from .scndopt import _Optimizer

class _Unconstrained_Model(nn.Module):
    def __init__(self, model, penalty_factor):
        super().__init__()
        self.model = model
        self.pf = penalty_factor

    # def update_u(self, error):
    #     self.model.u += self.rho * self.model.cnst(inputs)
    #     self.lmd += error * self.pf
    #     return self.lmd

    # def update_penalty_factor(self, pnf_update_step, safe_guard):
    #     pf = self.pf * pnf_update_step
    #     self.pf = pf if pf < safe_guard else safe_guard

    def forward(self, inputs, target=None):


        R, C = self.model(inputs)
        self.u = self.u if hasattr(self, 'u') \
                else torch.zeros_like(C) #torch.zeros(1)
        self.u = self.u.to(R.device)

        penalty_term = torch.square(torch.norm(C))
        L = R + torch.dot(self.u, C) + self.pf * penalty_term / 2
        return L

class ADMMOptim(_Optimizer):
    def __init__(self, model, inner_optimizer_x, inner_optimizer_z, rho=0.04, tolerance=1e-5, min=1e-6, max=1e32):
        defaults = {**{'min':min, 'max':max}}
        super().__init__(model.parameters(), defaults=defaults)
        self.terminate = False
        self.model = model
        self.inner_optimizer_x = inner_optimizer_x
        self.inner_optimizer_z = inner_optimizer_z
        self.rho = rho
        # self.inner_iter = 400
        self.tolerance = tolerance
        self.admm_model = _Unconstrained_Model(self.model, penalty_factor=rho)

    def step(self, inputs):

        z_old = self.model.z.clone()
        ## add last value
        with torch.no_grad():
            obj, cnst = self.model(inputs)
            self.last_object_value = obj
            self.last_violation = torch.norm(cnst)

        for i in range(self.inner_iter):
            self.inner_optimizer_x.zero_grad()
            loss_x = self.admm_model(inputs)
            # loss_x = self.model.obj(inputs) + self.rho/2 * (self.model.cnst(inputs) + self.model.u) ** 2
            # scalar_loss_x = loss_x.sum()
            loss_x.backward()
            self.inner_optimizer_x.step()
            # if i % 20 == 0:
            #     print("scalar_loss_x",loss_x)

        #self.inner_schd_x.step()

        # z-update step
        for j in range(self.inner_iter):
            self.inner_optimizer_z.zero_grad()
            loss_z = self.admm_model(inputs)
            # loss_z = self.model.obj(inputs) + self.rho/2 * (self.model.cnst(inputs) + self.model.u) ** 2
            # scalar_loss_z = loss_z.sum()
            loss_z.backward()
            self.inner_optimizer_z.step()
            # if j % 20 == 0:
            #     print("scalar_loss_z",loss_z)

        #self.inner_schd_z.step()
        # fix sgd optimization issue and improve accuracy
        with torch.no_grad():
            self.loss = loss_x + loss_z
            # u-update step
            self.admm_model.u += self.rho * self.model.cnst(inputs)

            object_value, violation = self.model(inputs)
            self.object_value = object_value
            self.violation_norm = torch.norm(violation)
            g_x, h_z = self.model.obj_all(inputs)
            

            # primal_residual = torch.norm(violation)
            dual_residual = self.rho * torch.norm(self.model.z - z_old)
            print("dual_residual",dual_residual)

            # primal_residual = torch.norm(self.model.x + self.model.z - 2)
            # dual_residual = self.rho * torch.norm(self.model.z - z_old)

            # Check convergence

            if self.violation_norm < self.tolerance and dual_residual < self.tolerance:
                print("Converged!")
                self.terminate = True
                return self.model.obj(inputs), self.admm_model.u,self.violation_norm, dual_residual

            z_old = self.model.z.clone()

            # Check the results


            # print('--------------------NEW-ADMM-EPOCH-------------------')
            print('u:', self.admm_model.u)
            # print('object_loss:', self.object_value)
            # print('absolute violation:', violation)
            print("x",self.model.x)
            print("z",self.model.z)
            # print("x.Exp()",self.model.x.Exp())
            # print("z.Exp()",self.model.z.Exp())
            print("x.euler(:", self.model.x.euler())
            print("z.euler(:", self.model.z.euler())
            print("f_x",g_x)
            print("g_z",h_z)

            ## why best_vlolation
            self.best_violation = torch.norm(violation)

        return self.model.obj(inputs), self.admm_model.u,self.violation_norm, dual_residual
