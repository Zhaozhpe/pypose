import torch
import numpy as np
import pypose as pp
from torch import nn
import numpy as np
from pypose.optim import ADMMOptim
from pypose.optim.scheduler import CnstOptSchduler
# import random
# random.seed(1)
torch.set_printoptions(precision=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ADMMModel(nn.Module):
#     def __init__(self, *dim) -> None:
#         super().__init__()
#         # init = torch.randn(*dim)
#         # self.x = torch.nn.Parameter(init.clone())
#         # self.z = torch.nn.Parameter(init.clone())
#         # self.u = torch.zeros_like(init).to(device)
#         self.x = nn.Parameter(torch.tensor([5.0]))  # Initialize K_p (proportional gain)
#         self.z = nn.Parameter(torch.tensor([5.0]))  # Initialize K_d (derivative gain)
#         self.u = torch.tensor([0.0]).to(device)                         #initialization for example 3'''

#     def obj(self, inputs):
#         # Define your specific objective functions f(x) and g(z)
#         #result = torch.square(self.x) + torch.square(self.z)                   #example 1 ans x and z 1 and 1
#         #result = (self.x - 3)**2 + (self.z + 1)**2                             #example 2 ans x and z 3 and -1
#         #result= torch.exp(self.x)+ torch.log(self.z + 2)
#         s = 1.0  # A dummy Laplace variable for the sake of this example
#         y = self.x / (s**2 + self.z * s + self.x)
#         f_x = torch.abs(0.98 - y)
#         g_z = y**2
#         result =f_x+g_z                                                      #example 3 ans x and z 5.0009 and 4.9991
#         #result =self.x**3 + self.z**3 - self.z**2
#         return result

#     def cnst(self, inputs):
#         # Define your specific constraints, Ax + Bz - c
#         #violation = self.x +self.z- 2                                          #example 1
#         #violation = self.x +self.z- 2                                          #example 2
#         violation=self.x + self.z - 10                                         #example 3
#         #violation=2*self.x + self.z - 2
#         return violation

#     def forward(self, inputs):
        # self.fx=torch.square(self.x)
        # self.gx=torch.square(self.z)
#         return self.fx,self.gx,self.obj(inputs), self.cnst(inputs)


# class ADMMModelLieAlgebra(nn.Module):
#     def __init__(self, device) -> None:
#         super().__init__()
#         self.device = device
#         self.x = nn.Parameter(torch.tensor([0.5], dtype=torch.float, requires_grad=True).to(device))
#         self.z = nn.Parameter(torch.tensor([0.5], dtype=torch.float, requires_grad=True).to(device))
#         self.u = nn.Parameter(torch.zeros(1).to(device))

#     def obj(self, inputs):
#         # Lie Algebra-related objective function
#         J = torch.tensor([[0, -1], [1, 0]], dtype=torch.float).to(self.device)
#         Z = torch.tensor([[0, -self.z], [self.z, 0]], dtype=torch.float).to(self.device)
#         exp_Z = torch.matrix_exp(Z)
#         I = torch.eye(2).to(self.device)
#         return self.x**2 + torch.norm(exp_Z - I, p='fro')**2

#     def cnst(self, inputs):
#         # Constraint function
#         return 2*self.x + self.z - 2

#     def forward(self, inputs):
#         return self.obj(inputs), self.cnst(inputs)

# class PoseInvCnstAdmm(nn.Module):
#             def __init__(self, *dim) -> None:
#                 super().__init__()
#                 self.pose = pp.Parameter(pp.randn_so3(*dim))
#                 self.z = pp.Parameter(pp.randn_so3(*dim))
#                 # self.u = torch.tensor([0.0,0.0,0.0]).to(device).unsqueeze(0)
#                 self.u = torch.tensor([0.0]).to(device)



#             def obj(self, inputs):
#                 identity = torch.eye(3).to(inputs.device)
#                 f_p = torch.norm((self.pose.Exp() @ inputs).matrix() - identity)
#                 fixed_quaternion = pp.euler2SO3(np.array([[0.0, 0.0, 0.0]])).to(torch.float).to(inputs.device)
#                 g_z = torch.norm(torch.sub(self.z.Exp(), fixed_quaternion), p=2, dim=1) - 0.35  # so3 not closed for sub
#                 result = f_p + g_z

#                 return torch.norm(result)
#                 # return result

#             def cnst(self, inputs):
#                 constraint_violation = self.pose - self.z
#                 return torch.norm(constraint_violation)

#             def forward(self, inputs):
#                 return self.obj(inputs), self.cnst(inputs)

class PoseInvCnstAdmm(nn.Module):
    def __init__(self, *dim) -> None:
        super().__init__()
        self.x = pp.Parameter(pp.randn_so3(*dim))
        self.z = pp.Parameter(pp.randn_so3(*dim))
        # self.u = torch.tensor([0.0,0.0,0.0]).to(device).unsqueeze(0)
        # self.u = torch.tensor([0.0]).to(device)
        self.current_pose = pp.SO3(pp.euler2SO3(np.array([[0.0, 0.0, np.pi/2]])).to(torch.float)).to(device)
        # print("current_pose", self.current_pose)

    def obj(self, inputs):

        g_x = torch.norm((self.x.Exp() @ inputs).Log())
        h_z = torch.norm((self.z.Exp() @ self.current_pose).Log())
        result = g_x + h_z

        return result

    def obj_all(self, inputs):

        g_x = torch.norm((self.x.Exp() @ inputs).Log())
        h_z = torch.norm((self.z.Exp() @ self.current_pose).Log())

        return g_x, h_z

    def cnst(self, inputs):
        constraint_violation = self.x - self.z
        return constraint_violation.squeeze(0)
        # return torch.norm(constraint_violation).unsqueeze(0)

    def forward(self, inputs):
        return self.obj(inputs), self.cnst(inputs)
            
# class test(nn.Module):
#             def __init__(self, *dim) -> None:
#                 super().__init__()
#                 self.x = pp.SO3(pp.euler2SO3(np.array([[0.0, 0.0, -7 * np.pi/16]])).to(torch.float)).to(device)
#                 self.z = pp.SO3(pp.euler2SO3(np.array([[0.0, 0.0, -7 * np.pi/16]])).to(torch.float)).to(device)
#                 # self.u = torch.tensor([0.0,0.0,0.0]).to(device).unsqueeze(0)
#                 # self.u = torch.tensor([0.0]).to(device)
#                 self.current_pose = pp.SO3(pp.euler2SO3(np.array([[0.0, 0.0, np.pi/2]])).to(torch.float)).to(device)
#                 # print("current_pose", self.current_pose)



#             def obj(self, inputs):

#                 g_x = torch.norm((self.x @ inputs).Log())
#                 h_z = torch.norm((self.z @ self.current_pose).Log())
#                 result = g_x + h_z

#                 return result
#             def obj_all(self, inputs):

#                 g_x = torch.norm((self.x.Exp() @ inputs).Log())
#                 h_z = torch.norm((self.z.Exp() @ self.current_pose).Log())
#                 # result = g_x + h_z

#                 return g_x, h_z

#             def cnst(self, inputs):
#                 constraint_violation = self.x - self.z
#                 return constraint_violation.squeeze(0)
#                 # return torch.norm(constraint_violation).unsqueeze(0)

#             def forward(self, inputs):
#                 return self.obj(inputs), self.cnst(inputs)



if __name__ == "__main__":
    # test
    
    # t1 = test(1).to(device)
    # euler_angles = np.array([[0.0, 0.0, np.pi/4]])
    # quaternion = pp.euler2SO3(euler_angles).to(torch.float)
    # input = pp.SO3(quaternion).to(device)
    # obj, con = t1(inputs=input)
    # print("obj", obj)
    # print("con", con)


    euler_angles = np.array([[0.0, 0.0, np.pi/4]])
    quaternion = pp.euler2SO3(euler_angles).to(torch.float)
    input = pp.SO3(quaternion).to(device)
    admm_model = PoseInvCnstAdmm(1).to(device)
    inner_optimizer_x = torch.optim.Adam([admm_model.x], lr=1e-2)
    inner_schd_x = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_x, step_size=10, gamma=0.5)
    inner_optimizer_z = torch.optim.Adam([admm_model.z], lr=1e-2)
    inner_schd_z = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_z, step_size=10, gamma=0.5)
    optimizer = ADMMOptim(admm_model, inner_optimizer_x, inner_optimizer_z)

    scheduler = CnstOptSchduler(optimizer, steps=100, inner_scheduler=[inner_schd_x, inner_schd_z], \
                                inner_iter=30, object_decrease_tolerance=1e-5, violation_tolerance=1e-5, \
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
    print("x*:", admm_model.x.Exp())
    print("z*:", admm_model.z.Exp())
    print("x*:", admm_model.x.euler())
    print("z*:", admm_model.z.euler())
    
    # # print("Primal Residual:", primal_residual)
    # # print("Dual Residual:", dual_residual)
