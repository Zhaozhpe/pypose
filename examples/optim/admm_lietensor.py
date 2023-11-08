import torch
import numpy as np
import pypose as pp
from torch import nn
import numpy as np
from pypose.optim import ADMMOptim
from pypose.optim.scheduler import CnstOptSchduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ADMMModelLieAlgebra(nn.Module):   # this example is working
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.x = pp.Parameter(pp.randn_so3(1).to(device))  # Initializing with random tensor
        self.z = pp.Parameter(pp.randn_so3(1, requires_grad=True, device=device))  # Lie Tensor
        self.u = pp.Parameter(pp.randn_so3(1).to(device))

    def obj(self, inputs):
        # Lie Algebra-related objective function
        lie_tensor = pp.LieTensor(self.z, ltype=pp.so3_type)
        # Just an illustrative objective, you might want to replace it with a meaningful one
        return torch.norm(lie_tensor.exp() - torch.eye(3).to(self.device), p='fro')**2 + self.x**2

    def cnst(self, inputs):
        # Constraint function (a simple illustrative linear constraint)
        return self.x - torch.norm(self.z, p='fro')

    def forward(self, inputs):
        return self.obj(inputs), self.cnst(inputs)

# class ADMMModelSE3(nn.Module):
#     def __init__(self, *dim) -> None:
#         super().__init__()
#         # self.device = device

#         # Creating a random SE3 tensor and extracting translation and rotation
#         random_se3 = pp.Parameter(pp.randn_SE3(*dim))
#         print(type(random_se3))
#         random_se3_new=pp.randn_like(random_se3)
#         self.x = random_se3_new.translation()# Translation
#         print(type(self.x))
#         self.z =random_se3_new.rotation() # Rotation
#         print(self.z)

#         self.u = torch.zeros(3, requires_grad=False).to(device)  # Dual variable for translation


#     def obj(self, inputs):
#         goal = pp.SE3([0, 0, 0, 0, 0, 0, 1]).to(device)  # Identity as the goal pose
#         transformation_matrix = torch.cat((self.z.matrix(), self.x.unsqueeze(-1)), dim=-1)
#         last_row = torch.tensor([0, 0, 0, 1], dtype=torch.float).to(device).view(1, 1, -1)
#         transformation_matrix = torch.cat((transformation_matrix, last_row), dim=-2)

#         transformed_goal = transformation_matrix @ goal.matrix()
#         return torch.norm(transformed_goal - goal.matrix(), p=' fro')**2

#     def cnst(self, inputs):
#         original_translation = torch.zeros(3).to(device)
#         return torch.norm(self.x - original_translation, p=2) - 1.0

#     def forward(self, inputs):
#         return self.obj(inputs), self.cnst(inputs)






if __name__ == "__main__":
    input = None
    # goal = pp.SE3([0, 0, 0, 0, 0, 0, 1]).to(device)  # Identity as the goal pose
    # original_translation = torch.zeros(3).to(device)  # Assuming original translation is zero
    admm_model = ADMMModelLieAlgebra(device).to(device)
    #print(len(list(admm_model.parameters())))
    # target_rotation = pp.randn_SE3(1).to(device)  # Replace with your target rotation
    # admm_model = ADMMRotation(device).to(device)


    #admm_model = ComplexADMMModel(5, 3).to(device)
    inner_optimizer_x = torch.optim.SGD([admm_model.x], lr=1e-2)
    inner_schd_x = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_x, step_size=20, gamma=0.5)
    inner_optimizer_z = torch.optim.SGD([admm_model.z], lr=1e-2)
    inner_schd_z = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_z, step_size=20, gamma=0.5)
    optimizer = ADMMOptim(admm_model, inner_optimizer_x, inner_optimizer_z)

    scheduler = CnstOptSchduler(optimizer, steps=100, inner_scheduler=[inner_schd_x, inner_schd_z], \
                                inner_iter=300, object_decrease_tolerance=1e-6, violation_tolerance=1e-6, \
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
