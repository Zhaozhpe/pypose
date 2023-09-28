import torch
import numpy as np
from torch import nn
import pypose as pp
from pypose.optim import ADMMOptim  # Assuming you have this optimizer defined similarly to SAL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoseADMMExample:

    class PoseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pose = pp.Parameter(pp.randn_so3())

        def objective(self, inputs):
            result = (self.pose.Exp() @ inputs).matrix() - torch.eye(3).to(inputs.device)
            return torch.norm(result)

        def constrain(self, inputs):
            fixed_euler_angles = np.array([[0.0, 0.0, 0.0]])
            fixed_quaternion = pp.euler2SO3(fixed_euler_angles).to(torch.float).to(inputs.device)
            quaternion = self.pose.Exp()
            difference_quaternions = torch.sub(quaternion, fixed_quaternion)
            distance = torch.norm(difference_quaternions, p=2, dim=1)
            d_fixed = 0.35
            constraint_violation = distance - d_fixed
            return constraint_violation

        def forward(self, inputs):
            return self.objective(inputs), self.constrain(inputs)

    def run_admm_example(self):
        euler_angles = np.array([[0.0, 0.0, np.pi/4]])
        quaternion = pp.euler2SO3(euler_angles).to(torch.float)
        input = pp.SO3(quaternion).to(device)

        model = self.PoseModel().to(device)
        inner_optimizer_x = torch.optim.SGD([{'params': model.pose}], lr=1e-2, momentum=0.9)
        inner_optimizer_z = torch.optim.SGD([{'params': model.pose}], lr=1e-2, momentum=0.9)
        optimizer = ADMMOptim(model=model, inner_optimizer_x=inner_optimizer_x, inner_optimizer_z=inner_optimizer_z)

        for idx in range(20):
            loss, u, primal_residual, dual_residual = optimizer.step(input)
            print('-----------optimized result----------------')
            print("Lambda:",u)
            print('x axis:', np.around(model.pose.cpu().detach().numpy(), decimals=4))
            print('f(x):', model.objective(input))
            print('final violation:', model.constrain(input))
            print('primal_residual:', primal_residual)
            print('dual_residual:', dual_residual)

            if primal_residual < optimizer.tolerance and dual_residual < optimizer.tolerance:
                print("Converged!")
                break

if __name__ == "__main__":
    admm_example = PoseADMMExample()
    admm_example.run_admm_example()
