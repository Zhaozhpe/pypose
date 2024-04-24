import torch
import torch.nn as nn
import pypose as pp
import numpy as np
from pypose.optim import ADMMOptim
from pypose.optim.scheduler import CnstOptSchduler
from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import pypose as pp
import numpy as np

class RobotArmPoseOptimization(nn.Module):
    def __init__(self, *dim) -> None:
        super().__init__()
        self.x = pp.Parameter(pp.randn_se3(*dim))  # Pose of the robot arm
        self.z = pp.Parameter(pp.randn_se3(*dim))  # Auxiliary variable
        self.target_pose = pp.SE3(torch.tensor([[0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0]]).to(torch.float)).to(device)
        self.obstacle_pose = pp.SE3(torch.tensor([[0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 1.0]]).to(torch.float)).to(device)

    def obj(self, inputs):
        # Objective function: Minimize the distance between the robot arm pose and the target pose
        # print("self.x.Exp() shape:", self.x.Exp().shape)
        # print("inputs shape:", inputs.shape)
        g_x = torch.norm((self.x.Exp() @ inputs).Log() - self.target_pose.Log())
        h_z = torch.norm((self.z.Exp() @ inputs).Log() - self.target_pose.Log())
        result = g_x + h_z
        return result

    def obj_all(self, inputs):
        # Objective function components
        g_x = torch.norm((self.x.Exp() @ inputs).Log() - self.target_pose.Log())
        h_z = torch.norm((self.z.Exp() @ inputs).Log() - self.target_pose.Log())
        return g_x, h_z

    def cnst(self, inputs):
        # Constraint function: Avoid collision with the obstacle
        constraint_violation = torch.norm((self.x.Exp() @ inputs).translation() - self.obstacle_pose.translation()) - 0.1
        return constraint_violation.unsqueeze(0)

    def forward(self, inputs):
        return self.obj(inputs), self.cnst(inputs)

if __name__ == "__main__":
    euler_angles = np.array([[0.0, 0.0, np.pi/8]])
    quaternion = pp.euler2SO3(euler_angles).to(torch.float)
    input = pp.SO3(quaternion).to(device)
    input = pp.SE3(torch.cat([torch.zeros(1, 3).to(device), input.tensor()], dim=-1))

    admm_model = RobotArmPoseOptimization(1).to(device)
    inner_optimizer_x = torch.optim.Adam([admm_model.x], lr=1e-3)
    inner_schd_x = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_x, step_size=10, gamma=0.5)
    inner_optimizer_z = torch.optim.Adam([admm_model.z], lr=1e-3)
    inner_schd_z = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_z, step_size=10, gamma=0.5)
    optimizer = ADMMOptim(admm_model, inner_optimizer_x, inner_optimizer_z)

    scheduler = CnstOptSchduler(optimizer, steps=50, inner_scheduler=[inner_schd_x, inner_schd_z], \
                                inner_iter=100, object_decrease_tolerance=1e-5, violation_tolerance=1e-5, \
                                verbose=True)
    while scheduler.continual():
        loss = optimizer.step(input)
        scheduler.step(loss)

    print('-----------Optimized Result----------------')
    print("Optimized Robot Arm Pose:")
    print("Translation:", admm_model.x.translation())
    print("Rotation (Euler angles):", admm_model.x.euler())

    # Retrieve the optimized objective values for x and z
    # optimized_obj_x, optimized_obj_z = admm_model.obj_all(input)
    # print("Optimized Objective (x):", optimized_obj_x)
    # print("Optimized Objective (z):", optimized_obj_z)
    # # Verification
    # optimized_pose = admm_model.x
    # target_pose = admm_model.target_pose
    # obstacle_pose = admm_model.obstacle_pose

    # # Check if the optimized pose is close to the target pose
    # distance_to_target = torch.norm(optimized_pose.Log() - target_pose.Log())
    # threshold = 1e-3
    # assert distance_to_target < threshold, f"Optimized pose is not close enough to the target pose. Distance: {distance_to_target}"

    # # Check if the optimized pose satisfies the constraint
    # distance_to_obstacle = torch.norm(optimized_pose.translation() - obstacle_pose.translation())
    # min_distance = 0.1
    # assert distance_to_obstacle > min_distance, f"Optimized pose violates the obstacle avoidance constraint. Distance: {distance_to_obstacle}"

    print("Verification passed!")
    optimized_pose = admm_model.x
    target_pose = admm_model.target_pose
    obstacle_pose = admm_model.obstacle_pose

    # Check if the optimized pose is close to the target pose
    distance_to_target = torch.norm(optimized_pose.tensor() - target_pose.tensor())
    threshold = 1e-3
    assert distance_to_target < threshold, f"Optimized pose is not close enough to the target pose. Distance: {distance_to_target}"

    # Check if the optimized pose satisfies the constraint
    distance_to_obstacle = torch.norm(optimized_pose.translation() - obstacle_pose.translation())
    min_distance = 0.1
    assert distance_to_obstacle > min_distance, f"Optimized pose violates the obstacle avoidance constraint. Distance: {distance_to_obstacle}"

    print("Verification passed!")
