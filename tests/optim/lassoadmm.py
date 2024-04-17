import torch
import numpy as np
import pypose as pp
from torch import nn
import numpy as np
from pypose.optim import ADMMOptim
from pypose.optim.scheduler import CnstOptSchduler
from pypose.optim.scheduler import CnstOptSchduler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import numpy as np

class LassoADMM(nn.Module):
    def __init__(self, A, b, tau, rho, device='cpu'):
        super().__init__()
        self.A = A
        self.b = b
        self.tau = tau
        self.rho = rho
        self.device = device

        num_features = A.size(1)
        self.x = nn.Parameter(torch.zeros(num_features, device=device))
        self.z = nn.Parameter(torch.zeros(num_features, device=device))
        self.u = nn.Parameter(torch.zeros(num_features, device=device))

    def obj(self,inputs):
        """Calculate the LASSO objective for the current x and z."""
        residual = torch.norm(torch.mm(self.A, self.x.unsqueeze(-1)) - self.b, p=2)
        regularization = self.tau * torch.norm(self.z, p=1)
        # print(residual**2 + regularization)
        return residual**2 + regularization

    def obj_all(self,inputs):
        """Separately calculate and return the components of the objective related to x and z."""
        print(torch.norm(torch.mm(self.A, self.x.unsqueeze(-1)) - self.b, p=2)**2)
        g_x = torch.norm(torch.mm(self.A, self.x.unsqueeze(-1)) - self.b, p=2)**2
        print(g_x)
        h_z = self.tau * torch.norm(self.z, p=1)
        return g_x, h_z

    def cnst(self,inputs):
        """Calculate the constraint violation."""
        return torch.norm(self.x - self.z, p=2)

    def forward(self,inputs):
        """Perform one step of the ADMM."""
        return self.obj(inputs), self.cnst(inputs)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(0)  # For reproducibility
    A = torch.randn(100, 20).to(device)
    b = torch.mm(A, torch.randn(20, 1).to(device)) + 0.1 * torch.randn(100, 1).to(device)
    A_p=A.detach().cpu().numpy()
    b_p=b.detach().cpu().numpy()
    tau = 0.1
    rho = 1.0
    model = LassoADMM(A, b, tau, rho, device)

    # Assume ADMMOptim and CnstOptScheduler are defined elsewhere and imported
    inner_optimizer_x = torch.optim.Adam([model.x], lr=1e-3)
    inner_scheduler_x = torch.optim.lr_scheduler.StepLR(inner_optimizer_x, step_size=10, gamma=0.5)
    inner_optimizer_z = torch.optim.Adam([model.z], lr=1e-3)
    inner_scheduler_z = torch.optim.lr_scheduler.StepLR(inner_optimizer_z, step_size=10, gamma=0.5)
    optimizer = ADMMOptim(model, inner_optimizer_x, inner_optimizer_z)
    scheduler = CnstOptSchduler(optimizer, steps=500, inner_scheduler=[inner_scheduler_x, inner_scheduler_z],
                                 inner_iter=50, object_decrease_tolerance=1e-5, violation_tolerance=1e-5,
                                 verbose=True)

    # Run the optimization process
    while scheduler.continual():
            loss = optimizer.step(input)
            scheduler.step(loss)

    print('-----------Optimized Result----------------')
    print("x*:", model.x.detach().cpu().numpy())
    print("z*:", model.z.detach().cpu().numpy())
    # Setting up the Lasso regression model
    tau = 0.1  # Regularization strength
    lasso_model = Lasso(alpha=tau, fit_intercept=False)
    lasso_model.fit(A_p, b_p)

    # The coefficients
    print("Lasso coefficients:", lasso_model.coef_)

    # Prediction and error
    b_pred = lasso_model.predict(A_p)
    mse = mean_squared_error(b_p, b_pred)
    print("Mean Squared Error sklearn:", mse)
    A_np = A.detach().cpu().numpy()  # Assuming A is your feature matrix tensor
    b_np = b.detach().cpu().numpy()  # Assuming b is your target vector tensor
    x_np = model.x.detach().cpu().numpy()  # Coefficients found by ADMM

    # Calculate predictions and MSE
    b_pred = np.dot(A_np, x_np)
    mse = np.mean((b_np - b_pred) ** 2)
    print("Final Coefficients:", x_np)
    print("Mean Squared Error:", mse)
