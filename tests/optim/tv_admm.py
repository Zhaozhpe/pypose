import torch
import torch.nn as nn
import pypose as pp
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from pypose.optim import ADMMOptim
from pypose.optim.scheduler import CnstOptSchduler
from PIL import Image
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageDenoisingADMM(nn.Module):
    def __init__(self, image_size, lambda_tv):
        super().__init__()
        # self.x = nn.Parameter(torch.randn(image_size, device=device))
        # self.z = nn.Parameter(torch.randn(image_size[0]-1, image_size[1]-1, 2, device=device))
        # self.lambda_tv = lambda_tv
        self.x = nn.Parameter(torch.randn(image_size, device=device))
        self.z = nn.Parameter(torch.randn(image_size[0], image_size[1], 2, device=device))
        self.lambda_tv = lambda_tv

    def obj(self, noisy_image):
        data_fidelity = torch.norm(self.x - noisy_image)**2
        tv_h = torch.norm(self.z[..., 0], p=1)
        tv_v = torch.norm(self.z[..., 1], p=1)
        tv_regularization = self.lambda_tv * (tv_h + tv_v)
        return data_fidelity + tv_regularization

    def obj_all(self, noisy_image):
        data_fidelity = torch.norm(self.x - noisy_image)**2
        tv_h = torch.norm(self.z[..., 0], p=1)
        tv_v = torch.norm(self.z[..., 1], p=1)
        return data_fidelity, self.lambda_tv * (tv_h + tv_v)

    def cnst(self, noisy_image):
        constraint_h = self.x[:-1, :] - self.x[1:, :] - self.z[:-1, :, 0]
        constraint_v = self.x[:, :-1] - self.x[:, 1:] - self.z[:, :-1, 1]
        return torch.cat((constraint_h.flatten(), constraint_v.flatten()))

    def forward(self, noisy_image):
        return self.obj(noisy_image), self.cnst(noisy_image)

if __name__ == "__main__":
    # image_size = (256, 256)
    # lambda_tv = 0.01

    # # Generate a noisy image
    # clean_image = torch.rand(image_size, device=device)
    # noise = 0.1 * torch.randn(image_size, device=device)
    # noisy_image = clean_image + noise
    image_path = r"C:\Users\harsh\Downloads\ddadmm.jpg"  # Replace with the path to your image file
    image = Image.open(image_path).convert('L')  # Convert the image to grayscale
    image_np = np.array(image) / 255.0  # Normalize the image to [0, 1]
    clean_image = torch.from_numpy(image_np).float().to(device)

    # Add noise to the clean image
    noise_level = 0.1
    noise = noise_level * torch.randn_like(clean_image)
    noisy_image = clean_image + noise

    image_size = clean_image.shape
    lambda_tv = 0.1

    admm_model = ImageDenoisingADMM(image_size, lambda_tv).to(device)
    inner_optimizer_x = torch.optim.Adam([admm_model.x], lr=1e-2)
    inner_schd_x = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_x, step_size=30, gamma=0.5)
    inner_optimizer_z = torch.optim.Adam([admm_model.z], lr=1e-2)
    inner_schd_z = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer_z, step_size=30, gamma=0.5)
    optimizer = ADMMOptim(admm_model, inner_optimizer_x, inner_optimizer_z)

    scheduler = CnstOptSchduler(optimizer, steps=50, inner_scheduler=[inner_schd_x, inner_schd_z],
                                inner_iter=500, object_decrease_tolerance=1e-5, violation_tolerance=1e-5,
                                verbose=True)
    while scheduler.continual():
        loss = optimizer.step(noisy_image)
        scheduler.step(loss)

    print('-----------Optimized Result----------------')
    # print("Denoised Image:", admm_model.x)
    # mse = torch.mean((clean_image - admm_model.x) ** 2)
    # print( 10 * torch.log10(1.0 / mse))
    # ssim_value = ssim(clean_image.cpu().numpy(), admm_model.x.detach().cpu().numpy(), data_range=1.0)
    # Compute PSNR
    mse_torch = torch.mean((clean_image - admm_model.x) ** 2)
    psnr_torch = 10 * torch.log10(1.0 / mse_torch)
    print(f"PSNR (PyTorch): {psnr_torch}")

    # Calculate PSNR using OpenCV, scale images to [0, 255] range if necessary
    clean_image_cv = (clean_image.cpu().numpy() * 255).astype(np.uint8)
    denoised_image_cv = (admm_model.x.detach().cpu().numpy() * 255).astype(np.uint8)
    psnr_value_cv = cv2.PSNR(clean_image_cv, denoised_image_cv)
    print(f"PSNR (OpenCV): {psnr_value_cv}")
    # Compute SSIM
    ssim_value = ssim(clean_image.cpu().numpy(), admm_model.x.detach().cpu().numpy(), data_range=1.0)
    print(f"SSIM: {ssim_value}")

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(clean_image.cpu().numpy(), cmap='gray')
    plt.title('Clean Image')
    plt.subplot(132)
    plt.imshow(noisy_image.cpu().numpy(), cmap='gray')
    plt.title('Noisy Image')
    plt.subplot(133)
    plt.imshow(admm_model.x.detach().cpu().numpy(), cmap='gray')
    plt.title('Denoised Image')
    plt.tight_layout()
    plt.show()
