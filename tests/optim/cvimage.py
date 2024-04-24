import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from matplotlib import pyplot as plt
from PIL import Image
import os
import bm3d
# Load the noisy image
# noisy_image_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\pypose\tests\optim\ddadmm.jpg'  # Replace with your image path
# noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
image_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\pypose\tests\optim\ddadmm.jpg' # Replace with the path to your image file
image = Image.open(image_path).convert('L')  # Convert the image to grayscale
image_np = np.array(image) / 255.0  # Normalize the image to [0, 1]
# print(image_np)
# clean_image = torch.from_numpy(image_np).float().to(device)

# Add noise to the clean image
noise_level = 0.1
noise = noise_level * np.random.randn(*image_np.shape)
noisy_image = image_np + noise
noisy_image = np.clip(noisy_image, 0, 1)
# print(noisy_image)
if not os.path.exists(image_path):
    raise ValueError(f"File not found: {image_path}")


# Check if the image is loaded properly
if noisy_image is None:
    raise ValueError("Could not open or find the image.")

# Denoise the image using Non-Local Means Denoising
noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)

# Denoise the image using Non-Local Means Denoising
denoised_image = cv2.fastNlMeansDenoising(noisy_image_uint8, None, h=10, templateWindowSize=7, searchWindowSize=21)

# If you need to bring the denoised image back to floating point representation
denoised_image_float = denoised_image.astype(np.float32) / 255.0
median_denoised_image = cv2.medianBlur(noisy_image_uint8, ksize=3)
bm3d_denoised_image = bm3d.bm3d(noisy_image_uint8, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
# Load the clean image (ground truth) if available for PSNR and SSIM comparison
clean_image_path = r'C:\Users\harsh\OneDrive\Documents\GitHub\pypose\tests\optim\ddadmm.jpg' # Replace with your image path
clean_image = cv2.imread(clean_image_path, cv2.IMREAD_GRAYSCALE)

if clean_image is None:
    raise ValueError("Could not open or find the clean image.")

# Ensure both images are floating point in the range [0, 1] for SSIM
clean_image_float = img_as_float(clean_image)
denoised_image_float = img_as_float(median_denoised_image)

# Compute PSNR between the original and denoised image
psnr_value = cv2.PSNR(clean_image, denoised_image)
print(f"PSNR: {psnr_value}")

# Compute SSIM between the original and denoised image
ssim_value = ssim(clean_image_float, denoised_image_float,data_range=1.0)
print(f"SSIM: {ssim_value}")

# Optionally, display the images
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(clean_image, cmap='gray')
plt.title('Clean Image')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.subplot(1, 3, 3)
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')

plt.tight_layout()
plt.show()
