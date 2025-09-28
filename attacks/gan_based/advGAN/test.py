from generator_2d import AdvGANGenerator
from PIL import Image
import torchvision.transforms as transforms



import torch

G = AdvGANGenerator(in_ch=1) 
G.load_state_dict(torch.load("outputs/advgan/generator.pt", map_location="cuda"))
G.eval()
G.to("cuda") 
# Define preprocessing transformer is used to transform images to matrices and grayscale it
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # convert to 1 channel
    transforms.Resize((64, 254)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # scale to [-1,1] if trained that way
])


img = Image.open("C:/Users/Idra/Desktop/imgECG_AdvGAN_FL/data/raw/NSR/JS10675_0.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # shape: (1, 1, 64, 64)

with torch.no_grad():
    perturbation = G(img_tensor.cuda())  # if G is on CUDA


import matplotlib.pyplot as plt
import numpy as np
# Detach and move to CPU
perturbation_np = perturbation.squeeze().cpu().numpy()


img_matrix = img_tensor.squeeze().cpu().numpy()



fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img_matrix, cmap="gray")  # original
axs[0].set_title("Original Image (Matrix)")

axs[1].imshow(perturbation_np, cmap="gray")  # perturbation
axs[1].set_title("Generated Perturbation (Matrix)")

plt.show()

adv_img_tensor = img_tensor.cuda() + perturbation
adv_img_tensor = torch.clamp(adv_img_tensor, -1, 1)  # keep values in [-1,1]


# Convert to numpy for visualization
adv_img_matrix = adv_img_tensor.squeeze().cpu().numpy()


fig, axs = plt.subplots(1, 2, figsize=(10, 5))


axs[0].imshow(img_matrix, cmap="gray")  # original
axs[0].set_title("Original Image (Matrix)")

axs[1].imshow(adv_img_matrix, cmap="gray")  # perturbation
axs[1].set_title("adv (Matrix)")
plt.show()

# 2️⃣ Compute difference
difference = adv_img_matrix - img_matrix[0]

print("Mean perturbation:", difference.mean())
print("Max perturbation:", difference.max())
print("Min perturbation:", difference.min())
print("L2 norm of perturbation:", np.linalg.norm(difference))

# Save the difference matrix

np.savetxt("difference_matrix.csv", difference, delimiter=",")

# Visualize difference
plt.imshow(difference, cmap="bwr")  # red=negative, blue=positive
plt.title("Difference (Original - Perturbation)")
plt.colorbar()
plt.show()



plt.imshow(perturbation_np, cmap="gray")
plt.title("Generated Perturbation")
plt.colorbar()
plt.show()