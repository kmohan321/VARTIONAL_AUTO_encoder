import torch
import torch.nn as nn
from model import VAE
import torchvision.utils as vutils
from torchvision.utils import save_image
from celeb_data import dataloader

# Assuming you have your VAE class defined as shown in your code

# 1. Load your trained model
latent_dim = 64  # Make sure this matches your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim, device).to(device)

# Load the saved state dict
vae.load_state_dict(torch.load("model.pth"))
vae.eval()  # Set the model to evaluation mode

# 2. Sample from the latent space
def sample_latent(batch_size):
    return torch.randn(batch_size, 512, 8, 8).to(device)

# 3. Generate images
def generate_images(vae, num_images, latent_dim):
    with torch.no_grad():
        generated_noise = sample_latent(num_images)
        latent_samples = vae.inference_model(generated_noise)
        generated_images = vae.decoder(vae.decoder_input(latent_samples))
    return generated_images

# Generate a batch of images
num_images = 16
# generated_images = generate_images(vae, num_images, latent_dim)
images,text = next(iter(dataloader))
images = images.to(device)
generated_images,_,_ = vae.forward_decoder(images)
# print(generated_images.shape)
# Visualize or save the generated images
# grid = vutils.make_grid(generated_images, normalize=True, value_range=(-1, 1))
save_image(generated_images, "generated_images.png")
print("Images generated and saved as 'generated_images.png'")