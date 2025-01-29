from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTokenizerFast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Base configuration
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
training_dir = "images_training"
prompt = "photo of sks person"

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

# Load components for training
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
text_encoder_2 = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder_2")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

# Move to device
device = "mps"
text_encoder.to(device)
text_encoder_2.to(device)
vae.to(device)
unet.to(device)

# Freeze VAE and text encoders
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
vae.requires_grad_(False)

# Encode text prompt
tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
tokens_2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt")

with torch.no_grad():
    encoder_hidden_states = text_encoder(tokens.input_ids.to(device))[0]
    encoder_hidden_states_2 = text_encoder_2(tokens_2.input_ids.to(device))[0]

# Prepare data
dataset = CustomDataset(training_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training config
num_epochs = 100
learning_rate = 1e-4

optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    unet.train()
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)

        # Encode images
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (1,), device=device)

        # Add time embeddings
        added_cond_kwargs = {
            "text_embeds": torch.randn((1, 1280), device=device),
            "time_ids": torch.randn((1, 6), device=device)
        }

        # Forward pass
        noise_pred = unet(
            latents + noise,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

# Save model
output_dir = "lora_weights"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
unet.save_pretrained(os.path.join(output_dir, "unet"))
