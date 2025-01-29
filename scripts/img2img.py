import torch
import os
import random
from diffusers import StableDiffusionXLImg2ImgPipeline  # Changement ici
from diffusers.utils import logging
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image  # Pour charger l'image de base

# Load env var for prompt
load_dotenv()

# Add log verbose
logging.set_verbosity_error()

# Output dir image generated
output_dir = "generated_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate image name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_name = f"{timestamp}_image.png"
image_path = os.path.join(output_dir, image_name)

# Charger le pipeline img2img
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True,
    variant="fp16"
).to("mps")

# Enable because has low VRAM
pipe.enable_attention_slicing()

# Load input image
init_image = Image.open(os.path.join(os.getcwd(), os.getenv('INPUT_IMG')))

# Resize image if needed
init_image = init_image.resize((1024, 1024))

# Create a random generator for image generation using MPS (Apple Silicon GPU)
# - torch.Generator("mps"): initializes a random number generator for M1/M2 GPU
# - random.randint(0, 999999999): generates a random seed number
# - manual_seed(): sets the seed for reproducible results
# Using the same seed will generate the same image
generator = torch.Generator("mps").manual_seed(random.randint(0, 999999999))

# Prompt
prompt = os.getenv('PROMPT_IMG_TO_IMG')
negative_prompt = os.getenv('NEGATIVE_PROMPT')

# Image generation avec contrôle de la force de modification
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.75,  # Contrôle combien l'image originale est modifiée (0-1)
    generator=generator,
    num_inference_steps=100,
    guidance_scale=12,
    num_images_per_prompt=1
).images[0]

# Save image
image.save(image_path)
