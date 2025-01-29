import torch
import os
import random
from diffusers import DiffusionPipeline  # Changé pour DiffusionPipeline
from diffusers.utils import logging
from datetime import datetime

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

# Pipe with opti
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True,
    variant="fp16"
).to("mps")

# Enable because has low VRAM
pipe.enable_attention_slicing()

# Create a random generator for image generation using MPS (Apple Silicon GPU)
# - torch.Generator("mps"): initializes a random number generator for M1/M2 GPU
# - random.randint(0, 999999999): generates a random seed number
# - manual_seed(): sets the seed for reproducible results
# Using the same seed will generate the same image
generator = torch.Generator("mps").manual_seed(random.randint(0, 999999999))

# Prompt
prompt = os.getenv("PROMPT")

# Negative prompt
negative_prompt = os.getenv("NEGATIVE_PROMPT")

# Image generation
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=generator,
    num_inference_steps=100,
    guidance_scale=12,
    width=1024,
    height=1024,
    num_images_per_prompt=1
).images[0]

# Save image
image.save(image_path)
