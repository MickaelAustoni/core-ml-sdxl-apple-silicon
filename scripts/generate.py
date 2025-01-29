import torch
import os
from diffusers import DiffusionPipeline  # Changé pour DiffusionPipeline
from diffusers.utils import logging
from datetime import datetime

# Add log verbose
logging.set_verbosity_error()

output_dir = "generated_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Génération d'un nom de fichier avec timestamp
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

# Prompt
prompt = (
    "professional portrait photography of a woman, "
    "crystal clear details, perfect lighting, high-end fashion photography, "
    "85mm lens, shot on Hasselblad, supreme quality, 8k resolution, "
    "perfect composition, magazine quality, award winning photography"
)

# Negative prompt
negative_prompt = (
    "deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, "
    "cartoon, drawing, anime, mutated hands and fingers, deformed, "
    "distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, "
    "extra limb, missing limb, floating limbs, disconnected limbs, "
    "mutation, mutated, ugly, disgusting, blurry, bad art"
)

# Image generation
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=70,
    guidance_scale=9,
    width=1024,
    height=1024,
    num_images_per_prompt=1
).images[0]

# Save image
image.save(image_path)
