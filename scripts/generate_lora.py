import torch
import os
import random
from diffusers import DiffusionPipeline
from diffusers.utils import logging
from datetime import datetime
from dotenv import load_dotenv

# Load env var for prompt
load_dotenv()

# Add log verbose
logging.set_verbosity_error()

# Output dir image generated
output_dir = "images_generated"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate image name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_name = f"{timestamp}_lora_image.png"
image_path = os.path.join(output_dir, image_name)

# Load base model and LoRA weights
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True,
    variant="fp16"
).to("mps")

# Load your trained LoRA weights
pipe.load_lora_weights("lora_output/final")

# Enable because has low VRAM
pipe.enable_attention_slicing()

# Create a random generator for image generation using MPS
generator = torch.Generator("mps").manual_seed(random.randint(0, 999999999))

# Get prompts from environment variables
prompt = os.getenv('PROMPT_LORA')  # Use specific prompt for LoRA
negative_prompt = os.getenv('NEGATIVE_PROMPT')

# Generate image
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=generator,
    num_inference_steps=100,
    guidance_scale=8.5,
    width=1024,
    height=1024,
    num_images_per_prompt=1
).images[0]

# Save image
image.save(image_path)
