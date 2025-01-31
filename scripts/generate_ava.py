import torch
import os
import random
from diffusers import DiffusionPipeline
from diffusers.utils import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable verbose logging
logging.set_verbosity_error()

# Create the output directory if it doesn't exist
output_dir = "images_generated"
os.makedirs(output_dir, exist_ok=True)

# 🕦Generate a timestamped image filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_name = f"{timestamp}_image.png"
image_path = os.path.join(output_dir, image_name)

# 📌 Load the base model
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = DiffusionPipeline.from_pretrained(
    base_model,
    variant="fp16",
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False,
)

# ⚡ Move the pipeline to GPU (CUDA for RTX, MPS if still on Mac)
device = "cuda" if torch.cuda.is_available() else "mps"
pipeline.to(device)

# 📌 Load the LoRA weights
lora_path = "lora_output/ava/pytorch_lora_weights.safetensors"

try:
    pipeline.load_lora_weights(lora_path)
    print(f"✅ LoRA successfully loaded from {lora_path}")
except Exception as e:
    print(f"⚠️ Error loading LoRA: {e}")

if os.path.exists(lora_path):
    try:
        pipeline.load_lora_weights(lora_path)
        print(f"✅ LoRA successfully loaded from {lora_path}")
    except Exception as e:
        print(f"⚠️ Error loading LoRA: {e}")
else:
    print(f"⚠️ LoRA file '{lora_path}' not found. Generating image without LoRA.")

# 🔀 Generate a random seed
generator = torch.Generator(device).manual_seed(random.randint(0, 999999999))

# 🗣️ Prompts
prompt = os.getenv('PROMPT_AVA', "a photo of sks ava, ultra realistic, 8k")
negative_prompt = os.getenv('NEGATIVE_PROMPT', "blurry, low quality")

# 🔥 Generate the image
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=generator,
    num_inference_steps=30,  # Number of denoising steps
    guidance_scale=7.5,  # Prompt influence
).images[0]

# 💾Save the generated image
image.save(image_path)
print(f"✅ Image generated and saved at: {image_path}")
