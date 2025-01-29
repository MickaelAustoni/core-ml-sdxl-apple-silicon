import torch
import os
import random
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import logging
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import face_recognition

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
image_name = f"{timestamp}_inpainted.png"
image_path = os.path.join(output_dir, image_name)

# Load pipeline inpainting
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True,
    variant="fp16"
).to("mps")

# Enable because has low VRAM
pipe.enable_attention_slicing()

# Load and process input image for face detection
input_path = os.path.join(os.getcwd(), os.getenv('INPUT_IMG_INPAINTING'))
init_image = Image.open(input_path)
init_image = init_image.resize((1024, 1024))

# Detect face in the input image
image_np = face_recognition.load_image_file(input_path)
face_locations = face_recognition.face_locations(image_np)

# Create mask (white = areas to regenerate, black = areas to preserve)
mask = Image.new("L", init_image.size, 255)
draw = ImageDraw.Draw(mask)

# Process detected faces and create mask
if face_locations:
    # Calculate scaling factors for face coordinates
    height_ratio = 1024 / image_np.shape[0]
    width_ratio = 1024 / image_np.shape[1]

    # Get coordinates of the first detected face
    top, right, bottom, left = face_locations[0]

    # Scale coordinates to match resized image
    top = int(top * height_ratio)
    right = int(right * width_ratio)
    bottom = int(bottom * height_ratio)
    left = int(left * width_ratio)

    # Add margin around the face
    margin = 50
    face_box = (
        max(0, left - margin),
        max(0, top - margin),
        min(1024, right + margin),
        min(1024, bottom + margin)
    )

    # Draw face area in mask
    draw.rectangle(face_box, fill=0)  # Black = area to preserve
else:
    print("No face detected in the input image!")
    exit()

# Initialize random generator for MPS (Apple Silicon GPU)
generator = torch.Generator("mps").manual_seed(random.randint(0, 999999999))

# Get prompts from environment variables
prompt = os.getenv('PROMPT_INPAINTING')
negative_prompt = os.getenv('NEGATIVE_PROMPT')

# Generate the image with inpainting
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask,
    generator=generator,
    num_inference_steps=100,
    guidance_scale=12,
    strength=0.75,
    num_images_per_prompt=1
).images[0]

# Save the generated image
image.save(image_path)
