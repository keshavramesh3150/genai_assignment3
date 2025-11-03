import torch
import torch.nn as nn
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from torchvision.utils import save_image
import io
import math

from helper_lib.model import get_model, Z_DIM

# Initialize FastAPI app
app = FastAPI(title="GAN MNIST Digit Generator")

# Check for available hardware acceleration (MPS > CUDA > CPU)
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print(f"API using device: {DEVICE}")

# Model Loading
MODEL_PATH = "gan_generator.pth"

try:
    # Load the model architecture
    generator, _ = get_model("GAN")
    # Load the trained weights
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.to(DEVICE)
    generator.eval() # Set to evaluation mode
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")

# Helper Function 
def generate_image_response(images_tensor, nrow=1):
    """Converts a tensor of images into a PNG StreamingResponse."""
    # Un-normalize from [-1, 1] to [0, 1]
    images_tensor = (images_tensor * 0.5) + 0.5
    
    # Create an in-memory buffer
    img_buffer = io.BytesIO()
    # Save the tensor to the buffer as a PNG
    save_image(images_tensor, img_buffer, format="PNG", nrow=nrow)
    img_buffer.seek(0) # Rewind the buffer to the beginning
    
    return StreamingResponse(img_buffer, media_type="image/png")

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the GAN MNIST Digit Generator API."}

@app.get("/generate", response_class=StreamingResponse)
async def generate_single_digit():
    """
    Generates a single new handwritten digit and returns it as a PNG image.
    """
    try:
        # Create noise vector for a batch of 1
        noise = torch.randn(1, Z_DIM).to(DEVICE)
        
        # Generate image
        with torch.no_grad():
            fake_image_tensor = generator(noise)
        
        # Return the single image as a response
        return generate_image_response(fake_image_tensor, nrow=1)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_grid", response_class=StreamingResponse)
async def generate_digit_grid(count: int = Query(9, gt=0, le=100)):
    """
    Generates a grid of [count] new handwritten digits.
    
    - **count**: The number of digits to generate (default: 9, max: 100).
    """
    try:
        # Create a batch of noise vectors
        noise = torch.randn(count, Z_DIM).to(DEVICE)
        
        # Generate a batch of images
        with torch.no_grad():
            fake_images_tensor = generator(noise)
        
        # Calculate optimal number of rows for a square-ish grid
        nrow = int(math.ceil(math.sqrt(count)))
        
        # Return the grid of images as a response
        return generate_image_response(fake_images_tensor, nrow=nrow)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))