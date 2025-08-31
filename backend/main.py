from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from diffusers import FluxPipeline
import uuid
import os
from datetime import datetime
import json
import asyncio
from typing import List, Optional
import base64
from io import BytesIO
from PIL import Image

app = FastAPI(title="FLUX Image Generator API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("generated_images", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="generated_images"), name="static")

# Initialize the model (lazy loading)
pipe = None
metadata_file = "image_metadata.json"

def load_model():
    global pipe
    if pipe is None:
        print("Loading FLUX.1-dev model...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        print("Model loaded successfully!")

def load_metadata():
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return []

def save_metadata(metadata):
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

class GenerateImageRequest(BaseModel):
    prompt: str
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    guidance_scale: Optional[float] = 3.5
    num_inference_steps: Optional[int] = 50
    seed: Optional[int] = None

class ImageResponse(BaseModel):
    id: str
    filename: str
    url: str
    prompt: str
    created_at: str
    width: int
    height: int

@app.get("/")
async def root():
    return {"message": "FLUX Image Generator API"}

@app.post("/api/generate", response_model=ImageResponse)
async def generate_image(request: GenerateImageRequest):
    try:
        # Load model if not already loaded
        load_model()
        
        # Generate unique filename
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"
        filepath = os.path.join("generated_images", filename)
        
        # Set seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator("cpu").manual_seed(request.seed)
        else:
            generator = torch.Generator("cpu").manual_seed(0)
        
        print(f"Generating image with prompt: {request.prompt}")
        
        # Generate image
        image = pipe(
            request.prompt,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            max_sequence_length=512,
            generator=generator
        ).images[0]
        
        # Save image
        image.save(filepath)
        
        # Create metadata
        metadata = load_metadata()
        image_data = {
            "id": image_id,
            "filename": filename,
            "url": f"/static/{filename}",
            "prompt": request.prompt,
            "created_at": datetime.now().isoformat(),
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps
        }
        metadata.append(image_data)
        save_metadata(metadata)
        
        return ImageResponse(**image_data)
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

@app.get("/api/images", response_model=List[ImageResponse])
async def list_images():
    try:
        metadata = load_metadata()
        # Return in reverse order (newest first)
        return [ImageResponse(**img) for img in reversed(metadata)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load images: {str(e)}")

@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    try:
        metadata = load_metadata()
        image_to_delete = None
        updated_metadata = []
        
        for img in metadata:
            if img["id"] == image_id:
                image_to_delete = img
            else:
                updated_metadata.append(img)
        
        if not image_to_delete:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Delete file
        filepath = os.path.join("generated_images", image_to_delete["filename"])
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Update metadata
        save_metadata(updated_metadata)
        
        return {"message": "Image deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)