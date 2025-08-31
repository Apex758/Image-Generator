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
import requests
import aiohttp
from dotenv import load_dotenv

# Load environment variables from .env file
print("Loading environment variables from .env file...")
load_dotenv()
print(f"Environment variables loaded. HF_API_TOKEN configured: {bool(os.getenv('HF_API_TOKEN'))}")

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

# Configuration
USE_HF_API = os.getenv("USE_HF_API", "true").lower() == "true"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # Get your token from https://huggingface.co/settings/tokens
HF_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

# Initialize the local model (lazy loading) - only if not using HF API
pipe = None
metadata_file = "image_metadata.json"

def load_model():
    global pipe
    if pipe is None and not USE_HF_API:
        print("Loading FLUX.1-dev model locally...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        print("Local model loaded successfully!")

def load_metadata():
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return []

def save_metadata(metadata):
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

async def generate_with_hf_api(prompt: str, width: int = 1024, height: int = 1024,
                               guidance_scale: float = 3.5, num_inference_steps: int = 50,
                               seed: Optional[int] = None) -> Image.Image:
    """Generate image using Hugging Face Inference API"""
    
    print(f"==== START: generate_with_hf_api ====")
    print(f"Starting HF API generation with prompt: '{prompt[:50]}...' (truncated)")
    
    if not HF_API_TOKEN:
        print("ERROR: HF_API_TOKEN is not configured")
        raise HTTPException(
            status_code=500,
            detail="Hugging Face API token not configured. Please set HF_API_TOKEN environment variable."
        )
    
    # Mask token for logging (show first 4 chars only)
    masked_token = HF_API_TOKEN[:4] + "..." if len(HF_API_TOKEN) > 4 else "***"
    print(f"Using HF API token: {masked_token}")
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    # Prepare payload for HF API
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps
        }
    }
    
    if seed is not None:
        payload["parameters"]["seed"] = seed
    
    print(f"Sending request to HF API: {HF_API_URL}")
    print(f"Request payload: {json.dumps(payload)}")
    
    try:
        async with aiohttp.ClientSession() as session:
            print("aiohttp session created")
            try:
                print("Sending POST request to HF API...")
                print(f"Request URL: {HF_API_URL}")
                print(f"Request headers: {headers}")
                print(f"Request payload: {json.dumps(payload)}")
                async with session.post(HF_API_URL, headers=headers, json=payload, timeout=120) as response:
                    print(f"Received response from HF API with status: {response.status}")
                    
                    if response.status == 200:
                        print("HF API returned 200 OK, reading response body...")
                        image_bytes = await response.read()
                        print(f"Received {len(image_bytes)} bytes from HF API")
                        
                        try:
                            image = Image.open(BytesIO(image_bytes))
                            print(f"Successfully parsed image: {image.format}, {image.size}")
                            return image
                        except Exception as img_error:
                            print(f"Error parsing image data: {type(img_error).__name__}: {str(img_error)}")
                            raise HTTPException(status_code=500, detail=f"Failed to parse image from API response: {str(img_error)}")
                    
                    elif response.status == 503:
                        # Model is loading
                        error_text = await response.text()
                        print(f"HF API returned 503 (model loading): {error_text}")
                        raise HTTPException(status_code=503, detail="Model is loading on Hugging Face. Please try again in a few moments.")
                    
                    else:
                        error_text = await response.text()
                        print(f"HF API Error: Status {response.status}")
                        print(f"Response headers: {response.headers}")
                        print(f"Response body: {error_text}")
                        raise HTTPException(status_code=response.status, detail=f"Hugging Face API error: {error_text}")
            
            except aiohttp.ClientError as client_error:
                print(f"aiohttp client error: {type(client_error).__name__}: {str(client_error)}")
                raise HTTPException(status_code=500, detail=f"Network error connecting to Hugging Face API: {str(client_error)}")
    
    except Exception as e:
        print(f"==== ERROR: Unexpected error in generate_with_hf_api ====")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating image with Hugging Face API: {str(e)}")
    finally:
        print(f"==== END: generate_with_hf_api ====")

def generate_with_local_model(prompt: str, width: int = 1024, height: int = 1024,
                             guidance_scale: float = 3.5, num_inference_steps: int = 50,
                             seed: Optional[int] = None) -> Image.Image:
    """Generate image using local model"""
    load_model()
    
    # Set seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator("cpu").manual_seed(seed)
    else:
        generator = torch.Generator("cpu").manual_seed(0)
    
    print(f"Generating image locally with prompt: {prompt}")
    
    # Generate image
    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=generator
    ).images[0]
    
    return image

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
    generation_method: Optional[str] = "unknown"  # "hf_api" or "local" or "unknown" for existing

@app.get("/")
async def root():
    method = "Hugging Face API" if USE_HF_API else "Local Model"
    return {
        "message": "FLUX Image Generator API", 
        "generation_method": method,
        "hf_api_configured": bool(HF_API_TOKEN) if USE_HF_API else "N/A"
    }

@app.post("/api/generate", response_model=ImageResponse)
async def generate_image(request: GenerateImageRequest):
    try:
        print("==== START: /api/generate endpoint called ====")
        print(f"Request data: {request.dict()}")
        
        # Generate unique filename
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"
        filepath = os.path.join("generated_images", filename)
        
        print(f"Generating image with prompt: {request.prompt}")
        print(f"Using method: {'Hugging Face API' if USE_HF_API else 'Local Model'}")
        print(f"Request parameters: width={request.width}, height={request.height}, guidance_scale={request.guidance_scale}, steps={request.num_inference_steps}, seed={request.seed}")
        
        # Generate image based on configuration
        if USE_HF_API:
            print(f"HF API Token configured: {bool(HF_API_TOKEN)}")
            print(f"HF API URL: {HF_API_URL}")
            try:
                image = await generate_with_hf_api(
                    request.prompt,
                    request.width,
                    request.height,
                    request.guidance_scale,
                    request.num_inference_steps,
                    request.seed
                )
                print("Image successfully generated from HF API")
                generation_method = "hf_api"
            except Exception as hf_error:
                print(f"Detailed HF API error: {type(hf_error).__name__}: {str(hf_error)}")
                if isinstance(hf_error, HTTPException):
                    print(f"HTTP Exception status: {hf_error.status_code}, detail: {hf_error.detail}")
                raise
        else:
            try:
                image = generate_with_local_model(
                    request.prompt,
                    request.width,
                    request.height,
                    request.guidance_scale,
                    request.num_inference_steps,
                    request.seed
                )
                print("Image successfully generated from local model")
                generation_method = "local"
            except Exception as local_error:
                print(f"Detailed local model error: {type(local_error).__name__}: {str(local_error)}")
                raise
        
        # Save image
        try:
            print(f"Saving image to {filepath}")
            image.save(filepath)
            print("Image saved successfully")
        except Exception as save_error:
            print(f"Error saving image: {type(save_error).__name__}: {str(save_error)}")
            raise
        
        # Create metadata
        try:
            print("Loading and updating metadata")
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
                "num_inference_steps": request.num_inference_steps,
                "generation_method": generation_method
            }
            metadata.append(image_data)
            save_metadata(metadata)
            print("Metadata updated successfully")
        except Exception as metadata_error:
            print(f"Error updating metadata: {type(metadata_error).__name__}: {str(metadata_error)}")
            raise
        
        print(f"Successfully completed image generation for ID: {image_id}")
        return ImageResponse(**image_data)
        
    except HTTPException:
        print("==== ERROR: HTTPException in /api/generate endpoint ====")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"==== ERROR: Exception in /api/generate endpoint ====")
        print(f"Error type: {error_type}")
        print(f"Error message: {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {error_type}: {error_msg}")
    finally:
        print("==== END: /api/generate endpoint processing ====")

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

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "use_hf_api": USE_HF_API,
        "hf_api_configured": bool(HF_API_TOKEN),
        "generation_method": "Hugging Face API" if USE_HF_API else "Local Model"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)