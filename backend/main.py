# Essential imports first (fast)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import uuid
from datetime import datetime
import json
import asyncio
from typing import List, Optional, Dict, Any
import base64
from io import BytesIO
import requests
import aiohttp
import aiofiles
from dotenv import load_dotenv
import re
import time
import random
import logging

# Lazy imports (only import when needed)
def lazy_import_torch():
    global torch
    if 'torch' not in globals():
        import torch
    return torch

def lazy_import_diffusers():
    global FluxPipeline
    if 'FluxPipeline' not in globals():
        from diffusers import FluxPipeline
    return FluxPipeline

def lazy_import_nltk():
    global nltk, sent_tokenize, word_tokenize, stopwords, pos_tag
    if 'nltk' not in globals():
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from nltk.tag import pos_tag
    return nltk

def lazy_import_image_processing():
    global Image, cairosvg
    if 'Image' not in globals():
        from PIL import Image
        import cairosvg
    return Image, cairosvg

def lazy_import_document_processing():
    global ET, Document, Inches, tempfile
    if 'ET' not in globals():
        import xml.etree.ElementTree as ET
        from docx import Document
        from docx.shared import Inches
        import tempfile
    return ET, Document, Inches, tempfile


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
IMAGES_DIR = os.path.join(ROOT_DIR, "generated_images")
SVG_TEMPLATES_DIR = os.path.join(BASE_DIR, "svg_templates")
SVG_EXPORTS_DIR = os.path.join(ROOT_DIR, "svg_exports")

# Create directories
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "static"), exist_ok=True)
os.makedirs(SVG_EXPORTS_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=IMAGES_DIR), name="static")
app.mount("/exports", StaticFiles(directory=SVG_EXPORTS_DIR), name="exports")

# Configuration
USE_HF_API = os.getenv("USE_HF_API", "true").lower() == "true"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # Get your token from https://huggingface.co/settings/tokens
HF_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

# Initialize the local model (lazy loading) - only if not using HF API
pipe = None
metadata_file = "image_metadata.json"
svg_metadata_file = "svg_metadata.json"

# Update the load_model function to use lazy imports
def load_model():
    global pipe
    if pipe is None and not USE_HF_API:
        print("Loading FLUX.1-dev model locally...")
        torch = lazy_import_torch()
        FluxPipeline = lazy_import_diffusers()
        
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        print("Local model loaded successfully!")

async def load_metadata():
    if os.path.exists(metadata_file):
        async with aiofiles.open(metadata_file, 'r') as f:
            return json.loads(await f.read())
    return []

async def save_metadata(metadata):
    async with aiofiles.open(metadata_file, 'w') as f:
        await f.write(json.dumps(metadata, indent=2))

async def load_svg_metadata():
    if os.path.exists(svg_metadata_file):
        async with aiofiles.open(svg_metadata_file, 'r') as f:
            return json.loads(await f.read())
    return []

async def save_svg_metadata(metadata):
    async with aiofiles.open(svg_metadata_file, 'w') as f:
        await f.write(json.dumps(metadata, indent=2))

async def generate_with_hf_api(prompt: str, width: int = 1024, height: int = 1024,
                               guidance_scale: float = 3.5, num_inference_steps: int = 50,
                               seed: Optional[int] = None):
    """Generate image using Hugging Face Inference API with retry mechanism"""
    Image, _ = lazy_import_image_processing()
    
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
    
    # Retry configuration
    max_retries = 3
    retry_delay_base = 5  # Base delay in seconds
    timeout_seconds = 180  # Increased timeout to 3 minutes
    
    # Retry loop
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} of {max_retries}")
            
            async with aiohttp.ClientSession() as session:
                print(f"aiohttp session created for attempt {attempt}")
                try:
                    print(f"Sending POST request to HF API (attempt {attempt})...")
                    print(f"Request URL: {HF_API_URL}")
                    print(f"Request timeout: {timeout_seconds} seconds")
                    
                    start_time = time.time()
                    async with session.post(HF_API_URL, headers=headers, json=payload, timeout=timeout_seconds) as response:
                        elapsed_time = time.time() - start_time
                        print(f"Received response from HF API with status: {response.status} (took {elapsed_time:.2f} seconds)")
                        
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
                            # Model is loading - this is retryable
                            error_text = await response.text()
                            print(f"HF API returned 503 (model loading): {error_text}")
                            
                            if attempt < max_retries:
                                retry_delay = retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 1)
                                print(f"Will retry in {retry_delay:.2f} seconds...")
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                raise HTTPException(status_code=503,
                                    detail="Model is loading on Hugging Face and did not become available after multiple attempts. Please try again later.")
                        
                        elif response.status == 504:
                            # Gateway timeout - this is retryable
                            error_text = await response.text()
                            print(f"HF API returned 504 (Gateway Timeout): {error_text}")
                            
                            if attempt < max_retries:
                                retry_delay = retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 1)
                                print(f"Gateway timeout occurred. Will retry with simplified parameters in {retry_delay:.2f} seconds...")
                                
                                # For subsequent attempts, reduce complexity if possible
                                if attempt > 1 and num_inference_steps > 30:
                                    payload["parameters"]["num_inference_steps"] = 30
                                    print(f"Reducing num_inference_steps to 30 for retry attempt")
                                
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                raise HTTPException(status_code=504,
                                    detail="The image generation request timed out after multiple attempts. Try reducing the image quality settings or try again later.")
                        
                        else:
                            error_text = await response.text()
                            print(f"HF API Error: Status {response.status}")
                            print(f"Response headers: {response.headers}")
                            print(f"Response body: {error_text}")
                            
                            # Determine if this error is retryable
                            retryable_status_codes = [429, 500, 502, 503, 504]
                            if response.status in retryable_status_codes and attempt < max_retries:
                                retry_delay = retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 1)
                                print(f"Retryable error occurred. Will retry in {retry_delay:.2f} seconds...")
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                raise HTTPException(status_code=response.status, detail=f"Hugging Face API error: {error_text}")
                
                except asyncio.TimeoutError:
                    print(f"Request timed out after {timeout_seconds} seconds")
                    
                    if attempt < max_retries:
                        retry_delay = retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        print(f"Timeout occurred. Will retry with simplified parameters in {retry_delay:.2f} seconds...")
                        
                        # For subsequent attempts, reduce complexity
                        if num_inference_steps > 30:
                            payload["parameters"]["num_inference_steps"] = 30
                            print(f"Reducing num_inference_steps to 30 for retry attempt")
                        
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise HTTPException(status_code=504,
                            detail="The image generation request timed out after multiple attempts. Try reducing the image quality settings or try again later.")
                
                except aiohttp.ClientError as client_error:
                    print(f"aiohttp client error: {type(client_error).__name__}: {str(client_error)}")
                    
                    if attempt < max_retries:
                        retry_delay = retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        print(f"Network error occurred. Will retry in {retry_delay:.2f} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise HTTPException(status_code=500, detail=f"Network error connecting to Hugging Face API after multiple attempts: {str(client_error)}")
        
        except Exception as e:
            if isinstance(e, HTTPException):
                raise  # Re-raise HTTP exceptions directly
            
            print(f"==== ERROR: Unexpected error in generate_with_hf_api (attempt {attempt}) ====")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            if attempt < max_retries:
                retry_delay = retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"Unexpected error occurred. Will retry in {retry_delay:.2f} seconds...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                raise HTTPException(status_code=500, detail=f"Error generating image with Hugging Face API after multiple attempts: {str(e)}")
    
    # This should not be reached due to the raise statements in the loop
    raise HTTPException(status_code=500, detail="Failed to generate image after exhausting all retry attempts")

def generate_with_local_model(prompt: str, width: int = 1024, height: int = 1024,
                             guidance_scale: float = 3.5, num_inference_steps: int = 50,
                             seed: Optional[int] = None):
    """Generate image using local model"""
    load_model()  # This will do the lazy loading
    torch = lazy_import_torch()
    
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

class LessonPlanAnalysisRequest(BaseModel):
    lesson_plan: str
    max_images: Optional[int] = 5

class ImagePrompt(BaseModel):
    prompt: str
    explanation: str

class LessonPlanAnalysisResponse(BaseModel):
    image_prompts: List[ImagePrompt]

class SVGGenerationRequest(BaseModel):
    content_type: str
    subject: str
    topic: str
    grade_level: str
    layout_style: Optional[str] = "layout1"
    num_questions: int = 5
    question_types: Optional[List[str]] = ["fill_blank", "short_answer", "multiple_choice"]
    question_type: Optional[str] = None
    image_format: Optional[str] = "landscape"
    image_aspect_ratio: Optional[dict] = {"width": 16, "height": 9}
    image_count: int = 1
    custom_instructions: Optional[str] = None
    include_activity_box: Optional[bool] = True
    include_word_bank: Optional[bool] = True  # NEW: Toggle for word bank
    max_pages: Optional[int] = 2


class SVGProcessingRequest(BaseModel):
    svg_content: str
    text_replacements: Dict[str, str]
    add_writing_lines: bool = False
    
class SVGExportRequest(BaseModel):
    svg_content: str
    format: str  # "pdf", "docx", "png"
    filename: str

class SVGTemplate(BaseModel):
    id: str
    name: str
    description: str
    content_type: str
    placeholder_count: int

class SVGGenerationResponse(BaseModel):
    svg_content: str
    template_id: str
    placeholders: List[str]
    images_generated: List[str]  # URLs to generated images

class SVGProcessingResponse(BaseModel):
    processed_svg: str
    replaced_placeholders: List[str]

class SVGExportResponse(BaseModel):
    download_url: str
    filename: str
    format: str
class SVGItem(BaseModel):
    id: str
    filename: str
    url: str
    template_id: str
    created_at: str

@app.get("/api/svg-items", response_model=List[SVGItem])
async def list_svg_items():
    """List all generated SVG items."""
    try:
        svg_metadata = await load_svg_metadata()
        # Return in reverse order (newest first)
        return [SVGItem(**item) for item in reversed(svg_metadata)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load SVG items: {str(e)}")

@app.delete("/api/svg-items/{item_id}")
async def delete_svg_item(item_id: str):
    """Delete a generated SVG item."""
    try:
        svg_metadata = await load_svg_metadata()
        item_to_delete = None
        updated_metadata = []
        
        for item in svg_metadata:
            if item["id"] == item_id:
                item_to_delete = item
            else:
                updated_metadata.append(item)
        
        if not item_to_delete:
            raise HTTPException(status_code=404, detail="SVG item not found")
        
        # Delete file
        filepath = os.path.join(SVG_EXPORTS_DIR, item_to_delete["filename"])
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Update metadata
        await save_svg_metadata(updated_metadata)
        
        return {"message": "SVG item deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete SVG item: {str(e)}")


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
        # Acquire the semaphore to ensure only one image generation happens at a time
        async with image_generation_semaphore:
            print("==== START: /api/generate endpoint called ====")
            print(f"Request data: {request.dict()}")
        
        # Generate unique filename
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        print(f"Generating image with prompt: {request.prompt}")
        print(f"Using method: {'Hugging Face API' if USE_HF_API else 'Local Model'}")
        print(f"Request parameters: width={request.width}, height={request.height}, guidance_scale={request.guidance_scale}, steps={request.num_inference_steps}, seed={request.seed}")
        
        # Check if parameters are within reasonable limits to avoid timeouts
        adjusted_request = validate_and_adjust_parameters(request)
        if adjusted_request != request:
            print(f"Parameters were adjusted to improve reliability:")
            print(f"Original: width={request.width}, height={request.height}, steps={request.num_inference_steps}")
            print(f"Adjusted: width={adjusted_request.width}, height={adjusted_request.height}, steps={adjusted_request.num_inference_steps}")
        
        # Generate image based on configuration
        if USE_HF_API:
            print(f"HF API Token configured: {bool(HF_API_TOKEN)}")
            print(f"HF API URL: {HF_API_URL}")
            try:
                image = await generate_with_hf_api(
                    adjusted_request.prompt,
                    adjusted_request.width,
                    adjusted_request.height,
                    adjusted_request.guidance_scale,
                    adjusted_request.num_inference_steps,
                    adjusted_request.seed
                )
                print("Image successfully generated from HF API")
                generation_method = "hf_api"
            except Exception as hf_error:
                print(f"Detailed HF API error: {type(hf_error).__name__}: {str(hf_error)}")
                if isinstance(hf_error, HTTPException):
                    print(f"HTTP Exception status: {hf_error.status_code}, detail: {hf_error.detail}")
                    
                    # Provide more user-friendly error messages
                    if hf_error.status_code == 504:
                        raise HTTPException(
                            status_code=504,
                            detail="The image generation request timed out. This can happen with complex images. Try reducing the image size or quality settings, or try again later."
                        )
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
        
        # Save image and ensure it's fully written to disk
        try:
            print(f"Saving image to {filepath}")
            image.save(filepath)
            
            # Verify the image was saved correctly by opening it again
            try:
                # Flush any file system buffers to ensure the file is fully written
                os.fsync(os.open(filepath, os.O_RDONLY))
                
                # Attempt to open the saved image to verify it's valid
                Image, _ = lazy_import_image_processing()
                verification_image = Image.open(filepath)
                verification_image.verify()  # Verify the image data
                print(f"Image verified successfully: {verification_image.format}, {verification_image.size}")
            except Exception as verify_error:
                print(f"Warning: Image verification failed: {type(verify_error).__name__}: {str(verify_error)}")
                # If verification fails, we'll still continue but log the warning
            
            print("Image saved successfully")
        except Exception as save_error:
            print(f"Error saving image: {type(save_error).__name__}: {str(save_error)}")
            raise
        
        # Create metadata
        try:
            print("Loading and updating metadata")
            metadata = await load_metadata()
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
            await save_metadata(metadata)
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
        # Provide more user-friendly error messages based on error type
        if error_type == "TimeoutError" or "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail="The image generation request timed out. This can happen with complex images. Try reducing the image size or quality settings, or try again later."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Failed to generate image: {error_type}: {error_msg}")
    finally:
        print("==== END: /api/generate endpoint processing ====")

@app.get("/api/images", response_model=List[ImageResponse])
async def list_images():
    try:
        metadata = await load_metadata()
        # Return in reverse order (newest first)
        return [ImageResponse(**img) for img in reversed(metadata)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load images: {str(e)}")

@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    try:
        metadata = await load_metadata()
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
        filepath = os.path.join(IMAGES_DIR, image_to_delete["filename"])
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Update metadata
        await save_metadata(updated_metadata)
        
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

def check_nltk_resources():
    """Check if all required NLTK resources are available."""
    nltk = lazy_import_nltk()
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for resource_path, resource_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.error(f"NLTK resource '{resource_name}' not found.")
            logger.error("Please run the setup script to download the required resources:")
            logger.error("python backend/setup.py")
            # Exit the application if resources are missing
            exit(1)

# Check for NLTK resources on startup
#check_nltk_resources()

class AIService:
    """Service for interacting with AI models via OpenRouter."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the AI service with API key."""
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
            
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.default_model = "meta-llama/llama-4-maverick:free"
    
    def query_model(self, prompt: str, system_message: Optional[str] = None) -> Optional[str]:
        """Query the AI model with a text prompt."""
        logger.info(f"Generating response using OpenRouter...")
        
        default_system_message = (
            "You are an expert educational content creator specializing in lesson planning. "
            "You provide clear, structured, and detailed lesson plan components. "
            "For all educational queries, ensure accuracy, completeness, and relevance. "
            "Include specific teaching objectives, materials, procedures, and assessment strategies."
        )
        
        try:
            response = requests.post(
                url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Lesson Plan Generator",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": self.default_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message or default_system_message
                        },
                        {
                            "role": "user",
                            "content": prompt.strip() if prompt else ""
                        }
                    ]
                })
            )
            
            response_data = response.json()
            
            if response.status_code == 200 and "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Received response from OpenRouter (first 100 chars): {content[:100]}...")
                print("Model Response:", content)
                return content
            else:
                error_message = response_data.get("error", {}).get("message", "Unknown API error")
                logger.error(f"API Error: {error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying model: {e}")
            return None

class SVGService:
    """Service for handling SVG template operations and processing."""
    
    def __init__(self):
        """Initialize the SVG service."""
        self.templates_dir = SVG_TEMPLATES_DIR
        self.exports_dir = SVG_EXPORTS_DIR
        
        # Template metadata
        self.template_metadata = {
            "image_comprehension": {
                "id": "image_comprehension",
                "name": "Image Comprehension Worksheet",
                "description": "Template for image analysis and comprehension questions",
                "content_type": "image_comprehension",
                "placeholder_count": 7,
                "placeholders": ["title", "subject", "grade", "image", "question1", "question2", "question3", "instructions"],
                "minImages": 1,
                "maxImages": 1
            },
            "comic": {
                "id": "comic",
                "name": "Comic Strip Template",
                "description": "Four-panel comic strip template with speech bubbles",
                "content_type": "comic",
                "placeholder_count": 10,
                "placeholders": ["title", "subject", "grade", "image1", "speech1", "image2", "speech2", "image3", "speech3", "image4", "speech4"],
                "minImages": 2,
                "maxImages": 4
            },
            "math": {
                "id": "math",
                "name": "Math Worksheet",
                "description": "Template for math problems with visual aids",
                "content_type": "math",
                "placeholder_count": 9,
                "placeholders": ["title", "subject", "grade", "instructions", "image", "problem1", "problem2", "problem3", "problem4", "problem5", "problem6"],
                "minImages": 1,
                "maxImages": 1
            },
            "worksheet": {
                "id": "worksheet",
                "name": "General Worksheet Template",
                "description": "Flexible worksheet template with multiple sections",
                "content_type": "worksheet",
                "placeholder_count": 12,
                "placeholders": ["title", "subject", "grade", "section1_title", "image1", "text1", "section2_title", "image2", "text2", "section3_title", "activity1", "activity2", "activity3", "activity4"],
                "minImages": 1,
                "maxImages": 4
            }
        }
    
    def get_available_templates(self) -> List[SVGTemplate]:
        """Get list of available SVG templates."""
        templates = []
        for template_id, metadata in self.template_metadata.items():
            templates.append(SVGTemplate(
                id=metadata["id"],
                name=metadata["name"],
                description=metadata["description"],
                content_type=metadata["content_type"],
                placeholder_count=metadata["placeholder_count"]
            ))
        return templates
    
    def load_template(self, template_id: str) -> str:
        """Load SVG template content."""
        template_path = os.path.join(self.templates_dir, f"{template_id}.svg")
        if not os.path.exists(template_path):
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading template: {str(e)}")
    
    def extract_placeholders(self, svg_content: str) -> List[str]:
        """Extract placeholder IDs from SVG content."""
        placeholders = []
        try:
            # Find bracket-style placeholders: [placeholder]
            import re
            bracket_matches = re.findall(r'\[([^\]]+)\]', svg_content)
            placeholders.extend(bracket_matches)
            
            # Also find curly brace placeholders: {placeholder}
            brace_matches = re.findall(r'\{([^}]+)\}', svg_content)
            placeholders.extend(brace_matches)
            
            # Remove duplicates while preserving order
            unique_placeholders = []
            for placeholder in placeholders:
                if placeholder not in unique_placeholders:
                    unique_placeholders.append(placeholder)
            
            logger.info(f"Extracted {len(unique_placeholders)} placeholders: {unique_placeholders}")
            return unique_placeholders
            
        except Exception as e:
            logger.error(f"Error extracting placeholders: {e}")
            return []
    
    def replace_text_placeholders(self, svg_content: str, replacements: Dict[str, str]) -> str:
        """Replace text placeholders in SVG content."""
        try:
            processed_content = svg_content
            
            # Replace bracket-style placeholders: [placeholder] -> replacement
            for placeholder, replacement in replacements.items():
                # Handle both [placeholder] and placeholder patterns
                patterns = [
                    f'[{placeholder}]',
                    f'{{{placeholder}}}',  # Also handle curly braces
                    f'placeholder_{placeholder}',  # Handle ID-style placeholders
                ]
                
                for pattern in patterns:
                    processed_content = processed_content.replace(pattern, replacement)
            
            logger.info(f"Text replacement completed for {len(replacements)} placeholders")
            return processed_content
            
        except Exception as e:
            logger.error(f"Error replacing text placeholders: {e}")
            return svg_content
        
    async def embed_images_in_svg(self, svg_content: str, image_urls: List[str]) -> str:
        """Embed generated images into SVG placeholders."""
        ET, _, _, _ = lazy_import_document_processing()
        try:
            root = ET.fromstring(svg_content)
            image_index = 0
            elements_to_remove = []  # Track elements to remove
            
            # Look for image placeholder elements by ID
            for elem in root.iter():
                if (elem.get('id') == 'placeholder_image' and image_index < len(image_urls)):
                    # Get the image file path
                    image_path = os.path.join(IMAGES_DIR, os.path.basename(image_urls[image_index]))
                    if os.path.exists(image_path):
                        with open(image_path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()
                        
                        # Create image element to replace the placeholder
                        img_elem = ET.Element('image')
                        
                        # Use the same position and size as the placeholder rect
                        img_elem.set('x', elem.get('x', '200'))
                        img_elem.set('y', elem.get('y', '90'))
                        img_elem.set('width', elem.get('width', '400'))
                        img_elem.set('height', elem.get('height', '300'))
                        img_elem.set('href', f'data:image/png;base64,{img_data}')
                        img_elem.set('preserveAspectRatio', 'xMidYMid meet')
                        
                        # Find parent element and replace
                        parent = elem.getparent() if hasattr(elem, 'getparent') else None
                        if parent is not None:
                            parent_index = list(parent).index(elem)
                            parent.remove(elem)
                            parent.insert(parent_index, img_elem)
                        else:
                            # If no parent found, replace in root
                            root_index = list(root).index(elem)
                            root.remove(elem)
                            root.insert(root_index, img_elem)
                        
                        image_index += 1
                        logger.info(f"Embedded image {image_index}: {image_urls[image_index-1]}")
                
                # Also look for placeholder text elements and mark them for removal
                elif (elem.tag == 'text' and 
                    elem.text and 
                    ('Image will appear here' in elem.text or 
                    '[IMAGE HERE]' in elem.text or
                    'image will appear here' in elem.text.lower())):
                    elements_to_remove.append(elem)
            
            # Remove placeholder text elements
            for elem in elements_to_remove:
                parent = elem.getparent() if hasattr(elem, 'getparent') else None
                if parent is not None:
                    parent.remove(elem)
                elif elem in root:
                    root.remove(elem)
            
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            logger.error(f"Error embedding images in SVG: {e}")
            # Return original content if embedding fails
            return svg_content
    
    def _detect_and_split_svg_pages(self, svg_content: str) -> List[str]:
        """
        Detect if SVG is multi-page and split into individual A4 pages.
        Returns list of individual page SVG strings.
        """
        try:
            # Check if this is a multi-page SVG
            single_page_height = 1123  # A4 height
            
            # Extract height from SVG
            height_match = re.search(r'height="(\d+)"', svg_content)
            if not height_match:
                return [svg_content]  # Can't determine height, return as single page
            
            total_height = int(height_match.group(1))
            
            # If height is close to single page, treat as single page
            if total_height <= single_page_height + 100:  # Allow some tolerance
                return [svg_content]
            
            # Multi-page detected - split into pages
            pages = []
            
            # Calculate number of pages (assuming 50px spacing between pages)
            page_spacing = 50
            estimated_pages = int((total_height + page_spacing) / (single_page_height + page_spacing))
            
            for page_num in range(estimated_pages):
                page_y_offset = page_num * (single_page_height + page_spacing)
                page_svg = self._extract_svg_page(svg_content, page_y_offset, single_page_height)
                if page_svg:
                    pages.append(page_svg)
            
            return pages if pages else [svg_content]
            
        except Exception as e:
            logger.error(f"Error splitting SVG pages: {e}")
            return [svg_content]  # Return original on error

    def _extract_svg_page(self, svg_content: str, y_offset: int, page_height: int) -> str:
        """
        Extract a single page from multi-page SVG by adjusting viewBox and clipping content.
        """
        try:
            # Create a new SVG for this page with A4 dimensions
            page_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="794" height="{page_height}" viewBox="0 {y_offset} 794 {page_height}">
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #2c3e50; }}
      .subtitle {{ font-family: Arial, sans-serif; font-size: 12px; fill: #34495e; }}
      .small {{ font-family: Arial, sans-serif; font-size: 10px; fill: #2c3e50; }}
      .instruction {{ font-family: Arial, sans-serif; font-size: 11px; fill: #2c3e50; font-weight: bold; }}
      .question {{ font-family: Arial, sans-serif; font-size: 11px; fill: #2c3e50; }}
      .placeholder {{ font-family: Arial, sans-serif; font-size: 12px; fill: #7f8c8d; }}
      @media print {{
        .page-break {{ page-break-before: always; }}
      }}
    </style>
  </defs>
  
  <!-- Page background -->
  <rect x="0" y="{y_offset}" width="794" height="{page_height}" fill="#ffffff" stroke="none"/>
'''
            
            # Extract content elements that fall within this page's Y range
            page_end_y = y_offset + page_height
            
            # Parse the original SVG to extract elements
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(svg_content)
                
                # Extract all elements that should appear on this page
                for elem in root.iter():
                    if elem.tag in ['rect', 'text', 'foreignObject', 'line', 'circle', 'ellipse', 'image']:
                        elem_y = self._get_element_y_position(elem)
                        if elem_y is not None and y_offset <= elem_y < page_end_y:
                            # Add this element to the page
                            elem_svg = ET.tostring(elem, encoding='unicode')
                            page_svg += f"\n  {elem_svg}"
                
            except Exception as parse_error:
                logger.warning(f"Could not parse SVG for page extraction: {parse_error}")
                # Fallback: use simple clipping
                page_svg += f'\n  <g clip-path="url(#page-clip)">'
                # Extract content between defs and closing svg
                content_match = re.search(r'</defs>(.*)</svg>', svg_content, re.DOTALL)
                if content_match:
                    page_svg += content_match.group(1)
                page_svg += '\n  </g>'
            
            page_svg += '\n</svg>'
            return page_svg
            
        except Exception as e:
            logger.error(f"Error extracting SVG page: {e}")
            return None

    def _get_element_y_position(self, element) -> int:
        """Extract Y position from SVG element."""
        try:
            if element.get('y'):
                return int(float(element.get('y')))
            elif element.get('cy'):  # For circles
                return int(float(element.get('cy')))
            elif element.get('y1'):  # For lines
                return int(float(element.get('y1')))
            return None
        except (ValueError, TypeError):
            return None

    def export_svg_to_pdf(self, svg_content: str, filename: str) -> str:
        """Export SVG to PDF format with proper A4 sizing and multi-page support."""
        _, cairosvg = lazy_import_image_processing()
        try:
            # Split into pages
            pages = self._detect_and_split_svg_pages(svg_content)
            
            if len(pages) == 1:
                # Single page export
                output_path = os.path.join(self.exports_dir, f"{filename}.pdf")
                clean_svg = self._clean_svg_for_export(pages[0])
                
                cairosvg.svg2pdf(
                    bytestring=clean_svg.encode('utf-8'),
                    write_to=output_path,
                    output_width=794,   # A4 width in pixels at 96 DPI
                    output_height=1123  # A4 height in pixels at 96 DPI
                )
                return output_path
            else:
                # Multi-page export using reportlab
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import A4
                import tempfile
                
                output_path = os.path.join(self.exports_dir, f"{filename}.pdf")
                
                # Create PDF with multiple pages
                pdf_canvas = canvas.Canvas(output_path, pagesize=A4)
                
                for i, page_svg in enumerate(pages):
                    # Convert each page to PNG first, then add to PDF
                    clean_svg = self._clean_svg_for_export(page_svg)
                    
                    # Create temporary PNG
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                        cairosvg.svg2png(
                            bytestring=clean_svg.encode('utf-8'),
                            write_to=temp_file.name,
                            dpi=150,
                            output_width=1191,   # A4 width at 150 DPI
                            output_height=1684   # A4 height at 150 DPI
                        )
                        
                        # Add image to PDF page
                        pdf_canvas.drawImage(temp_file.name, 0, 0, width=A4[0], height=A4[1])
                        
                        # Add new page if not last page
                        if i < len(pages) - 1:
                            pdf_canvas.showPage()
                        
                        # Clean up temp file
                        os.unlink(temp_file.name)
                
                pdf_canvas.save()
                return output_path
                
        except Exception as e:
            logger.error(f"Error exporting SVG to PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Error exporting to PDF: {str(e)}")

    def export_svg_to_png(self, svg_content: str, filename: str) -> str:
        """Export SVG to PNG format with high quality A4 dimensions and multi-page support."""
        _, cairosvg = lazy_import_image_processing()
        try:
            # Split into pages
            pages = self._detect_and_split_svg_pages(svg_content)
            
            if len(pages) == 1:
                # Single page export
                output_path = os.path.join(self.exports_dir, f"{filename}.png")
                clean_svg = self._clean_svg_for_export(pages[0])
                
                cairosvg.svg2png(
                    bytestring=clean_svg.encode('utf-8'),
                    write_to=output_path,
                    dpi=300,
                    output_width=2384,   # A4 width at 300 DPI
                    output_height=3369   # A4 height at 300 DPI
                )
                return output_path
            else:
                # Multi-page export - create separate PNG files
                from PIL import Image
                
                page_images = []
                temp_files = []
                
                # Create PNG for each page
                for i, page_svg in enumerate(pages):
                    clean_svg = self._clean_svg_for_export(page_svg)
                    
                    # Create temporary file for this page
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_page_{i+1}.png')
                    temp_files.append(temp_file.name)
                    
                    cairosvg.svg2png(
                        bytestring=clean_svg.encode('utf-8'),
                        write_to=temp_file.name,
                        dpi=300,
                        output_width=2384,   # A4 width at 300 DPI
                        output_height=3369   # A4 height at 300 DPI
                    )
                    
                    # Load the image
                    page_img = Image.open(temp_file.name)
                    page_images.append(page_img)
                    temp_file.close()
                
                # Combine pages vertically into one tall image
                total_height = sum(img.height for img in page_images)
                combined_width = page_images[0].width
                
                combined_image = Image.new('RGB', (combined_width, total_height), 'white')
                
                y_offset = 0
                for img in page_images:
                    combined_image.paste(img, (0, y_offset))
                    y_offset += img.height
                
                # Save combined image
                output_path = os.path.join(self.exports_dir, f"{filename}.png")
                combined_image.save(output_path, 'PNG', dpi=(300, 300))
                
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                
                return output_path
                
        except Exception as e:
            logger.error(f"Error exporting SVG to PNG: {e}")
            raise HTTPException(status_code=500, detail=f"Error exporting to PNG: {str(e)}")

    def export_svg_to_docx(self, svg_content: str, filename: str) -> str:
        """Export SVG to DOCX format with proper A4 page setup and multi-page support."""
        _, cairosvg = lazy_import_image_processing()
        _, Document, Inches, tempfile = lazy_import_document_processing()
        try:
            # Split into pages
            pages = self._detect_and_split_svg_pages(svg_content)
            
            # Create DOCX document with A4 page setup
            doc = Document()
            
            # Set A4 page size and margins
            section = doc.sections[0]
            section.page_width = Inches(8.27)   # A4 width: 210mm
            section.page_height = Inches(11.69)  # A4 height: 297mm
            section.left_margin = Inches(0.5)
            section.right_margin = Inches(0.5)
            section.top_margin = Inches(0.5)
            section.bottom_margin = Inches(0.5)
            
            # Process each page
            for i, page_svg in enumerate(pages):
                # Convert page to PNG
                clean_svg = self._clean_svg_for_export(page_svg)
                png_data = cairosvg.svg2png(
                    bytestring=clean_svg.encode('utf-8'),
                    dpi=300,
                    output_width=2384,   # A4 width at 300 DPI
                    output_height=3369   # A4 height at 300 DPI
                )
                
                # Add the image to document
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_file.write(png_data)
                    temp_file.flush()
                    
                    # Add to document with A4-appropriate size
                    doc.add_picture(temp_file.name, width=Inches(7.27))  # Width minus margins
                    
                    # Add page break if not last page
                    if i < len(pages) - 1:
                        doc.add_page_break()
                    
                    # Clean up temp file
                    os.unlink(temp_file.name)
            
            output_path = os.path.join(self.exports_dir, f"{filename}.docx")
            doc.save(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting SVG to DOCX: {e}")
            raise HTTPException(status_code=500, detail=f"Error exporting to DOCX: {str(e)}")

    def _clean_svg_for_export(self, svg_content: str) -> str:
        """Clean SVG content for better export compatibility with A4 format."""
        try:
            # Remove any XML declarations
            content = re.sub(r'<\?xml[^>]*\?>', '', svg_content)
            
            # Ensure proper SVG namespace
            if 'xmlns="http://www.w3.org/2000/svg"' not in content:
                content = content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
            
            # Clean up namespace prefixes
            content = re.sub(r'ns\d+:', '', content)
            content = re.sub(r'xmlns:ns\d+="[^"]*"', '', content)
            
            # Ensure proper A4 viewBox and dimensions
            if 'viewBox=' not in content:
                content = content.replace('<svg', '<svg viewBox="0 0 794 1123"')
            
            # Ensure width and height are set for A4
            content = re.sub(r'width="[^"]*"', 'width="794"', content)
            content = re.sub(r'height="[^"]*"', 'height="1123"', content)
            
            # Add CSS for page breaks when printed
            css_styles = '''
            <style>
            @media print {
                @page {
                    size: A4;
                    margin: 0.5in;
                }
                .page-break {
                    page-break-before: always;
                }
            }
            </style>
            '''
            
            # Insert CSS after the opening svg tag
            content = content.replace('<svg', css_styles + '<svg')
            
            return content.strip()
        except Exception as e:
            logger.error(f"Error cleaning SVG: {e}")
            return svg_content

# Initialize SVG service
svg_service = SVGService()

def generate_dynamic_svg_template(content_type: str, layout_style: str, num_questions: int,
                                 subject: str, grade_level: str, topic: str,
                                 include_activity_box: bool = True) -> str:
    """
    Generate a dynamic SVG template with overflow handling.
    """
    
    if content_type == "image_comprehension":
        return generate_dynamic_image_comprehension_template(
            layout_style, num_questions, subject, grade_level, topic, include_activity_box
        )
    elif content_type == "math":
        return generate_dynamic_math_template(num_questions, subject, grade_level, topic)
    elif content_type == "comic":
        return generate_dynamic_comic_template(num_questions, subject, grade_level, topic)
    elif content_type == "worksheet":
        return generate_dynamic_worksheet_template(num_questions, subject, grade_level, topic)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def generate_dynamic_image_comprehension_template(layout_style: str, num_questions: int,
                                                subject: str, grade_level: str, topic: str,
                                                include_activity_box: bool = True) -> str:
    """
    Generate a dynamic image comprehension template with overflow handling.
    """
    
    if layout_style == "layout1":
        return generate_layout1_template(num_questions, subject, grade_level, topic, include_activity_box)
    elif layout_style == "layout2":
        return generate_layout2_template(num_questions, subject, grade_level, topic)
    elif layout_style == "layout3":
        return generate_layout3_template(num_questions, subject, grade_level, topic)
    else:
        return generate_layout1_template(num_questions, subject, grade_level, topic, include_activity_box)
    
    
def extract_fill_in_blank_answers(questions: List[str]) -> List[str]:
    """
    Extract answer words from fill-in-the-blank questions.
    This function analyzes the context to suggest appropriate answer words.
    """
    answers = []
    
    for question in questions:
        if '_____' in question:
            # Basic context analysis to suggest answers
            # In a real implementation, you'd want more sophisticated NLP
            
            # Common patterns for different subjects
            if 'water' in question.lower():
                if 'process' in question.lower():
                    answers.extend(['evaporation', 'condensation'])
                elif 'cycle' in question.lower():
                    answers.extend(['precipitation', 'collection'])
                elif 'vapor' in question.lower():
                    answers.extend(['condenses', 'rises'])
            
            elif 'plant' in question.lower() or 'leaf' in question.lower():
                answers.extend(['photosynthesis', 'chlorophyll', 'oxygen', 'carbon dioxide'])
            
            elif 'animal' in question.lower():
                answers.extend(['habitat', 'adaptation', 'vertebrate', 'invertebrate'])
            
            elif 'math' in question.lower() or any(word in question.lower() for word in ['add', 'subtract', 'multiply', 'divide']):
                answers.extend(['sum', 'difference', 'product', 'quotient'])
            
            # Extract meaningful words from the question context
            words = question.replace('_____', '').split()
            content_words = [word.strip('.,!?()').lower() for word in words
                           if len(word) > 3 and word.isalpha() and
                           word.lower() not in ['the', 'and', 'for', 'are', 'this', 'that', 'with', 'from']]
            
            # Add context-relevant words
            answers.extend(content_words[:2])  # Take up to 2 relevant words per question
    
    # Remove duplicates and limit to reasonable number
    unique_answers = list(dict.fromkeys(answers))  # Preserves order while removing duplicates
    return unique_answers[:10]  # Max 10 words for word bank


def generate_layout1_template_with_wordbank(num_questions: int, subject: str, grade_level: str, topic: str,
                                          include_activity_box: bool = True, include_word_bank: bool = True) -> str:
    """
    Generate Layout 1 template with proper multi-page overflow handling.
    """
    
    # A4 dimensions with proper margins (50px on all sides)
    svg_width = 794
    svg_height = 1123
    margin = 50
    content_width = svg_width - (2 * margin)
    usable_height = svg_height - (2 * margin)
    
    # Calculate heights for different sections
    header_height = 40
    topic_height = 25
    image_height = 280
    instructions_height = 60
    question_height = 35
    word_bank_height = 80 if include_word_bank else 0
    activity_height = 120 if include_activity_box else 0
    footer_height = 25
    spacing = 15
    
    # Calculate Y positions for page 1
    header_y = margin
    topic_y = header_y + header_height + spacing
    image_y = topic_y + topic_height + spacing
    instructions_y = image_y + image_height + spacing
    questions_start_y = instructions_y + instructions_height + spacing
    
    # Calculate how many questions fit on page 1
    available_space_page1 = svg_height - questions_start_y - footer_height - margin - spacing
    
    # Reserve space for word bank and activity if they fit
    word_bank_space = word_bank_height + spacing if include_word_bank else 0
    activity_space = activity_height + spacing if include_activity_box else 0
    
    # Determine what fits on page 1
    space_for_questions_page1 = available_space_page1 - word_bank_space - activity_space
    questions_per_page1 = max(1, int(space_for_questions_page1 // question_height))
    
    # Check if we need page 2
    remaining_questions = max(0, num_questions - questions_per_page1)
    needs_second_page = remaining_questions > 0 or (include_activity_box and activity_space > space_for_questions_page1)
    
    # Calculate total SVG height
    if needs_second_page:
        total_svg_height = svg_height * 2 + 50  # Two A4 pages plus spacing
    else:
        total_svg_height = svg_height
    
    # Generate questions for page 1
    page1_questions_svg = ""
    for i in range(1, min(questions_per_page1 + 1, num_questions + 1)):
        question_y = questions_start_y + ((i - 1) * question_height)
        page1_questions_svg += f'''
  <foreignObject x="{margin + 20}" y="{question_y}" width="{content_width - 40}" height="{question_height - 5}">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 11px; color: #2c3e50; line-height: 1.4; word-wrap: break-word; margin: 0; padding: 2px 0;">
      {i}. [question{i}]
    </div>
  </foreignObject>
'''
    
    # Calculate positions for page 1 elements
    page1_questions_end_y = questions_start_y + (questions_per_page1 * question_height)
    word_bank_y = page1_questions_end_y + spacing
    
    # Word bank for page 1
    word_bank_svg = ""
    if include_word_bank and not needs_second_page:
        word_bank_svg = f'''
  <!-- Word Bank Section -->
  <rect x="{margin + 20}" y="{word_bank_y}" width="{content_width - 40}" height="{word_bank_height}" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
  <text x="{margin + 30}" y="{word_bank_y + 20}" class="instruction">Word Bank:</text>
  <text x="{margin + 30}" y="{word_bank_y + 35}" class="small">[word_bank_word1]</text>
  <text x="{margin + 130}" y="{word_bank_y + 35}" class="small">[word_bank_word2]</text>
  <text x="{margin + 230}" y="{word_bank_y + 35}" class="small">[word_bank_word3]</text>
  <text x="{margin + 330}" y="{word_bank_y + 35}" class="small">[word_bank_word4]</text>
  <text x="{margin + 430}" y="{word_bank_y + 35}" class="small">[word_bank_word5]</text>
  <text x="{margin + 30}" y="{word_bank_y + 55}" class="small">[word_bank_word6]</text>
  <text x="{margin + 130}" y="{word_bank_y + 55}" class="small">[word_bank_word7]</text>
  <text x="{margin + 230}" y="{word_bank_y + 55}" class="small">[word_bank_word8]</text>
  <text x="{margin + 330}" y="{word_bank_y + 55}" class="small">[word_bank_word9]</text>
  <text x="{margin + 430}" y="{word_bank_y + 55}" class="small">[word_bank_word10]</text>
'''
    
    # Activity for page 1 (only if it fits)
    activity_y = word_bank_y + (word_bank_height + spacing if include_word_bank and not needs_second_page else 0)
    activity_svg = ""
    if include_activity_box and not needs_second_page:
        activity_svg = f'''
  <!-- Activity Section -->
  <text x="{margin + 20}" y="{activity_y}" class="instruction">Activity:</text>
  <text x="{margin + 20}" y="{activity_y + 20}" class="small">Draw or write about what you learned from this image:</text>
  <rect x="{margin + 20}" y="{activity_y + 30}" width="{content_width - 40}" height="{activity_height - 30}" fill="none" stroke="#bdc3c7" stroke-width="1" stroke-dasharray="3,3"/>
'''
    
    # Generate Page 2 content if needed
    page2_svg = ""
    if needs_second_page:
        page2_y_start = svg_height + 50 + margin  # Start of page 2
        
        # Page 2 header
        page2_header = f'''
  
  <!-- Page 2 Content -->
  <!-- Page 2 Header -->
  <rect x="{margin}" y="{page2_y_start}" width="{content_width}" height="{header_height}" fill="#ffffff" stroke="#2c3e50" stroke-width="1"/>
  <text x="{margin + 20}" y="{page2_y_start + 22}" class="title">[subject] - Grade [grade] (continued)</text>
  <text x="{margin + 20}" y="{page2_y_start + 35}" class="small">Name: _______________</text>
  <text x="{margin + 450}" y="{page2_y_start + 35}" class="small">Date: ______</text>
  
  <!-- Page 2 Topic -->
  <text x="{svg_width // 2}" y="{page2_y_start + header_height + spacing + 15}" class="instruction" text-anchor="middle">Topic: [topic] (continued)</text>
'''
        
        # Generate remaining questions for page 2
        page2_questions_svg = ""
        page2_questions_start_y = page2_y_start + header_height + spacing + topic_height + spacing
        
        for i in range(questions_per_page1 + 1, num_questions + 1):
            question_index_on_page2 = i - questions_per_page1 - 1
            question_y = page2_questions_start_y + (question_index_on_page2 * question_height)
            page2_questions_svg += f'''
  <foreignObject x="{margin + 20}" y="{question_y}" width="{content_width - 40}" height="{question_height - 5}">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 11px; color: #2c3e50; line-height: 1.4; word-wrap: break-word; margin: 0; padding: 2px 0;">
      {i}. [question{i}]
    </div>
  </foreignObject>
'''
        
        # Word bank on page 2 (if enabled)
        page2_word_bank_svg = ""
        if include_word_bank:
            page2_word_bank_y = page2_questions_start_y + (remaining_questions * question_height) + spacing
            page2_word_bank_svg = f'''
  <!-- Page 2 Word Bank Section -->
  <rect x="{margin + 20}" y="{page2_word_bank_y}" width="{content_width - 40}" height="{word_bank_height}" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
  <text x="{margin + 30}" y="{page2_word_bank_y + 20}" class="instruction">Word Bank:</text>
  <text x="{margin + 30}" y="{page2_word_bank_y + 35}" class="small">[word_bank_word1]</text>
  <text x="{margin + 130}" y="{page2_word_bank_y + 35}" class="small">[word_bank_word2]</text>
  <text x="{margin + 230}" y="{page2_word_bank_y + 35}" class="small">[word_bank_word3]</text>
  <text x="{margin + 330}" y="{page2_word_bank_y + 35}" class="small">[word_bank_word4]</text>
  <text x="{margin + 430}" y="{page2_word_bank_y + 35}" class="small">[word_bank_word5]</text>
  <text x="{margin + 30}" y="{page2_word_bank_y + 55}" class="small">[word_bank_word6]</text>
  <text x="{margin + 130}" y="{page2_word_bank_y + 55}" class="small">[word_bank_word7]</text>
  <text x="{margin + 230}" y="{page2_word_bank_y + 55}" class="small">[word_bank_word8]</text>
  <text x="{margin + 330}" y="{page2_word_bank_y + 55}" class="small">[word_bank_word9]</text>
  <text x="{margin + 430}" y="{page2_word_bank_y + 55}" class="small">[word_bank_word10]</text>
'''
        
        # Activity box on page 2 (if enabled)
        page2_activity_svg = ""
        if include_activity_box:
            page2_activity_y = page2_word_bank_y + (word_bank_height + spacing if include_word_bank else 0)
            page2_activity_svg = f'''
  <!-- Page 2 Activity Section -->
  <text x="{margin + 20}" y="{page2_activity_y}" class="instruction">Activity:</text>
  <text x="{margin + 20}" y="{page2_activity_y + 20}" class="small">Draw or write about what you learned from this image:</text>
  <rect x="{margin + 20}" y="{page2_activity_y + 30}" width="{content_width - 40}" height="{activity_height - 30}" fill="none" stroke="#bdc3c7" stroke-width="1" stroke-dasharray="3,3"/>
  
  <!-- Additional space for more activities -->
  <text x="{margin + 20}" y="{page2_activity_y + activity_height + 20}" class="instruction">Additional Questions or Notes:</text>
  <rect x="{margin + 20}" y="{page2_activity_y + activity_height + 40}" width="{content_width - 40}" height="100" fill="none" stroke="#bdc3c7" stroke-width="1" stroke-dasharray="3,3"/>
'''
        
        # Page 2 footer
        page2_footer = f'''
  
  <!-- Page 2 Footer -->
  <rect x="{margin}" y="{page2_y_start + svg_height - 2*margin - footer_height}" width="{content_width}" height="{footer_height}" fill="#f8f9fa"/>
  <text x="{svg_width // 2}" y="{page2_y_start + svg_height - 2*margin - 10}" class="small" text-anchor="middle">Page 2</text>
'''
        
        # Combine all page 2 content
        page2_svg = page2_header + page2_questions_svg + page2_word_bank_svg + page2_activity_svg + page2_footer
    
    # Determine page 1 footer text
    page1_footer_text = "Page 1 of 2" if needs_second_page else "Page 1"
    
    # Generate the complete SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{total_svg_height}" viewBox="0 0 {svg_width} {total_svg_height}">
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #2c3e50; }}
      .subtitle {{ font-family: Arial, sans-serif; font-size: 12px; fill: #34495e; }}
      .small {{ font-family: Arial, sans-serif; font-size: 10px; fill: #2c3e50; }}
      .instruction {{ font-family: Arial, sans-serif; font-size: 11px; fill: #2c3e50; font-weight: bold; }}
      .question {{ font-family: Arial, sans-serif; font-size: 11px; fill: #2c3e50; }}
      .placeholder {{ font-family: Arial, sans-serif; font-size: 12px; fill: #7f8c8d; }}
      @media print {{
        .page-break {{ page-break-before: always; }}
      }}
    </style>
  </defs>
  
  <!-- A4 Background Page 1 -->
  <rect width="{svg_width}" height="{svg_height}" fill="#ffffff" stroke="none"/>
  
  <!-- Header section -->
  <rect x="{margin}" y="{header_y}" width="{content_width}" height="{header_height}" fill="#ffffff" stroke="#2c3e50" stroke-width="1"/>
  <text x="{margin + 20}" y="{header_y + 22}" class="title">[subject] - Grade [grade]</text>
  <text x="{margin + 20}" y="{header_y + 35}" class="small">Name: _______________</text>
  <text x="{margin + 450}" y="{header_y + 35}" class="small">Date: ______</text>
  
  <!-- Topic -->
  <text x="{svg_width // 2}" y="{topic_y + 15}" class="instruction" text-anchor="middle">Topic: [topic]</text>
  
  <!-- Image section -->
  <rect x="{(svg_width - 400) // 2}" y="{image_y}" width="400" height="{image_height}" fill="#e8f4fd" stroke="#bdc3c7" stroke-width="2" stroke-dasharray="5,5" id="placeholder_image"/>
  
  <!-- Instructions -->
  <foreignObject x="{margin + 20}" y="{instructions_y}" width="{content_width - 40}" height="{instructions_height}">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 11px; color: #2c3e50; font-weight: bold; line-height: 1.4; word-wrap: break-word; margin: 0; padding: 5px 0;">
      [instructions]
    </div>
  </foreignObject>
  
  <!-- Page 1 Questions section -->
{page1_questions_svg}
  
{word_bank_svg}
  
{activity_svg}
  
  <!-- Page 1 Footer -->
  <rect x="{margin}" y="{svg_height - margin - footer_height}" width="{content_width}" height="{footer_height}" fill="#f8f9fa"/>
  <text x="{svg_width // 2}" y="{svg_height - margin - 10}" class="small" text-anchor="middle">{page1_footer_text}</text>
{page2_svg}
</svg>'''
    
    return svg_content

def generate_layout2_template(num_questions: int, subject: str, grade_level: str, topic: str) -> str:
    """
    Generate Layout 2 template with dynamic number of questions (Reading comprehension style).
    """
    
    # A4 dimensions with proper margins
    svg_width = 794
    svg_height = 1123
    margin = 50
    content_width = svg_width - (2 * margin)
    
    # Calculate heights and positions
    header_height = 35
    topic_height = 25
    reading_section_height = 180
    instructions_height = 40
    question_height = 45  # Height per Q&A pair
    extra_space_height = 120
    footer_height = 25
    
    # Y positions
    header_y = margin
    topic_y = header_y + header_height + 15
    reading_y = topic_y + topic_height + 15
    instructions_y = reading_y + reading_section_height + 20
    questions_start_y = instructions_y + instructions_height + 20
    
    # Generate questions dynamically
    questions_svg = ""
    for i in range(1, min(num_questions + 1, 9)):  # Limit to 8 questions to fit page
        question_y = questions_start_y + ((i - 1) * question_height) + 25
        answer_y = question_y + 15
        line_y = answer_y
        
        questions_svg += f'''
  <foreignObject x="{margin + 20}" y="{question_y - 15}" width="{content_width - 40}" height="20">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 10px; color: #2c3e50; margin: 0;">
      Q{i}. [question{i}]
    </div>
  </foreignObject>
  <text x="{margin + 20}" y="{answer_y}" class="question">Ans: _______________</text>
  <line x1="{margin + 55}" y1="{line_y}" x2="{margin + content_width - 20}" y2="{line_y}" stroke="#bdc3c7" stroke-width="1"/>
'''
    
    # Calculate extra space position
    extra_space_y = questions_start_y + (min(num_questions, 8) * question_height) + 40
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #2c3e50; }}
      .subtitle {{ font-family: Arial, sans-serif; font-size: 12px; fill: #34495e; }}
      .small {{ font-family: Arial, sans-serif; font-size: 9px; fill: #2c3e50; }}
      .instruction {{ font-family: Arial, sans-serif; font-size: 11px; fill: #2c3e50; font-weight: bold; }}
      .question {{ font-family: Arial, sans-serif; font-size: 10px; fill: #2c3e50; }}
      .passage {{ font-family: Arial, sans-serif; font-size: 10px; fill: #2c3e50; }}
    </style>
  </defs>
  
  <!-- A4 Background -->
  <rect width="{svg_width}" height="{svg_height}" fill="#ffffff" stroke="none"/>
  
  <!-- Header -->
  <rect x="{margin}" y="{header_y}" width="{content_width}" height="{header_height}" fill="#ffffff" stroke="#2c3e50" stroke-width="1"/>
  <text x="{margin + 20}" y="{header_y + 20}" class="title">[subject] / Practice Comprehension / Grade [grade]</text>
  <text x="{margin + 20}" y="{header_y + 30}" class="small">Name: _______________</text>
  <text x="{margin + 470}" y="{header_y + 30}" class="small">Date: ______</text>
  
  <!-- Topic -->
  <text x="{svg_width // 2}" y="{topic_y + 15}" class="instruction" text-anchor="middle">Topic - [topic]</text>
  
  <!-- Reading passage area -->
  <text x="{margin + 20}" y="{reading_y - 10}" class="instruction">Read:</text>
  <rect x="{margin + 20}" y="{reading_y}" width="480" height="{reading_section_height}" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
  <foreignObject x="{margin + 30}" y="{reading_y + 10}" width="460" height="{reading_section_height - 20}">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 10px; color: #2c3e50; line-height: 1.3; word-wrap: break-word; margin: 0; padding: 5px;">
      [READING_PASSAGE]
    </div>
  </foreignObject>
  
  <!-- Small image on right -->
  <rect x="{margin + 520}" y="{reading_y}" width="154" height="120" fill="#e8f4fd" stroke="#bdc3c7" stroke-width="2" stroke-dasharray="3,3" id="placeholder_image"/>
  
  <!-- Instructions for questions -->
  <foreignObject x="{margin + 20}" y="{instructions_y}" width="{content_width - 40}" height="{instructions_height}">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 11px; color: #2c3e50; font-weight: bold; line-height: 1.4; word-wrap: break-word; margin: 0; padding: 5px 0;">
      [instructions]
    </div>
  </foreignObject>
  
  <!-- Q&A Section -->
{questions_svg}
  
  <!-- Extra space -->
  <rect x="{margin + 20}" y="{extra_space_y}" width="{content_width - 40}" height="{extra_space_height}" fill="none" stroke="#bdc3c7" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="{margin + 30}" y="{extra_space_y + 15}" class="small">Extra space for additional work:</text>
  
  <!-- Footer -->
  <rect x="{margin}" y="{svg_height - margin - footer_height}" width="{content_width}" height="{footer_height}" fill="#f8f9fa"/>
  <text x="{svg_width // 2}" y="{svg_height - margin - 10}" class="small" text-anchor="middle">Page 1</text>
</svg>'''
    
    return svg_content


def generate_layout3_template(num_questions: int, subject: str, grade_level: str, topic: str) -> str:
    """
    Generate Layout 3 template with dynamic number of questions (Multiple choice style).
    """
    
    # A4 dimensions with proper margins
    svg_width = 794
    svg_height = 1123
    margin = 50
    content_width = svg_width - (2 * margin)
    
    # Calculate heights and positions
    header_height = 50
    title_height = 60
    instructions_height = 35
    image_height = 220
    question_height = 85  # Height per multiple choice question (with 4 options)
    extra_space_height = 100
    footer_height = 25
    
    # Y positions
    header_y = margin
    title_y = header_y + 30
    instructions_y = title_y + title_height
    image_y = instructions_y + instructions_height + 15
    
    # Generate questions dynamically
    questions_svg = ""
    
    # First questions go on the right side of the image (max 3)
    right_side_questions = min(3, num_questions)
    for i in range(1, right_side_questions + 1):
        question_y = image_y + 15 + ((i - 1) * 75)
        
        questions_svg += f'''
  <foreignObject x="430" y="{question_y - 10}" width="294" height="20">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 11px; color: #2c3e50; margin: 0;">
      {i}. [question{i}]
    </div>
  </foreignObject>
  <circle cx="440" cy="{question_y + 12}" r="3" fill="none" stroke="#2c3e50" stroke-width="1"/>
  <text x="448" y="{question_y + 16}" class="option">A) [OPTION_A]</text>
  <circle cx="440" cy="{question_y + 25}" r="3" fill="none" stroke="#2c3e50" stroke-width="1"/>
  <text x="448" y="{question_y + 29}" class="option">B) [OPTION_B]</text>
  <circle cx="440" cy="{question_y + 38}" r="3" fill="none" stroke="#2c3e50" stroke-width="1"/>
  <text x="448" y="{question_y + 42}" class="option">C) [OPTION_C]</text>
  <circle cx="440" cy="{question_y + 51}" r="3" fill="none" stroke="#2c3e50" stroke-width="1"/>
  <text x="448" y="{question_y + 55}" class="option">D) [OPTION_D]</text>
'''
    
    # Additional questions go below the image
    if num_questions > 3:
        below_image_y = image_y + image_height + 20
        for i in range(4, min(num_questions + 1, 9)):  # Limit to total 8 questions
            question_y = below_image_y + ((i - 4) * 40)
            
            questions_svg += f'''
  <foreignObject x="{margin + 20}" y="{question_y - 10}" width="{content_width - 40}" height="20">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 11px; color: #2c3e50; margin: 0;">
      {i}. [question{i}]
    </div>
  </foreignObject>
  <circle cx="{margin + 30}" cy="{question_y + 12}" r="3" fill="none" stroke="#2c3e50" stroke-width="1"/>
  <text x="{margin + 38}" y="{question_y + 16}" class="option">A) [OPTION_A]</text>
  <circle cx="{margin + 150}" cy="{question_y + 12}" r="3" fill="none" stroke="#2c3e50" stroke-width="1"/>
  <text x="{margin + 158}" y="{question_y + 16}" class="option">B) [OPTION_B]</text>
  <circle cx="{margin + 270}" cy="{question_y + 12}" r="3" fill="none" stroke="#2c3e50" stroke-width="1"/>
  <text x="{margin + 278}" y="{question_y + 16}" class="option">C) [OPTION_C]</text>
  <circle cx="{margin + 390}" cy="{question_y + 12}" r="3" fill="none" stroke="#2c3e50" stroke-width="1"/>
  <text x="{margin + 398}" y="{question_y + 16}" class="option">D) [OPTION_D]</text>
'''
    
    # Calculate extra space position
    extra_space_y = max(image_y + image_height + 20, 
                        below_image_y + (max(0, min(num_questions - 3, 5)) * 40) + 40 if num_questions > 3 else image_y + image_height + 20)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; fill: #2c3e50; text-anchor: middle; }}
      .subtitle {{ font-family: Arial, sans-serif; font-size: 14px; fill: #34495e; text-anchor: middle; }}
      .small {{ font-family: Arial, sans-serif; font-size: 10px; fill: #2c3e50; }}
      .instruction {{ font-family: Arial, sans-serif; font-size: 12px; fill: #2c3e50; font-weight: bold; }}
      .question {{ font-family: Arial, sans-serif; font-size: 11px; fill: #2c3e50; }}
      .option {{ font-family: Arial, sans-serif; font-size: 10px; fill: #2c3e50; }}
    </style>
  </defs>
  
  <!-- A4 Background -->
  <rect width="{svg_width}" height="{svg_height}" fill="#ffffff" stroke="none"/>
  
  <!-- Header with name/date -->
  <text x="{margin + 20}" y="{header_y + 20}" class="small">Name: ________________</text>
  <text x="{margin + 520}" y="{header_y + 20}" class="small">Date: ________________</text>
  
  <!-- Title section -->
  <text x="{svg_width // 2}" y="{title_y + 20}" class="title">[subject]</text>
  <text x="{svg_width // 2}" y="{title_y + 40}" class="subtitle">[topic]</text>
  
  <!-- Instructions -->
  <foreignObject x="{margin + 20}" y="{instructions_y}" width="{content_width - 40}" height="{instructions_height}">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 12px; color: #2c3e50; font-weight: bold; line-height: 1.4; word-wrap: break-word; text-align: center; margin: 0; padding: 5px 0;">
      [instructions]
    </div>
  </foreignObject>
  
  <!-- Image positioned center-left -->
  <rect x="{margin + 20}" y="{image_y}" width="320" height="{image_height}" fill="#e8f4fd" stroke="#bdc3c7" stroke-width="2" stroke-dasharray="5,5" id="placeholder_image"/>
  
  <!-- Multiple choice questions -->
{questions_svg}
  
  <!-- Extra space -->
  <rect x="{margin + 20}" y="{extra_space_y}" width="{content_width - 40}" height="{extra_space_height}" fill="none" stroke="#bdc3c7" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="{margin + 30}" y="{extra_space_y + 15}" class="small">Extra space for work/explanations:</text>
  
  <!-- Footer -->
  <rect x="{margin}" y="{svg_height - margin - footer_height}" width="{content_width}" height="{footer_height}" fill="#f8f9fa"/>
  <text x="{svg_width // 2}" y="{svg_height - margin - 10}" class="small" text-anchor="middle">Page 1</text>
</svg>'''
    
    return svg_content

def generate_dynamic_math_template(num_questions: int, subject: str, grade_level: str, topic: str) -> str:
    """
    Generate a dynamic math worksheet template with proper A4 layout.
    """
    
    # A4 dimensions with proper margins
    svg_width = 794
    svg_height = 1123
    margin = 50
    content_width = svg_width - (2 * margin)
    
    # Calculate heights and positions
    header_height = 60
    instructions_height = 30
    image_height = 150
    problem_height = 50  # Height per math problem
    work_area_height = 120
    footer_height = 60
    
    # Y positions
    header_y = margin
    instructions_y = header_y + header_height + 10
    image_y = instructions_y + instructions_height + 10
    problems_start_y = image_y + image_height + 20
    
    # Generate problems dynamically
    problems_svg = ""
    
    # Split problems into two columns if there are many
    problems_per_column = min(6, num_questions)
    first_column_problems = min(problems_per_column, num_questions)
    
    # First column problems
    for i in range(1, first_column_problems + 1):
        problem_y = problems_start_y + ((i - 1) * problem_height)
        
        problems_svg += f'''
  <foreignObject x="{margin + 20}" y="{problem_y - 10}" width="280" height="25">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 16px; color: #2c3e50; font-weight: bold; margin: 0;">
      {i}. [problem{i}]
    </div>
  </foreignObject>
  <rect x="{margin + 320}" y="{problem_y - 15}" width="100" height="30" fill="#ffffff" stroke="#3498db" stroke-width="2"/>
'''
    
    # Second column problems if needed
    if num_questions > problems_per_column:
        for i in range(problems_per_column + 1, min(num_questions + 1, problems_per_column * 2 + 1)):
            problem_y = problems_start_y + ((i - problems_per_column - 1) * problem_height)
            
            problems_svg += f'''
  <foreignObject x="{margin + 450}" y="{problem_y - 10}" width="200" height="25">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 16px; color: #2c3e50; font-weight: bold; margin: 0;">
      {i}. [problem{i}]
    </div>
  </foreignObject>
  <rect x="{margin + 650}" y="{problem_y - 15}" width="80" height="30" fill="#ffffff" stroke="#3498db" stroke-width="2"/>
'''
    
    # Calculate work area position
    max_problems_in_column = max(first_column_problems, num_questions - problems_per_column if num_questions > problems_per_column else 0)
    work_area_y = problems_start_y + (max_problems_in_column * problem_height) + 20
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; fill: #2c3e50; }}
      .subtitle {{ font-family: Arial, sans-serif; font-size: 16px; fill: #34495e; }}
      .problem {{ font-family: Arial, sans-serif; font-size: 18px; fill: #2c3e50; font-weight: bold; }}
      .instruction {{ font-family: Arial, sans-serif; font-size: 14px; fill: #7f8c8d; }}
      .small {{ font-family: Arial, sans-serif; font-size: 10px; fill: #2c3e50; }}
    </style>
  </defs>
  
  <!-- A4 Background -->
  <rect width="{svg_width}" height="{svg_height}" fill="#ffffff" stroke="none"/>
  
  <!-- Header with proper margins -->
  <rect x="{margin}" y="{header_y}" width="{content_width}" height="{header_height}" fill="#f39c12" fill-opacity="0.1"/>
  <text x="{margin + 20}" y="{header_y + 25}" class="title">[topic] - Math Practice</text>
  <text x="{margin + 20}" y="{header_y + 45}" class="subtitle">[subject] - Grade [grade]</text>
  
  <!-- Instructions -->
  <foreignObject x="{margin + 20}" y="{instructions_y}" width="{content_width - 40}" height="{instructions_height}">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 14px; color: #7f8c8d; margin: 0;">
      [instructions]
    </div>
  </foreignObject>
  
  <!-- Visual aid -->
  <rect x="{margin + 20}" y="{image_y}" width="200" height="{image_height}" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2" stroke-dasharray="5,5" id="placeholder_image"/>
  
  <!-- Math Problems -->
{problems_svg}
  
  <!-- Show your work section -->
  <text x="{margin + 20}" y="{work_area_y}" class="subtitle">Show your work:</text>
  <rect x="{margin + 20}" y="{work_area_y + 10}" width="{content_width - 40}" height="{work_area_height}" fill="#ffffff" stroke="#bdc3c7" stroke-width="1"/>
  
  <!-- Footer -->
  <rect x="{margin}" y="{svg_height - margin - header_height}" width="{content_width}" height="{header_height}" fill="#95a5a6" fill-opacity="0.1"/>
  <text x="{margin + 20}" y="{svg_height - margin - 35}" class="subtitle">Name: ________________________</text>
  <text x="{margin + 350}" y="{svg_height - margin - 35}" class="subtitle">Date: ________________________</text>
</svg>'''
    
    return svg_content

def generate_dynamic_worksheet_template(num_questions: int, subject: str, grade_level: str, topic: str) -> str:
    """
    Generate a dynamic general worksheet template with proper A4 layout.
    """
    
    # A4 dimensions with proper margins
    svg_width = 794
    svg_height = 1123
    margin = 50
    content_width = svg_width - (2 * margin)
    
    # Calculate heights and positions
    header_height = 60
    section_height = 120  # Height per section
    footer_height = 60
    
    # Y positions
    header_y = margin
    sections_start_y = header_y + header_height + 20
    
    # Generate sections dynamically (limit to 3 main sections to fit page)
    sections_svg = ""
    max_sections = min(3, (num_questions + 2) // 3)  # Distribute questions across sections
    
    for section in range(1, max_sections + 1):
        section_y = sections_start_y + ((section - 1) * (section_height + 30))
        
        # Questions per section
        questions_in_section = num_questions // max_sections
        if section <= (num_questions % max_sections):
            questions_in_section += 1
        
        sections_svg += f'''
  <!-- Section {section} -->
  <text x="{margin + 20}" y="{section_y}" class="section-title">[SECTION_{section}_TITLE]</text>
  
  <!-- Image placeholder for section -->
  <rect x="{margin + 20}" y="{section_y + 10}" width="150" height="80" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2" stroke-dasharray="5,5"/>
  
  <!-- Content area for section -->
  <foreignObject x="{margin + 190}" y="{section_y + 10}" width="{content_width - 170}" height="30">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 14px; color: #2c3e50; margin: 0;">
      [SECTION_{section}_CONTENT]
    </div>
  </foreignObject>
  
  <!-- Answer lines for section -->
  <line x1="{margin + 190}" y1="{section_y + 50}" x2="{margin + content_width - 20}" y2="{section_y + 50}" stroke="#bdc3c7" stroke-width="1"/>
  <line x1="{margin + 190}" y1="{section_y + 70}" x2="{margin + content_width - 20}" y2="{section_y + 70}" stroke="#bdc3c7" stroke-width="1"/>
  <line x1="{margin + 190}" y1="{section_y + 90}" x2="{margin + content_width - 20}" y2="{section_y + 90}" stroke="#bdc3c7" stroke-width="1"/>
'''
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; fill: #2c3e50; }}
      .subtitle {{ font-family: Arial, sans-serif; font-size: 16px; fill: #34495e; }}
      .section-title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; fill: #2c3e50; }}
      .text {{ font-family: Arial, sans-serif; font-size: 14px; fill: #2c3e50; }}
      .small {{ font-family: Arial, sans-serif; font-size: 10px; fill: #2c3e50; }}
    </style>
  </defs>
  
  <!-- A4 Background -->
  <rect width="{svg_width}" height="{svg_height}" fill="#ffffff" stroke="none"/>
  
  <!-- Header -->
  <rect x="{margin}" y="{header_y}" width="{content_width}" height="{header_height}" fill="#9b59b6" fill-opacity="0.1"/>
  <text x="{margin + 20}" y="{header_y + 25}" class="title">[topic] - [subject] Worksheet</text>
  <text x="{margin + 20}" y="{header_y + 45}" class="subtitle">Grade [grade]</text>
  
  <!-- Dynamic sections -->
{sections_svg}
  
  <!-- Footer -->
  <rect x="{margin}" y="{svg_height - margin - header_height}" width="{content_width}" height="{header_height}" fill="#95a5a6" fill-opacity="0.1"/>
  <text x="{margin + 20}" y="{svg_height - margin - 35}" class="subtitle">Name: ________________________</text>
  <text x="{margin + 350}" y="{svg_height - margin - 35}" class="subtitle">Date: ________________________</text>
</svg>'''
    
    return svg_content


def generate_dynamic_comic_template(num_questions: int, subject: str, grade_level: str, topic: str) -> str:
    """
    Generate a dynamic comic template with proper A4 layout.
    """
    
    # A4 dimensions with proper margins
    svg_width = 794
    svg_height = 1123
    margin = 50
    content_width = svg_width - (2 * margin)
    
    # Calculate panel dimensions
    header_height = 60
    panel_width = (content_width - 30) // 2  # Two panels per row with spacing
    panel_height = 180
    panels_per_row = 2
    rows_needed = (num_questions + 1) // panels_per_row  # +1 for ceiling division
    footer_height = 60
    
    # Y positions
    header_y = margin
    panels_start_y = header_y + header_height + 20
    
    # Generate panels dynamically
    panels_svg = ""
    
    for i in range(1, min(num_questions + 1, 7)):  # Limit to 6 panels to fit page
        row = (i - 1) // panels_per_row
        col = (i - 1) % panels_per_row
        
        panel_x = margin + 20 + (col * (panel_width + 15))
        panel_y = panels_start_y + (row * (panel_height + 20))
        
        panels_svg += f'''
  <!-- Panel {i} -->
  <rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#2c3e50" stroke-width="3"/>
  
  <!-- Image area -->
  <rect x="{panel_x + 10}" y="{panel_y + 10}" width="{panel_width - 120}" height="{panel_height - 60}" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2" stroke-dasharray="5,5"/>
  
  <!-- Speech bubble -->
  <ellipse cx="{panel_x + panel_width - 60}" cy="{panel_y + 50}" rx="50" ry="25" fill="#ffffff" stroke="#2c3e50" stroke-width="2"/>
  <foreignObject x="{panel_x + panel_width - 100}" y="{panel_y + 35}" width="80" height="30">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; font-size: 10px; color: #2c3e50; text-align: center; margin: 0;">
      [SPEECH_{i}]
    </div>
  </foreignObject>
'''
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <defs>
    <style>
      .title {{ font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; fill: #2c3e50; }}
      .subtitle {{ font-family: Arial, sans-serif; font-size: 16px; fill: #34495e; }}
      .speech-text {{ font-family: Arial, sans-serif; font-size: 12px; fill: #2c3e50; }}
      .small {{ font-family: Arial, sans-serif; font-size: 10px; fill: #2c3e50; }}
    </style>
  </defs>
  
  <!-- A4 Background -->
  <rect width="{svg_width}" height="{svg_height}" fill="#ffffff" stroke="none"/>
  
  <!-- Header -->
  <rect x="{margin}" y="{header_y}" width="{content_width}" height="{header_height}" fill="#e74c3c" fill-opacity="0.1"/>
  <text x="{margin + 20}" y="{header_y + 25}" class="title">[topic] - Comic Strip</text>
  <text x="{margin + 20}" y="{header_y + 45}" class="subtitle">[subject] - Grade [grade]</text>
  
  <!-- Comic panels -->
{panels_svg}
  
  <!-- Footer -->
  <rect x="{margin}" y="{svg_height - margin - footer_height}" width="{content_width}" height="{footer_height}" fill="#95a5a6" fill-opacity="0.1"/>
  <text x="{margin + 20}" y="{svg_height - margin - 35}" class="subtitle">Name: ________________________</text>
  <text x="{margin + 350}" y="{svg_height - margin - 35}" class="subtitle">Date: ________________________</text>
</svg>'''
    
    return svg_content

def analyze_lesson_plan_with_ai(lesson_plan: str, max_images: int = 5) -> List[ImagePrompt]:
    """
    Use AI to analyze a lesson plan and generate appropriate image prompts.
    
    Args:
        lesson_plan: The text content of the lesson plan
        max_images: Maximum number of image prompts to generate
        
    Returns:
        A list of ImagePrompt objects with prompt and explanation
    """
    try:
        # Initialize the AI service
        ai_service = AIService()
        
        # Create a prompt for the AI model
        ai_prompt = f"""
        Analyze the following lesson plan and identify {max_images} key concepts that would benefit from visual aids or illustrations.
        For each concept, provide:
        1. A detailed image generation prompt that would create an educational illustration
        2. A brief explanation of why this image would be useful for teaching this lesson
        
        Format your response as a JSON array with objects containing 'prompt' and 'explanation' fields.
        
        LESSON PLAN:
        {lesson_plan}
        """
        
        # Query the AI model
        system_message = """
        You are an expert educational content creator and visual learning specialist.
        Your task is to analyze lesson plans and identify key concepts that would benefit from visual aids.
        For each concept, create a detailed image generation prompt and explain why the image would enhance learning.
        Always respond with properly formatted JSON only, with no additional text.
        """
        
        response = ai_service.query_model(ai_prompt, system_message)
        
        if not response:
            logger.error("Failed to get response from AI model")
            # Fall back to the simple approach if AI fails
            return generate_simple_image_prompts(lesson_plan, max_images)
        
        # Parse the JSON response
        try:
            # Try to extract JSON if it's wrapped in markdown code blocks
            if "```json" in response:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
                if json_match:
                    response = json_match.group(1)
            
            # Clean up any non-JSON text
            response = response.strip()
            if response.startswith('[') and response.endswith(']'):
                # It's already a JSON array
                pass
            elif '{' in response and '}' in response:
                # Extract just the JSON part
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    response = response[start:end]
            
            prompts_data = json.loads(response)
            
            # Convert to ImagePrompt objects
            image_prompts = []
            for item in prompts_data:
                if isinstance(item, dict) and 'prompt' in item and 'explanation' in item:
                    image_prompts.append(ImagePrompt(
                        prompt=item['prompt'],
                        explanation=item['explanation']
                    ))
            
            # If we got valid prompts, return them
            if image_prompts:
                return image_prompts
            else:
                logger.warning("AI response didn't contain valid prompts, falling back to simple approach")
                return generate_simple_image_prompts(lesson_plan, max_images)
                
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            logger.error(f"Raw response: {response}")
            # Fall back to the simple approach if parsing fails
            return generate_simple_image_prompts(lesson_plan, max_images)
    
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        # Fall back to the simple approach if anything goes wrong
        return generate_simple_image_prompts(lesson_plan, max_images)

def extract_key_concepts(lesson_plan: str) -> List[str]:
    """
    Extract key educational concepts from a lesson plan that would benefit from visualization.
    
    Args:
        lesson_plan: The text content of the lesson plan
        
    Returns:
        A list of key concepts suitable for visualization
    """
    nltk = lazy_import_nltk()
    # Tokenize the text
    sentences = sent_tokenize(lesson_plan)
    
    # Extract potential concepts (nouns and noun phrases)
    concepts = []
    
    # Look for "Objective:" section which often contains key concepts
    objective_match = re.search(r'Objective:(.+?)(?:\n\n|\n[A-Z]|$)', lesson_plan, re.DOTALL)
    if objective_match:
        objective_text = objective_match.group(1).strip()
        concepts.append(objective_text)
    
    # Look for "Materials:" section which often indicates visual elements
    materials_match = re.search(r'Materials:(.+?)(?:\n\n|\n[A-Z]|$)', lesson_plan, re.DOTALL)
    materials = []
    if materials_match:
        materials_text = materials_match.group(1).strip()
        materials = [m.strip() for m in materials_text.split(',')]
        visual_materials = [m for m in materials if any(vm in m.lower() for vm in
                           ['diagram', 'model', 'picture', 'image', 'map', 'chart', 'graph', 'visual'])]
        concepts.extend(visual_materials)
    
    # Look for "Activities:" section which often contains key learning activities
    activities_match = re.search(r'Activities:(.+?)(?:\n\n|\n[A-Z]|$)', lesson_plan, re.DOTALL)
    if activities_match:
        activities_text = activities_match.group(1).strip()
        # Extract numbered activities
        activities = re.findall(r'\d+\.\s*(.+?)(?:\n\d+\.|\n\n|\n[A-Z]|$)', activities_text, re.DOTALL)
        concepts.extend([a.strip() for a in activities])
    
    # Extract subject-specific terms
    words = word_tokenize(lesson_plan)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    
    # Get POS tags
    tagged_words = pos_tag(filtered_words)
    
    # Extract nouns (NN, NNS, NNP, NNPS)
    nouns = [word for word, tag in tagged_words if tag.startswith('NN')]
    
    # Count frequency of nouns
    noun_freq = {}
    for noun in nouns:
        if len(noun) > 3:  # Filter out short nouns
            noun_freq[noun] = noun_freq.get(noun, 0) + 1
    
    # Get most frequent nouns
    important_nouns = sorted(noun_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Add important nouns to concepts
    concepts.extend([noun for noun, _ in important_nouns])
    
    # Remove duplicates while preserving order
    unique_concepts = []
    for concept in concepts:
        if concept and concept not in unique_concepts:
            unique_concepts.append(concept)
    
    return unique_concepts

def generate_simple_image_prompts(lesson_plan: str, max_images: int = 5) -> List[ImagePrompt]:
    """
    Generate appropriate image prompts for a lesson plan using a simple approach.
    This is used as a fallback if the AI analysis fails.
    
    Args:
        lesson_plan: The text content of the lesson plan
        max_images: Maximum number of image prompts to generate
        
    Returns:
        A list of ImagePrompt objects with prompt and explanation
    """
    # Extract key concepts using the NLTK-based approach
    concepts = extract_key_concepts(lesson_plan)
    
    image_prompts = []
    
    # Limit to max_images
    concepts = concepts[:max_images]
    
    for concept in concepts:
        # Clean up the concept
        concept = concept.strip()
        
        # Skip if concept is too short
        if len(concept) < 4:
            continue
            
        # Generate a detailed prompt for the image generator
        prompt = f"Educational illustration of {concept}, detailed, clear, colorful, educational style"
        
        # Generate an explanation for why this image would be useful
        explanation = f"This image will help visualize the concept of '{concept}' which is a key element in the lesson plan."
        
        image_prompts.append(ImagePrompt(prompt=prompt, explanation=explanation))
    
    return image_prompts

# Add a semaphore to limit concurrent image generation
image_generation_semaphore = asyncio.Semaphore(1)

@app.post("/api/analyze-lesson-plan", response_model=LessonPlanAnalysisResponse)
async def analyze_lesson_plan(request: LessonPlanAnalysisRequest):
    """
    Analyze a lesson plan and generate appropriate image prompts.
    
    Args:
        request: LessonPlanAnalysisRequest containing the lesson plan text
        
    Returns:
        LessonPlanAnalysisResponse with a list of image prompts
    """
    try:
        print("==== START: /api/analyze-lesson-plan endpoint called ====")
        print(f"Analyzing lesson plan of length: {len(request.lesson_plan)} characters")
        
        # Use the AI-based approach to analyze the lesson plan
        image_prompts = analyze_lesson_plan_with_ai(request.lesson_plan, request.max_images)
        print(f"Generated {len(image_prompts)} image prompts using AI")
        
        return LessonPlanAnalysisResponse(image_prompts=image_prompts)
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"==== ERROR: Exception in /api/analyze-lesson-plan endpoint ====")
        print(f"Error type: {error_type}")
        print(f"Error message: {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze lesson plan: {error_type}: {error_msg}")
    finally:
        print("==== END: /api/analyze-lesson-plan endpoint processing ====")

@app.get("/api/svg-templates", response_model=List[SVGTemplate])
async def get_svg_templates():
    """Get list of available SVG templates."""
    try:
        logger.info("Getting available SVG templates")
        templates = svg_service.get_available_templates()
        return templates
    except Exception as e:
        logger.error(f"Error getting SVG templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

# Add validation function
def validate_svg_request(request: SVGGenerationRequest):
    """Validate SVG generation request for K-6 education requirements."""
    
    # Valid grades K-6
    valid_grades = ['Kindergarten', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6']
    if request.grade_level not in valid_grades:
        raise HTTPException(status_code=400, detail=f"Grade level must be one of: {', '.join(valid_grades)}")
    
    # Valid subjects
    valid_subjects = ['Mathematics', 'Language Arts', 'Science', 'Social Studies']
    if request.subject not in valid_subjects:
        raise HTTPException(status_code=400, detail=f"Subject must be one of: {', '.join(valid_subjects)}")
    
    # Valid question types
    valid_question_types = ['fill_blank', 'short_answer', 'multiple_choice', 'true_false', 'matching']
    invalid_types = [qt for qt in request.question_types if qt not in valid_question_types]
    if invalid_types:
        raise HTTPException(status_code=400, detail=f"Invalid question types: {', '.join(invalid_types)}")
    
    return True

@app.post("/api/generate-svg", response_model=SVGGenerationResponse)
async def generate_svg(request: SVGGenerationRequest):
    """Generate SVG with word bank support and enhanced overflow handling."""
    try:
        async with image_generation_semaphore:
            logger.info(f"Generating SVG for content type: {request.content_type}")
            
            # Validate request for K-6 education requirements
            validate_svg_request(request)
            
            # Generate dynamic SVG template with word bank option
            svg_content = generate_layout1_template_with_wordbank(
                request.num_questions,
                request.subject,
                request.grade_level,
                request.topic,
                request.include_activity_box if request.include_activity_box is not None else True,
                request.include_word_bank if request.include_word_bank is not None else True
            )
            
            # Use AI to analyze the prompt and generate content
            generated_images = []
            text_replacements = {}
            ai_generated_placeholders = []

            # ALWAYS populate basic fields first
            text_replacements["subject"] = request.subject
            text_replacements["grade"] = request.grade_level
            text_replacements["topic"] = request.topic

            ai_service = AIService()

            if request.content_type == "image_comprehension":
                # Step 1: Generate the image first
                image_prompt = f"Educational illustration showing {request.topic} for {request.subject} class, {request.grade_level}. Clear, colorful, educational style with no text or labels visible."
                
                try:
                    aspect_ratio = request.image_aspect_ratio or {"width": 16, "height": 9}
                    base_size = 512
                    if aspect_ratio["width"] >= aspect_ratio["height"]:
                        width = base_size
                        height = int(base_size * aspect_ratio["height"] / aspect_ratio["width"])
                    else:
                        height = base_size
                        width = int(base_size * aspect_ratio["width"] / aspect_ratio["height"])
                    
                    if USE_HF_API:
                        image = await generate_with_hf_api(image_prompt, width, height)
                    else:
                        image = generate_with_local_model(image_prompt, width, height)
                    
                    image_id = str(uuid.uuid4())
                    filename = f"svg_{image_id}.png"
                    filepath = os.path.join(IMAGES_DIR, filename)
                    image.save(filepath)
                    generated_images.append(f"/static/{filename}")
                    logger.info(f"Successfully generated image: {filename}")
                    
                except Exception as img_error:
                    logger.error(f"Error generating image: {img_error}")

                # Step 2: Generate questions and word bank
                if request.question_type:
                    question_types_to_use = [request.question_type]
                else:
                    question_types_to_use = request.question_types or ["fill_blank"]
                
                question_types_str = ", ".join(question_types_to_use)
                
                word_bank_instruction = ""
                if request.include_word_bank and "fill_blank" in question_types_to_use:
                    word_bank_instruction = "\n- ALSO provide a 'word_bank' array with 8-10 answer words for the fill-in-the-blank questions"

                analysis_prompt = f"""
                Create educational content for an image comprehension worksheet about {request.topic} in {request.subject} for {request.grade_level} students.

                The image shows: {image_prompt}

                REQUIREMENTS:
                - Generate EXACTLY {request.num_questions} fill-in-the-blank questions
                - Each question should have 1-2 blanks using _____ in the sentence
                - Make questions appropriate for {request.grade_level} level
                - Questions should test comprehension of the image content
                {word_bank_instruction}

                Generate:
                1. Clear instructions (mention using the word bank if applicable)
                2. Exactly {request.num_questions} fill-in-the-blank questions
                {word_bank_instruction and "3. A word bank with answer words"}

                Return as JSON with fields: "instructions", "questions" (array){", and 'word_bank' (array)" if "word_bank" in word_bank_instruction else ""}.
                
                Example question format: "The process shown in the image is called the _____ cycle."
                """
                
                response = ai_service.query_model(analysis_prompt, f"Educational content creator for {request.grade_level}")
                
                # Default content
                text_replacements["instructions"] = f"Look at the image showing {request.topic}. Use the word bank to fill in the blanks." if request.include_word_bank else f"Fill in the blanks about {request.topic}."
                
                default_questions = [
                    f"The image shows the _____ of {request.topic}.",
                    f"One important part of {request.topic} is _____.",
                    f"The process helps us understand _____ about {request.topic}.",
                    f"Students can learn that {request.topic} involves _____.",
                    f"The illustration demonstrates _____ in nature.",
                ]
                
                default_word_bank = ["process", "cycle", "important", "nature", "science", "water", "energy", "system"]
                
                if response:
                    try:
                        if "```json" in response:
                            response = re.search(r'```json\s*([\s\S]*?)\s*```', response).group(1)
                        
                        content_data = json.loads(response.strip())
                        
                        text_replacements["instructions"] = content_data.get("instructions", text_replacements["instructions"])
                        ai_generated_placeholders.append("instructions")
                        
                        questions = content_data.get("questions", default_questions)
                        for i in range(request.num_questions):
                            if i < len(questions):
                                text_replacements[f"question{i+1}"] = questions[i]
                            else:
                                text_replacements[f"question{i+1}"] = default_questions[i % len(default_questions)]
                            ai_generated_placeholders.append(f"question{i+1}")
                        
                        # Handle word bank
                        if request.include_word_bank:
                            word_bank = content_data.get("word_bank", default_word_bank)
                            for i in range(10):  # Support up to 10 words
                                if i < len(word_bank):
                                    text_replacements[f"word_bank_word{i+1}"] = word_bank[i]
                                    ai_generated_placeholders.append(f"word_bank_word{i+1}")
                                else:
                                    text_replacements[f"word_bank_word{i+1}"] = ""
                        
                    except Exception as e:
                        logger.error(f"Error parsing AI content response: {e}")
                        # Use defaults
                        text_replacements["instructions"] = f"Fill in the blanks about {request.topic}."
                        for i in range(request.num_questions):
                            text_replacements[f"question{i+1}"] = default_questions[i % len(default_questions)]
                        
                        if request.include_word_bank:
                            for i in range(10):
                                if i < len(default_word_bank):
                                    text_replacements[f"word_bank_word{i+1}"] = default_word_bank[i]
                                else:
                                    text_replacements[f"word_bank_word{i+1}"] = ""

            # ... rest of the content type handling (math, etc.) ...

            # Step 3: Embed images into SVG
            if generated_images:
                try:
                    svg_content = await svg_service.embed_images_in_svg(svg_content, generated_images)
                    logger.info(f"Successfully embedded {len(generated_images)} images")
                except Exception as embed_error:
                    logger.error(f"Error embedding images: {embed_error}")

            # Step 4: Replace text placeholders
            svg_content = svg_service.replace_text_placeholders(svg_content, text_replacements)
            
            features = []
            if request.include_word_bank: features.append("word bank")
            if request.include_activity_box: features.append("activity box")
            features_str = f" with {', '.join(features)}" if features else ""
            
            logger.info(f"Generated dynamic SVG{features_str} with {len(generated_images)} images")
            
            # Save the generated SVG
            svg_id = str(uuid.uuid4())
            svg_filename = f"{svg_id}.svg"
            svg_filepath = os.path.join(SVG_EXPORTS_DIR, svg_filename)
            async with aiofiles.open(svg_filepath, "w", encoding="utf-8") as f:
                await f.write(svg_content)
            
            # Save metadata
            svg_metadata = await load_svg_metadata()
            new_svg_item = {
                "id": svg_id,
                "filename": svg_filename,
                "url": f"/exports/{svg_filename}",
                "template_id": f"{request.content_type}_{request.layout_style or 'layout1'}",
                "created_at": datetime.now().isoformat(),
            }
            svg_metadata.append(new_svg_item)
            await save_svg_metadata(svg_metadata)
            
            return SVGGenerationResponse(
                svg_content=svg_content,
                template_id=f"{request.content_type}_{request.layout_style or 'layout1'}",
                placeholders=ai_generated_placeholders,
                images_generated=generated_images
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating SVG: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate SVG: {str(e)}")
@app.post("/api/process-svg", response_model=SVGProcessingResponse)
async def process_svg(request: SVGProcessingRequest):
    """Replace text placeholders in SVG content."""
    try:
        logger.info(f"Processing SVG with {len(request.text_replacements)} text replacements")
        
        # Replace text placeholders
        processed_svg = svg_service.replace_text_placeholders(
            request.svg_content,
            request.text_replacements
        )
        
        # Add writing lines if requested
        if request.add_writing_lines:
            # This is a simple implementation - could be enhanced
            # For now, we'll just return the processed SVG as-is
            pass
        
        # Get list of replaced placeholders
        replaced_placeholders = list(request.text_replacements.keys())
        
        logger.info(f"Successfully processed SVG with {len(replaced_placeholders)} replacements")
        
        return SVGProcessingResponse(
            processed_svg=processed_svg,
            replaced_placeholders=replaced_placeholders
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing SVG: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process SVG: {str(e)}")

@app.post("/api/export-svg", response_model=SVGExportResponse)
async def export_svg(request: SVGExportRequest):
    """Export SVG to different formats (PDF, DOCX, PNG)."""
    try:
        logger.info(f"Exporting SVG to {request.format} format")
        
        # Validate format
        if request.format not in ["pdf", "docx", "png"]:
            raise HTTPException(status_code=400, detail="Invalid export format. Use 'pdf', 'docx', or 'png'")
        
        # Clean filename
        clean_filename = re.sub(r'[^\w\-_\.]', '_', request.filename)
        if not clean_filename:
            clean_filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Export based on format
        if request.format == "pdf":
            output_path = svg_service.export_svg_to_pdf(request.svg_content, clean_filename)
        elif request.format == "png":
            output_path = svg_service.export_svg_to_png(request.svg_content, clean_filename)
        elif request.format == "docx":
            output_path = svg_service.export_svg_to_docx(request.svg_content, clean_filename)
        
        # Create download URL (relative to the exports directory)
        download_url = f"/exports/{os.path.basename(output_path)}"
        
        logger.info(f"Successfully exported SVG to {output_path}")
        
        return SVGExportResponse(
            download_url=download_url,
            filename=os.path.basename(output_path),
            format=request.format
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting SVG: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export SVG: {str(e)}")

def validate_and_adjust_parameters(request: GenerateImageRequest) -> GenerateImageRequest:
    """
    Validate and adjust request parameters to improve reliability and reduce timeout risk.
    Returns a new request object with adjusted parameters if needed.
    """
    adjusted = GenerateImageRequest(**request.dict())
    
    # Check if dimensions are too large (which can cause timeouts)
    max_pixels = 1024 * 1024  # 1M pixels is a reasonable limit
    current_pixels = request.width * request.height
    
    if current_pixels > max_pixels:
        # Scale down while preserving aspect ratio
        scale_factor = (max_pixels / current_pixels) ** 0.5
        adjusted.width = int(request.width * scale_factor)
        adjusted.height = int(request.height * scale_factor)
        
        # Ensure dimensions are multiples of 8 (common requirement for image models)
        adjusted.width = (adjusted.width // 8) * 8
        adjusted.height = (adjusted.height // 8) * 8
    
    # Check if inference steps are too high
    if request.num_inference_steps > 50:
        adjusted.num_inference_steps = 50
    
    # Return original request if no adjustments were made
    if (adjusted.width == request.width and
        adjusted.height == request.height and
        adjusted.num_inference_steps == request.num_inference_steps):
        return request
    
    return adjusted

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)