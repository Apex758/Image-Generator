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
from typing import List, Optional, Dict, Any
import base64
from io import BytesIO
from PIL import Image
import requests
import aiohttp
import aiofiles
from dotenv import load_dotenv
import re
import nltk
import time
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import logging
import xml.etree.ElementTree as ET
from docx import Document
from docx.shared import Inches
import cairosvg
import tempfile
import io

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
                               seed: Optional[int] = None) -> Image.Image:
    """Generate image using Hugging Face Inference API with retry mechanism"""
    
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
    topic: str  # NEW
    grade_level: str
    layout_style: Optional[str] = "layout1"  # NEW: layout1, layout2, layout3
    
    # Question configuration - NEW
    num_questions: int = 5
    question_types: List[str] = ["fill_blank", "short_answer", "multiple_choice"]  # NEW
    
    image_count: int = 1  # Fixed to 1 for image comprehension
    aspect_ratio: str = "square"
    custom_instructions: Optional[str] = None


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
check_nltk_resources()

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
            root = ET.fromstring(svg_content)
            for elem in root.iter():
                if elem.get('id') and elem.get('id').startswith('placeholder_'):
                    placeholder_name = elem.get('id').replace('placeholder_', '')
                    placeholders.append(placeholder_name)
        except Exception as e:
            logger.error(f"Error extracting placeholders: {e}")
        
        return placeholders
    
    def replace_text_placeholders(self, svg_content: str, replacements: Dict[str, str]) -> str:
        """Replace text placeholders in SVG content."""
        try:
            # Parse SVG
            root = ET.fromstring(svg_content)
            
            # Replace text placeholders
            for elem in root.iter():
                if elem.get('id') and elem.get('id').startswith('placeholder_'):
                    placeholder_name = elem.get('id').replace('placeholder_', '')
                    if placeholder_name in replacements:
                        # Replace the text content
                        replacement_text = replacements[placeholder_name]
                        if elem.text and '[' in elem.text and ']' in elem.text:
                            elem.text = replacement_text
                        elif elem.text:
                            elem.text = replacement_text
                        else:
                            elem.text = "" # Clear placeholder if no replacement
            
            # Convert back to string
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            logger.error(f"Error replacing text placeholders: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing SVG: {str(e)}")
    
    async def embed_images_in_svg(self, svg_content: str, image_urls: List[str]) -> str:
        """Embed generated images into SVG placeholders."""
        try:
            root = ET.fromstring(svg_content)
            image_index = 0
            
            # Create a parent map to look up parent elements
            parent_map = {c: p for p in root.iter() for c in p}
            
            # Find image placeholders and replace with actual images
            for elem in root.iter():
                if (elem.get('id') and elem.get('id').startswith('placeholder_image') and
                    image_index < len(image_urls)):
                    
                    # Get the parent rect element for positioning
                    parent = parent_map.get(elem)
                    if parent is not None and 'rect' in parent.tag:
                        x = parent.get('x', '0')
                        y = parent.get('y', '0')
                        width = parent.get('width', '100')
                        height = parent.get('height', '100')
                        
                        # Read image and convert to base64
                        image_path = os.path.join(IMAGES_DIR, os.path.basename(image_urls[image_index]))
                        if os.path.exists(image_path):
                            with open(image_path, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode()
                                
                            # Create image element
                            img_elem = ET.Element('image')
                            img_elem.set('x', x)
                            img_elem.set('y', y)
                            img_elem.set('width', width)
                            img_elem.set('height', height)
                            img_elem.set('href', f'data:image/png;base64,{img_data}')
                            img_elem.set('preserveAspectRatio', 'xMidYMid meet')
                            
                            # Replace the placeholder
                            parent.remove(elem)
                            parent.append(img_elem)
                            
                            image_index += 1
            
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            logger.error(f"Error embedding images in SVG: {e}")
            raise HTTPException(status_code=500, detail=f"Error embedding images: {str(e)}")
    
    def export_svg_to_pdf(self, svg_content: str, filename: str) -> str:
        """Export SVG to PDF format."""
        try:
            output_path = os.path.join(self.exports_dir, f"{filename}.pdf")
            
            # Use cairosvg to convert SVG to PDF
            cairosvg.svg2pdf(bytestring=svg_content.encode('utf-8'), write_to=output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Error exporting SVG to PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Error exporting to PDF: {str(e)}")
    
    def export_svg_to_png(self, svg_content: str, filename: str) -> str:
        """Export SVG to PNG format."""
        try:
            output_path = os.path.join(self.exports_dir, f"{filename}.png")
            
            # Use cairosvg to convert SVG to PNG
            cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_path, dpi=300)
            
            return output_path
        except Exception as e:
            logger.error(f"Error exporting SVG to PNG: {e}")
            raise HTTPException(status_code=500, detail=f"Error exporting to PNG: {str(e)}")
    
    def export_svg_to_docx(self, svg_content: str, filename: str) -> str:
        """Export SVG to DOCX format."""
        try:
            # First convert SVG to PNG
            temp_png = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=temp_png.name, dpi=300)
            
            # Create DOCX document
            doc = Document()
            doc.add_picture(temp_png.name, width=Inches(7.5))
            
            output_path = os.path.join(self.exports_dir, f"{filename}.docx")
            doc.save(output_path)
            
            # Clean up temp file
            os.unlink(temp_png.name)
            
            return output_path
        except Exception as e:
            logger.error(f"Error exporting SVG to DOCX: {e}")
            raise HTTPException(status_code=500, detail=f"Error exporting to DOCX: {str(e)}")

# Initialize SVG service
svg_service = SVGService()

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

@app.post("/api/generate-svg", response_model=SVGGenerationResponse)
async def generate_svg(request: SVGGenerationRequest):
    """Generate SVG with text placeholders and embedded images."""
    try:
        async with image_generation_semaphore:
            logger.info(f"Generating SVG for content type: {request.content_type}")
            
            # Select template based on content type and layout style
            if request.content_type == "image_comprehension":
                template_id = f"{request.content_type}_{request.layout_style}"
            else:
                template_id = request.content_type

            if not os.path.exists(os.path.join(SVG_TEMPLATES_DIR, f"{template_id}.svg")):
                 raise HTTPException(status_code=400, detail=f"Template not found for content type '{request.content_type}' and layout '{request.layout_style}'")

            # Validate image count
            content_type_metadata = svg_service.template_metadata.get(request.content_type)
            if content_type_metadata:
                min_images = content_type_metadata.get("minImages", 1)
                max_images = content_type_metadata.get("maxImages", 10)
                if not (min_images <= request.image_count <= max_images):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid image count for {request.content_type}. Must be between {min_images} and {max_images}."
                    )
            
            # Load SVG template
            svg_content = svg_service.load_template(template_id)
            
            # Use AI to analyze the prompt and generate content
            generated_images = []
            text_replacements = {}

            ai_service = AIService()

            if request.content_type == "image_comprehension":
                analysis_prompt = f"""
                Create educational content for an image comprehension worksheet:
                
                Subject: {request.subject}
                Topic: {request.topic}
                Grade Level: {request.grade_level}
                Layout Style: {request.layout_style}
                Number of Questions: {request.num_questions}
                Question Types: {', '.join(request.question_types)}
                
                Generate:
                1. A detailed, one-sentence art prompt for an educational illustration for the worksheet.
                2. Appropriate instructions for students based on the questions.
                3. {request.num_questions} questions about the scene described in the art prompt, using the specified question types.
                4. For "fill_blank" questions, create sentences with blanks. For "multiple_choice", provide options A, B, C, D.
                5. For "short_answer", create open-ended questions.
                6. Make all content age-appropriate for {request.grade_level}.
                
                Return as a single JSON object with three fields: "image_prompt" (a string), "instructions" (a string), and "questions" (an array of objects, each with "type" and "text" fields, and "options" if applicable).
                """
                system_message = "You are an expert educational content creator. Generate worksheet content as a valid JSON object with no extra text."
                
                response = ai_service.query_model(analysis_prompt, system_message)
                
                image_prompt = f"A vibrant educational illustration for {request.subject} on {request.topic}."
                if response:
                    try:
                        if "```json" in response:
                            response = re.search(r'```json\s*([\s\S]*?)\s*```', response).group(1)
                        
                        content_data = json.loads(response.strip())
                        image_prompt = content_data.get("image_prompt", image_prompt)
                        text_replacements["instructions"] = content_data.get("instructions", "")
                        
                        questions = content_data.get("questions", [])
                        for i, q in enumerate(questions[:request.num_questions]):
                            q_text = q.get("text", "")
                            if q.get("type") == "multiple_choice" and q.get("options"):
                                options = " ".join([f"{key}) {value}" for key, value in q.get("options", {}).items()])
                                q_text = f"{q_text} {options}"
                            text_replacements[f"question{i+1}"] = q_text

                    except Exception as e:
                        logger.error(f"Error parsing AI content response: {e}")
                        text_replacements["instructions"] = "Look at the image and answer the questions."
                        for i in range(request.num_questions):
                            text_replacements[f"question{i+1}"] = f"Question {i+1}."
                
                image_prompts = [image_prompt]

            else: # Other content types
                analysis_prompt = f"""
                Create {request.image_count} educational image prompts for {request.subject} content
                for grade {request.grade_level} students on the topic of {request.topic}.
                The content type is {request.content_type}.
                
                User prompt: {request.custom_instructions or ""}
                
                Generate detailed, educational image prompts suitable for the {request.aspect_ratio} aspect ratio.
                Return as JSON array with objects containing 'prompt' field.
                """
                system_message = "You are an expert educational content creator. Generate appropriate image prompts for educational materials. Always respond with properly formatted JSON only."
                response = ai_service.query_model(analysis_prompt, system_message)
                
                image_prompts = []
                if response:
                    try:
                        if "```json" in response:
                            response = re.search(r'```json\s*([\s\S]*?)\s*```', response).group(1)
                        
                        prompts_data = json.loads(response.strip())
                        for item in prompts_data:
                            if isinstance(item, dict) and 'prompt' in item:
                                image_prompts.append(item['prompt'])
                    except:
                        image_prompts = [f"Educational illustration for {request.subject} on {request.topic}"]
                if not image_prompts:
                     image_prompts = [f"Educational illustration for {request.subject} on {request.topic}"]


            # Generate and embed images
            if request.image_count > 0 and image_prompts:
                for i, img_prompt in enumerate(image_prompts[:request.image_count]):
                    try:
                        if request.aspect_ratio == "square": width, height = 512, 512
                        elif request.aspect_ratio == "landscape": width, height = 768, 512
                        elif request.aspect_ratio == "portrait": width, height = 512, 768
                        else: width, height = 512, 512
                        
                        if USE_HF_API: image = await generate_with_hf_api(img_prompt, width, height)
                        else: image = generate_with_local_model(img_prompt, width, height)
                        
                        image_id = str(uuid.uuid4())
                        filename = f"svg_{image_id}.png"
                        filepath = os.path.join(IMAGES_DIR, filename)
                        image.save(filepath)
                        generated_images.append(f"/static/{filename}")
                    except Exception as img_error:
                        logger.error(f"Error generating image {i}: {img_error}")
                        continue
                
                if generated_images:
                    svg_content = await svg_service.embed_images_in_svg(svg_content, generated_images)

            # Populate text placeholders
            svg_content = svg_service.replace_text_placeholders(svg_content, text_replacements)
            
            # Extract placeholders
            placeholders = svg_service.extract_placeholders(svg_content)
            
            logger.info(f"Generated SVG with {len(generated_images)} images and {len(placeholders)} placeholders")
            
            # Save the generated SVG to a file
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
                "template_id": template_id,
                "created_at": datetime.now().isoformat(),
            }
            svg_metadata.append(new_svg_item)
            await save_svg_metadata(svg_metadata)
            return SVGGenerationResponse(
                svg_content=svg_content,
                template_id=template_id,
                placeholders=placeholders,
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