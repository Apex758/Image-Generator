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
from dotenv import load_dotenv
import re
import nltk
import time
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import logging

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

# Create directories
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "static"), exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=IMAGES_DIR), name="static")

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
                import os
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
        filepath = os.path.join(IMAGES_DIR, image_to_delete["filename"])
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

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    print("NLTK resources downloaded successfully")

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