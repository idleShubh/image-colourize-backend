from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import base64
from image_coloriser import colorize_image
import logging
from datetime import datetime
from PIL import Image
from database import db, connect_db, disconnect_db
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum image dimensions (in pixels)
MAX_DIMENSION = 2048
# Maximum file size (in bytes) - 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024

app = FastAPI(
    title="Image Colorization API",
    description="API for colorizing black and white images using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await connect_db()

@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()

def resize_image_if_needed(image: np.ndarray) -> np.ndarray:
    """Resize image if it exceeds maximum dimensions while maintaining aspect ratio."""
    height, width = image.shape[:2]
    if height > MAX_DIMENSION or width > MAX_DIMENSION:
        scale = min(MAX_DIMENSION / height, MAX_DIMENSION / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert a numpy array image to base64 string with compression."""
    # Convert to PIL Image for better compression
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Save to temporary buffer with compression
    buffer = io.BytesIO()
    image_pil.save(buffer, format='JPEG', quality=85, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.post("/colorize")
async def colorize_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to colorize a black and white image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response containing the original and colorized images as base64 strings
    """
    try:
        # Check file size
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # Read and decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Resize image if needed
        image = resize_image_if_needed(image)
        
        # Colorize the image
        original, colorized = colorize_image(image)
        
        # Convert images to base64 with compression
        original_base64 = encode_image_to_base64(original)
        colorized_base64 = encode_image_to_base64(colorized)
        
        # Save to database
        image_data = await db.image.create({
            "originalName": file.filename,
            "originalImage": original_base64,
            "colorizedImage": colorized_base64,
        })
        
        return JSONResponse({
            "status": "success",
            "id": image_data.id,
            "original_image": original_base64,
            "colorized_image": colorized_base64,
            "created_at": image_data.createdAt.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{image_id}")
async def get_image(image_id: str):
    """Get an image by ID."""
    try:
        image = await db.image.find_unique(where={"id": image_id})
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return JSONResponse({
            "id": image.id,
            "original_name": image.originalName,
            "original_image": image.originalImage,
            "colorized_image": image.colorizedImage,
            "created_at": image.createdAt.isoformat()
        })
    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 