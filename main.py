from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import requests
from datetime import datetime
import os
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import aiohttp
import base64
import io
from PIL import Image
import uuid
from pydantic import BaseModel
from typing import List, Optional
import json

app = FastAPI(title="FaceSwap API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
# MongoDB connection - UPDATED VERSION
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://harilogicgo_db_user:logicgoinfotech@cluster0.dcs1tnb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
client = AsyncIOMotorClient(MONGODB_URL)
database = client.FaceSwap
target_images_collection = database.get_collection("Target_Images")
source_images_collection = database.get_collection("Source_Images")
results_collection = database.get_collection("Results")

# Hugging Face Space URL
HF_SPACE_URL = "https://logicgoinfotechspaces-faceswap.hf.space"

# Pydantic models
class ImageResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    uploaded_at: datetime

class FaceSwapRequest(BaseModel):
    source_image_id: str
    target_image_id: str

class FaceSwapResponse(BaseModel):
    result_id: str
    message: str

# Utility functions
async def call_huggingface_space(src_img_data, tgt_img_data):
    """
    Call the Hugging Face Space API for face swapping
    """
    try:
        # Prepare files for the request
        files = {
            'src_img': ('source.jpg', src_img_data, 'image/jpeg'),
            'tgt_img': ('target.jpg', tgt_img_data, 'image/jpeg')
        }
        
        # Make the request to Hugging Face Space
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HF_SPACE_URL}/api/faceswap",
                data=files
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Hugging Face API error: {error_text}")
                
                # Get the image data from response
                result_data = await response.read()
                return result_data, None
                
    except Exception as e:
        return None, str(e)

# API Endpoints

@app.get("/")
async def root():
    return {"message": "FaceSwap API is running", "status": "healthy"}

# 1. Upload Source Image Endpoint
@app.post("/api/upload/source", response_model=ImageResponse)
async def upload_source_image(file: UploadFile = File(...)):
    """
    Upload a source image for face swapping
    """
    try:
        contents = await file.read()
        
        # Verify it's an image
        try:
            Image.open(io.BytesIO(contents))
        except:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Store in MongoDB
        image_doc = {
            "filename": file.filename,
            "image_data": contents,
            "content_type": file.content_type,
            "uploaded_at": datetime.utcnow()
        }
        
        result = await source_images_collection.insert_one(image_doc)
        
        return {
            "id": str(result.inserted_id),
            "filename": file.filename,
            "content_type": file.content_type,
            "uploaded_at": image_doc["uploaded_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Get Target Images Endpoint
@app.get("/api/target-images", response_model=List[ImageResponse])
async def get_target_images():
    """
    Get all available target images from database
    """
    try:
        images = []
        async for image in target_images_collection.find({}, {"image_data": 0}):  # Exclude image data for listing
            images.append({
                "id": str(image["_id"]),
                "filename": image["filename"],
                "content_type": image["content_type"],
                "uploaded_at": image["uploaded_at"]
            })
        
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Get Specific Target Image Endpoint
@app.get("/api/target-images/{image_id}")
async def get_target_image(image_id: str):
    """
    Get a specific target image by ID
    """
    try:
        image = await target_images_collection.find_one({"_id": ObjectId(image_id)})
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return StreamingResponse(
            io.BytesIO(image["image_data"]),
            media_type=image["content_type"],
            headers={"Content-Disposition": f"inline; filename={image['filename']}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. Upload Target Image Endpoint (Admin functionality)
@app.post("/api/upload/target", response_model=ImageResponse)
async def upload_target_image(file: UploadFile = File(...)):
    """
    Upload a target image to the database (Admin endpoint)
    """
    try:
        contents = await file.read()
        
        # Verify it's an image
        try:
            Image.open(io.BytesIO(contents))
        except:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Store in MongoDB
        image_doc = {
            "filename": file.filename,
            "image_data": contents,
            "content_type": file.content_type,
            "uploaded_at": datetime.utcnow()
        }
        
        result = await target_images_collection.insert_one(image_doc)
        
        return {
            "id": str(result.inserted_id),
            "filename": file.filename,
            "content_type": file.content_type,
            "uploaded_at": image_doc["uploaded_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Face Swap Endpoint
@app.post("/api/faceswap", response_model=FaceSwapResponse)
async def perform_face_swap(request: FaceSwapRequest):
    """
    Perform face swap between source and target images
    """
    try:
        # Get source and target images from MongoDB
        source_doc = await source_images_collection.find_one({"_id": ObjectId(request.source_image_id)})
        target_doc = await target_images_collection.find_one({"_id": ObjectId(request.target_image_id)})
        
        if not source_doc:
            raise HTTPException(status_code=404, detail="Source image not found")
        if not target_doc:
            raise HTTPException(status_code=404, detail="Target image not found")
        
        # Call Hugging Face Space for face swapping
        result_data, error = await call_huggingface_space(
            source_doc["image_data"], 
            target_doc["image_data"]
        )
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        # Store result in MongoDB
        result_doc = {
            "filename": f"faceswap_result_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png",
            "image_data": result_data,
            "content_type": "image/png",
            "source_image_id": ObjectId(request.source_image_id),
            "target_image_id": ObjectId(request.target_image_id),
            "created_at": datetime.utcnow()
        }
        
        result = await results_collection.insert_one(result_doc)
        
        return {
            "result_id": str(result.inserted_id),
            "message": "Face swap completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. Download Result Endpoint
@app.get("/api/results/{result_id}")
async def download_result(result_id: str):
    """
    Download the face swap result
    """
    try:
        result = await results_collection.find_one({"_id": ObjectId(result_id)})
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        
        return StreamingResponse(
            io.BytesIO(result["image_data"]),
            media_type=result["content_type"],
            headers={"Content-Disposition": f"attachment; filename={result['filename']}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. Get All Results Endpoint
@app.get("/api/results")
async def get_all_results():
    """
    Get metadata for all face swap results
    """
    try:
        results = []
        async for result in results_collection.find({}, {"image_data": 0}):  # Exclude image data
            results.append({
                "id": str(result["_id"]),
                "filename": result["filename"],
                "content_type": result["content_type"],
                "source_image_id": str(result["source_image_id"]),
                "target_image_id": str(result["target_image_id"]),
                "created_at": result["created_at"]
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 8. Health Check Endpoint
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Preload target images from folder (run once at startup)
async def preload_target_images():
    """
    Preload target images from Target_Images folder to MongoDB
    """
    target_images_dir = "Target_Images"
    if os.path.exists(target_images_dir):
        for filename in os.listdir(target_images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(target_images_dir, filename)
                
                # Check if image already exists in database
                existing = await target_images_collection.find_one({"filename": filename})
                if not existing:
                    with open(file_path, 'rb') as f:
                        image_data = f.read()
                    
                    image_doc = {
                        "filename": filename,
                        "image_data": image_data,
                        "content_type": f"image/{filename.split('.')[-1].lower()}",
                        "uploaded_at": datetime.utcnow()
                    }
                    
                    await target_images_collection.insert_one(image_doc)
                    print(f"Preloaded target image: {filename}")

# Run on startup
@app.on_event("startup")
async def startup_event():
    await preload_target_images()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)