from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from pydantic import BaseModel
from typing import List
from datetime import datetime
import aiohttp
import base64
import io
from PIL import Image
import os
import json
import asyncio
app = FastAPI(title="FaceSwap API", version="1.0.0")

# -------------------------------
# CORS middleware
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# MongoDB connection
# -------------------------------
MONGODB_URL = os.getenv(
    "MONGODB_URL",
    "mongodb+srv://harilogicgo_db_user:logicgoinfotech@cluster0.dcs1tnb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
client = AsyncIOMotorClient(MONGODB_URL)
database = client.FaceSwap
target_images_collection = database.get_collection("Target_Images")
source_images_collection = database.get_collection("Source_Images")
results_collection = database.get_collection("Results")

# -------------------------------
# Hugging Face Space URL
# -------------------------------
HF_SPACE_URL = "https://logicgoinfotechspaces-faceswap.hf.space/run/predict"

# -------------------------------
# Pydantic models
# -------------------------------
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

# -------------------------------
# Utility function: call HF Space
# -------------------------------
def to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")

async def call_huggingface_space(src_img_data, tgt_img_data):
    """
    Call the Hugging Face Space's /run/predict endpoint with base64 images.
    """
    try:
        # Convert images to base64
        src_b64 = base64.b64encode(src_img_data).decode("utf-8")
        tgt_b64 = base64.b64encode(tgt_img_data).decode("utf-8")
        
        # Gradio /run/predict expects JSON payload like this:
        payload = {
            "data": [src_b64, tgt_b64]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(HF_SPACE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Hugging Face Space error: {error_text}")
                    return None, f"Hugging Face API error: {error_text}"
                
                resp_json = await response.json()
                
                # The output image comes base64-encoded in resp_json["data"][0]
                result_b64 = resp_json.get("data", [])[0]
                if not result_b64:
                    return None, "Empty response from Hugging Face Space"

                result_bytes = base64.b64decode(result_b64)
                return result_bytes, None

    except asyncio.TimeoutError:
        return None, "Request timeout - Hugging Face Space took too long to respond"
    except Exception as e:
        return None, f"Failed to call Hugging Face Space: {str(e)}"

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")
async def root():
    return {"message": "FaceSwap API is running", "status": "healthy"}

# 1. Upload Source Image
@app.post("/api/upload/source", response_model=ImageResponse)
async def upload_source_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        Image.open(io.BytesIO(contents))  # validate image

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

# 2. Get All Target Images
@app.get("/api/target-images", response_model=List[ImageResponse])
async def get_target_images():
    try:
        images = []
        async for image in target_images_collection.find({}, {"image_data": 0}):
            images.append({
                "id": str(image["_id"]),
                "filename": image["filename"],
                "content_type": image["content_type"],
                "uploaded_at": image["uploaded_at"]
            })
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Get Specific Target Image
@app.get("/api/target-images/{image_id}")
async def get_target_image(image_id: str):
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

# 4. Upload Target Image (Admin)
@app.post("/api/upload/target", response_model=ImageResponse)
async def upload_target_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        Image.open(io.BytesIO(contents))  # validate image

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

# 5. Face Swap
@app.post("/api/faceswap", response_model=FaceSwapResponse)
async def perform_face_swap(request: FaceSwapRequest):
    try:
        source_doc = await source_images_collection.find_one({"_id": ObjectId(request.source_image_id)})
        target_doc = await target_images_collection.find_one({"_id": ObjectId(request.target_image_id)})

        if not source_doc:
            raise HTTPException(status_code=404, detail="Source image not found")
        if not target_doc:
            raise HTTPException(status_code=404, detail="Target image not found")

        # Call Hugging Face Space
        result_data, error = await call_huggingface_space(
            source_doc["image_data"], 
            target_doc["image_data"]
        )
        if error:
            raise HTTPException(status_code=500, detail=error)
        if not result_data:
            raise HTTPException(status_code=500, detail="No result returned from HF Space")

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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. Download Result
@app.get("/api/results/{result_id}")
async def download_result(result_id: str):
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

# 7. Get All Results
@app.get("/api/results")
async def get_all_results():
    try:
        results = []
        async for result in results_collection.find({}, {"image_data": 0}):
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

# 8. Health Check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# -------------------------------
# Startup: preload target images
# -------------------------------
async def preload_target_images():
    target_images_dir = "Target_Images"
    if os.path.exists(target_images_dir):
        for filename in os.listdir(target_images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(target_images_dir, filename)
                existing = await target_images_collection.find_one({"filename": filename})
                if not existing:
                    with open(file_path, 'rb') as f:
                        image_data = f.read()
                    await target_images_collection.insert_one({
                        "filename": filename,
                        "image_data": image_data,
                        "content_type": f"image/{filename.split('.')[-1].lower()}",
                        "uploaded_at": datetime.utcnow()
                    })

@app.on_event("startup")
async def startup_event():
    await preload_target_images()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
