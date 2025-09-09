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
import asyncio

app = FastAPI(title="FaceSwap API", version="1.0.0")

# -------------------------------
# CORS middleware
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# Hugging Face Space URL (fixed)
# -------------------------------
HF_SPACE_URL = "https://logicgoinfotechspaces-faceswap.hf.space"

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
# Utility: call Hugging Face Space
# -------------------------------
async def call_huggingface_space(src_img_data, tgt_img_data):
    """
    Call HF Space /api/faceswap endpoint with file uploads.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Prepare files for multipart form-data
            files = {
                'src_img': ('source.jpg', src_img_data, 'image/jpeg'),
                'tgt_img': ('target.jpg', tgt_img_data, 'image/jpeg')
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{HF_SPACE_URL}/api/faceswap",
                    data=files,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 404 and attempt < max_retries - 1:
                        # Space might be waking up, wait and retry
                        print(f"Attempt {attempt + 1} failed, retrying in 10 seconds...")
                        await asyncio.sleep(10)
                        continue
                        
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Hugging Face Space error: {error_text}")
                        return None, f"Hugging Face API error: {error_text}"

                    # Get the image data directly from response
                    result_data = await response.read()
                    return result_data, None

        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} timeout, retrying...")
                await asyncio.sleep(10)
                continue
            return None, "Request timeout - Hugging Face Space took too long"
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(10)
                continue
            return None, f"Failed to call Hugging Face Space: {str(e)}"
    
    return None, "All retry attempts failed"

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")
async def root():
    return {"message": "FaceSwap API is running", "status": "healthy"}

@app.post("/api/upload/source", response_model=ImageResponse)
async def upload_source_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        Image.open(io.BytesIO(contents))
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    doc = {
        "filename": file.filename,
        "image_data": contents,
        "content_type": file.content_type,
        "uploaded_at": datetime.utcnow()
    }
    result = await source_images_collection.insert_one(doc)
    return {
        "id": str(result.inserted_id),
        "filename": file.filename,
        "content_type": file.content_type,
        "uploaded_at": doc["uploaded_at"]
    }

@app.get("/api/target-images", response_model=List[ImageResponse])
async def get_target_images():
    images = []
    async for img in target_images_collection.find({}, {"image_data": 0}):
        images.append({
            "id": str(img["_id"]),
            "filename": img["filename"],
            "content_type": img["content_type"],
            "uploaded_at": img["uploaded_at"]
        })
    return images

@app.get("/api/target-images/{image_id}")
async def get_target_image(image_id: str):
    img = await target_images_collection.find_one({"_id": ObjectId(image_id)})
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    return StreamingResponse(io.BytesIO(img["image_data"]), media_type=img["content_type"],
                             headers={"Content-Disposition": f"inline; filename={img['filename']}"})

@app.post("/api/upload/target", response_model=ImageResponse)
async def upload_target_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        Image.open(io.BytesIO(contents))
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    doc = {
        "filename": file.filename,
        "image_data": contents,
        "content_type": file.content_type,
        "uploaded_at": datetime.utcnow()
    }
    result = await target_images_collection.insert_one(doc)
    return {
        "id": str(result.inserted_id),
        "filename": file.filename,
        "content_type": file.content_type,
        "uploaded_at": doc["uploaded_at"]
    }

@app.post("/api/faceswap", response_model=FaceSwapResponse)
async def perform_face_swap(request: FaceSwapRequest):
    source_doc = await source_images_collection.find_one({"_id": ObjectId(request.source_image_id)})
    target_doc = await target_images_collection.find_one({"_id": ObjectId(request.target_image_id)})

    if not source_doc:
        raise HTTPException(status_code=404, detail="Source image not found")
    if not target_doc:
        raise HTTPException(status_code=404, detail="Target image not found")

    result_data, error = await call_huggingface_space(source_doc["image_data"], target_doc["image_data"])
    if error:
        raise HTTPException(status_code=500, detail=error)

    doc = {
        "filename": f"faceswap_result_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png",
        "image_data": result_data,
        "content_type": "image/png",
        "source_image_id": ObjectId(request.source_image_id),
        "target_image_id": ObjectId(request.target_image_id),
        "created_at": datetime.utcnow()
    }
    result = await results_collection.insert_one(doc)
    return {"result_id": str(result.inserted_id), "message": "Face swap completed successfully"}

@app.get("/api/results/{result_id}")
async def download_result(result_id: str):
    result = await results_collection.find_one({"_id": ObjectId(result_id)})
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return StreamingResponse(io.BytesIO(result["image_data"]),
                             media_type=result["content_type"],
                             headers={"Content-Disposition": f"attachment; filename={result['filename']}"})

@app.get("/api/results")
async def get_all_results():
    results = []
    async for res in results_collection.find({}, {"image_data": 0}):
        results.append({
            "id": str(res["_id"]),
            "filename": res["filename"],
            "content_type": res["content_type"],
            "source_image_id": str(res["source_image_id"]),
            "target_image_id": str(res["target_image_id"]),
            "created_at": res["created_at"]
        })
    return results

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
                exists = await target_images_collection.find_one({"filename": filename})
                if not exists:
                    with open(file_path, 'rb') as f:
                        img_data = f.read()
                    await target_images_collection.insert_one({
                        "filename": filename,
                        "image_data": img_data,
                        "content_type": f"image/{filename.split('.')[-1].lower()}",
                        "uploaded_at": datetime.utcnow()
                    })

@app.on_event("startup")
async def startup_event():
    await preload_target_images()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
