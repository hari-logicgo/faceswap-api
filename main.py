import os
import uuid
import base64
import aiohttp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from pathlib import Path

# -----------------------------
# MongoDB Setup
# -----------------------------
MONGODB_URL = os.getenv(
    "MONGODB_URL",
    "mongodb+srv://harilogicgo_db_user:logicgoinfotech@cluster0.dcs1tnb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
client = AsyncIOMotorClient(MONGODB_URL)
database = client.FaceSwap
source_images_collection = database.get_collection("Source_Images")
target_images_collection = database.get_collection("Target_Images")
results_collection = database.get_collection("Results")

# -----------------------------
# Temporary Directories
# -----------------------------
BASE_DIR = "./workspace"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# Hugging Face Gradio API
# -----------------------------
HF_GRADIO_URL = "https://logicgoinfotechspaces-faceswap.hf.space/run/predict"
  # replace with your HF Space API

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Face Swap API")

@app.get("/health")
async def health():
    return {"status": "ok"}

async def save_upload_file(upload_file: UploadFile, folder: str) -> str:
    """Save uploaded file to temporary folder and return path"""
    ext = Path(upload_file.filename).suffix
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(folder, unique_name)
    with open(file_path, "wb") as f:
        f.write(await upload_file.read())
    return file_path

@app.post("/source")
async def upload_source(file: UploadFile = File(...)):
    """Upload user source image and store in MongoDB"""
    path = await save_upload_file(file, UPLOAD_DIR)
    doc = {"filename": file.filename, "path": path}
    result = await source_images_collection.insert_one(doc)
    return {"source_id": str(result.inserted_id), "path": path}

@app.post("/faceswap")
async def face_swap(source_id: str, target_id: str):
    """
    Perform face swap:
    - Source image uploaded by user (source_id)
    - Target image already stored in MongoDB (target_id)
    """
    # Fetch images from MongoDB
    source_doc = await source_images_collection.find_one({"_id": ObjectId(source_id)})
    target_doc = await target_images_collection.find_one({"_id": ObjectId(target_id)})

    if not source_doc or not target_doc:
        raise HTTPException(status_code=404, detail="Source or Target image not found")

    # Read image bytes
    with open(source_doc["path"], "rb") as f:
        src_bytes = f.read()
    with open(target_doc["path"], "rb") as f:
        tgt_bytes = f.read()

    # -----------------------------
    # Call HF Space Gradio endpoint
    # -----------------------------
    async with aiohttp.ClientSession() as session:
        payload = {
            "data": [
                {"name": "src", "data": src_bytes.hex()},
                {"name": "tgt", "data": tgt_bytes.hex()}
            ]
        }
        async with session.post(HF_GRADIO_URL, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(status_code=500, detail=f"HF Space call failed: {text}")
            response_json = await resp.json()

    final_img_b64 = response_json.get("data")[0]  # base64 string from HF Space
    if not final_img_b64:
        raise HTTPException(status_code=500, detail="HF Space did not return final image")

    # -----------------------------
    # Save final image locally
    # -----------------------------
    final_bytes = base64.b64decode(final_img_b64)
    final_name = f"result_{uuid.uuid4().hex}.png"
    final_path = os.path.join(RESULT_DIR, final_name)
    with open(final_path, "wb") as f:
        f.write(final_bytes)

    # -----------------------------
    # Store result in MongoDB
    # -----------------------------
    doc = {"source_id": source_id, "target_id": target_id, "path": final_path}
    result = await results_collection.insert_one(doc)

    return {"result_id": str(result.inserted_id), "path": final_path}

@app.get("/download/{result_id}")
async def download_result(result_id: str):
    """Download enhanced face image"""
    doc = await results_collection.find_one({"_id": ObjectId(result_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(doc["path"], media_type="image/png", filename="enhanced_face.png")

@app.get("/target_images")
async def list_target_images():
    """List all target images stored in MongoDB with their IDs"""
    cursor = target_images_collection.find({})
    results = []
    async for doc in cursor:
        results.append({"target_id": str(doc["_id"]), "filename": doc["filename"]})
    return results
