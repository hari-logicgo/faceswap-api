import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

async def preload_target_images():
    try:
        client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
        database = client.FaceSwap
        collection = database.Target_Images
        
        target_images_dir = "Target_Images"
        if os.path.exists(target_images_dir):
            for filename in os.listdir(target_images_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(target_images_dir, filename)
                    
                    # Check if image already exists
                    existing = await collection.find_one({"filename": filename})
                    if not existing:
                        with open(file_path, 'rb') as f:
                            image_data = f.read()
                        
                        image_doc = {
                            "filename": filename,
                            "image_data": image_data,
                            "content_type": f"image/{filename.split('.')[-1].lower()}",
                            "uploaded_at": datetime.utcnow()
                        }
                        
                        result = await collection.insert_one(image_doc)
                        print(f"✅ Uploaded {filename} to MongoDB Atlas")
                    else:
                        print(f"⚠️  {filename} already exists in database")
        
        print("✅ Preloading complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(preload_target_images())