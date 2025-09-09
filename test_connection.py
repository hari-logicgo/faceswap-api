import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def test_connection():
    try:
        client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
        await client.admin.command('ping')
        print("✅ Successfully connected to MongoDB Atlas!")
        
        # Test database operations
        db = client.FaceSwap
        collection = db.TestCollection
        
        # Insert a test document
        result = await collection.insert_one({"test": "connection", "timestamp": "now"})
        print(f"✅ Inserted test document with ID: {result.inserted_id}")
        
        # Read the test document
        document = await collection.find_one({"_id": result.inserted_id})
        print(f"✅ Retrieved document: {document}")
        
        # Clean up
        await collection.delete_one({"_id": result.inserted_id})
        print("✅ Cleaned up test document")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_connection())