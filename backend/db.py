import os
from typing import List

import motor.motor_asyncio
from beanie import init_beanie
from dotenv import load_dotenv

from backend.models.tile import Tile

async def init_db():
    """
    Initializes the database connection and the Beanie ODM.
    This function should be called on application startup.
    """
    # Load environment variables from .env file
    load_dotenv()

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI environment variable not set!")

    # Create a Motor client
    client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)

    # Initialize Beanie with the Tile document
    # The database name will be extracted from the MONGO_URI
    await init_beanie(database=client.get_default_database(), document_models=[Tile])
    print("Database connection initialized.")
