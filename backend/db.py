import os
import logging
from urllib.parse import urlparse

import motor.motor_asyncio
from beanie import init_beanie
from dotenv import load_dotenv

from backend.models.tile import Tile
from backend.models.user import User

# Configure logging
logger = logging.getLogger(__name__)

async def init_db():
    """Initializes the database connection and Beanie ODM."""
    logger.info("Initializing database connection...")

    # Load environment variables from .env file
    load_dotenv()

    mongo_uri = os.getenv("MONGO_URI")

    if not mongo_uri:
        logger.critical("MONGO_URI not found in environment variables. Please ensure it is set in your .env file.")
        raise ValueError("MONGO_URI environment variable not set.")

    # Parse the database name from the URI to provide better error handling
    try:
        parsed_uri = urlparse(mongo_uri)
        # db_name = parsed_uri.path.lstrip('/')
        db_name = "tilematcherdb"
        if not db_name:
            raise ValueError("Database name is missing from the MONGO_URI path.")
    except Exception as e:
        logger.critical(
            f"Could not parse database name from MONGO_URI. Error: {e}"
            "Please ensure your URI is in the format: "
            "'mongodb+srv://<user>:<password>@<cluster-url>/<database_name>?retryWrites=true&w=majority'"
        )
        raise

    logger.info(f"Connecting to database: '{db_name}'...")

    try:
        # Create a Motor client
        client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)

        # Ping the server to check the connection before proceeding
        await client.admin.command('ping')
        logger.info("Successfully pinged the database server.")

        # Initialize Beanie with the specific database from the URI
        await init_beanie(database=client[db_name], document_models=[Tile, User])
        logger.info("Database connection initialized and Beanie ODM is ready.")

    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}", exc_info=True)
        raise
