import asyncio
import logging
import os

# Add the project root to the Python path to allow imports from the 'ml' module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db import init_db

# Configure basic logging to see output from the db module and this script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get a logger instance
logger = logging.getLogger(__name__)

async def main():
    """
    Tests the database connection by calling the init_db function.
    """
    logger.info("--- Running Database Connection Test ---")
    try:
        await init_db()
        logger.info("\n[SUCCESS] Database connection test passed!")
        logger.info("Successfully connected to MongoDB and initialized Beanie.")
    except Exception as e:
        logger.error(f"\n[FAILURE] Database connection test failed: {e}", exc_info=True)
        logger.error("\nPlease check the following:")
        logger.error("1. Your MONGO_URI in the backend/.env file is correct.")
        logger.error("2. Your MongoDB Atlas cluster is active.")
        logger.error("3. The IP address of this machine is whitelisted in MongoDB Atlas.")
        logger.error("4. Your internet connection is working.")
    finally:
        logger.info("--- Test Complete ---")

if __name__ == "__main__":
    # This allows running the async main function from a synchronous script
    asyncio.run(main())
