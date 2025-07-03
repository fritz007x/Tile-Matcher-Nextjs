import certifi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# IMPORTANT: Replace <db_password> with your actual database password.
# NOTE: This URI is missing the database name, which is required for the main application to work.
# Example with database name: "mongodb+srv://user:pass@cluster.mongodb.net/my_database?..."
uri = "mongodb+srv://fritz007x:b013A8cyRbYibyQU@tilematchercluster.yef2uhh.mongodb.net/?retryWrites=true&w=majority&appName=TileMatcherCluster"

# Create a new client and connect to the server
# Use certifi to provide the latest CA certificates for SSL validation
ca = certifi.where()
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=ca)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
