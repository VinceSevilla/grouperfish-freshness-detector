"""
Configuration module for the backend
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# API Configuration
DEBUG = os.getenv("DEBUG", "False") == "True"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
# Railway uses PORT, others might use API_PORT
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8080")))

# CORS Configuration
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    '["http://localhost:5173", "http://localhost:3000", "*"]'
)

# Models Path
MODELS_PATH = os.getenv("MODELS_PATH", "./results")
