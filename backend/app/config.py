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
# Prefer API_PORT, fallback to Railway's PORT, default to 8001
API_PORT = int(os.getenv("API_PORT", os.getenv("PORT", "8001")))

# CORS Configuration
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    '["http://localhost:5173", "http://localhost:3000", "*"]'
)

# Models Path
MODELS_PATH = os.getenv("MODELS_PATH", "./results")
