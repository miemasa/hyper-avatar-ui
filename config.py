from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv(Path(__file__).with_name('.env'))

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
SEEDVC_API_KEY = os.getenv('SEEDVC_API_KEY', '')
API_HOST = os.getenv('API_HOST', 'http://127.0.0.1:8000')

