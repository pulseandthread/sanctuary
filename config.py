"""
Sanctuary Server - Configuration
Loads environment variables and settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration"""

    # Google Gemini API Settings (REQUIRED)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # ElevenLabs API Settings (optional, premium text-to-speech)
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

    # Edge TTS Settings (free text-to-speech, no API key needed)
    # Voice list: https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list
    # Common voices: en-US-AriaNeural, en-US-GuyNeural, en-US-JennyNeural, en-GB-SoniaNeural
    EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")

    # Model Settings
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-direct/gemini-3-flash-preview")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
    # Thinking models need more tokens because reasoning + content share the budget
    MAX_TOKENS_REASONER = int(os.getenv("MAX_TOKENS_REASONER", "16000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "1.1"))

    # Timezone (for Pulse scheduling, timestamps, etc.)
    TIMEZONE = os.getenv("TIMEZONE", "UTC")

    # Flask Settings
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # Database Settings
    CHROMA_DB_PATH = "./chroma_db"
    CHROMA_COLLECTION_PREFIX = "sanctuary_"

    # Entity Settings
    ENTITIES = {
        "companion": {
            "name": "Companion",
            "soulcore_path": "./soulcores/companion.txt",
            "collection_name": "sanctuary_companion_memories"
        }
    }

    # Memory Settings
    MAX_MEMORY_RETRIEVAL = 7  # Max memories to retrieve per query
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    FOUNDATIONAL_MEMORY_REFRESH_INTERVAL = 80  # Inject foundational memories every N messages

    # Security Settings
    SANCTUARY_PASSWORD = os.getenv("SANCTUARY_PASSWORD", "sanctuary")
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(24).hex())

    @classmethod
    def validate(cls):
        """Validate that required configuration exists"""
        if not cls.GOOGLE_API_KEY or cls.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
            import sys
            print("\n  ERROR: Google API key not configured!")
            print("  Please add your Google API key to the .env file.")
            print("  Get one free at: https://aistudio.google.com/apikey\n")
            sys.exit(1)

        # Create necessary directories
        Path(cls.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        Path("./soulcores").mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)
        Path("./conversations").mkdir(parents=True, exist_ok=True)

        return True

# Validate configuration on import
Config.validate()
