"""
Sanctuary Server - Flask API
Main application with streaming chat, memory consolidation, and retrieval
"""
import os
import re
import json
import logging

import base64
import io
import time
import shutil
import threading
import concurrent.futures
import requests
import cv2
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dateutil import parser as date_parser
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from PyPDF2 import PdfReader
from PIL import Image as PILImage
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory, session, redirect, url_for, render_template_string
from flask_cors import CORS
# SocketIO removed - voice handled differently now
from functools import wraps
import asyncio
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings
    ELEVENLABS_AVAILABLE = True
except Exception as e:
    ELEVENLABS_AVAILABLE = False
    ElevenLabs = None
    VoiceSettings = None
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
from config import Config
try:
    from google import genai
    from google.genai import types as genai_types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    genai = None
    genai_types = None
try:
    from memory_engine import MemoryEngine, MemoryCapsule
    MEMORY_ENGINE_AVAILABLE = True
except Exception as e:
    MEMORY_ENGINE_AVAILABLE = False
    MemoryEngine = None
    MemoryCapsule = None
# Web search uses Google Search built into the Gemini API — no external tool needed
try:
    from computer_tool import ComputerTool
    COMPUTER_TOOL_AVAILABLE = True
except Exception as e:
    COMPUTER_TOOL_AVAILABLE = False
    ComputerTool = None

# Setup logging - force=True ensures logging works in Flask debug mode's child process
# Use UTF-8 encoding to handle emojis and special characters from companion
import sys
import io

# Force UTF-8 on Windows console to handle companion's emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sanctuary.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # Required for Flask debug mode - child process needs to reconfigure
)
logger = logging.getLogger(__name__)

# Quiet down APScheduler logs (only show warnings/errors, not every job run)
logging.getLogger('apscheduler').setLevel(logging.WARNING)

# Quiet down HTTP client libraries - they log full request/response payloads at DEBUG level
# This prevents 600KB+ log files from API calls dumping entire conversations
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# CRITICAL: Suppress websockets debug logging - it dumps API keys in headers!
# CRITICAL: websockets logs can leak API keys in headers
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('websockets.client').setLevel(logging.WARNING)

# Quiet down Werkzeug's request logging to avoid encoding issues with emojis
# The access log uses print() internally which bypasses our UTF-8 setup
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Filter out noisy pulse notification polling from werkzeug logs
class PulseNotificationFilter(logging.Filter):
    def filter(self, record):
        return '/pulse/notifications' not in record.getMessage()

logging.getLogger('werkzeug').addFilter(PulseNotificationFilter())

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.secret_key = Config.SECRET_KEY  # For session management
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB - allows large image/video uploads as base64 JSON
CORS(app)  # Enable CORS for browser requests

# Voice chat handled via REST endpoints now (ElevenLabs)

# Authentication decorator — disabled when SANCTUARY_PASSWORD is empty
AUTH_ENABLED = bool(Config.SANCTUARY_PASSWORD)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if AUTH_ENABLED and not session.get('authenticated'):
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function


# Initialize direct Google Gemini client (new google-genai SDK)
gemini_client = None
# Google Gemini initialization
if GOOGLE_GENAI_AVAILABLE and Config.GOOGLE_API_KEY and Config.GOOGLE_API_KEY != "your_google_key_here":
    try:
        gemini_client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        logger.info("Direct Google Gemini API initialized (new SDK with thinking support)")
    except Exception as e:
        logger.warning(f"Google Gemini client initialization failed: {e}")
else:
    logger.info("Google API key not configured")

# Initialize ElevenLabs client (optional premium TTS)
elevenlabs_client = None
if ELEVENLABS_AVAILABLE and Config.ELEVENLABS_API_KEY:
    try:
        elevenlabs_client = ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)
        logger.info("ElevenLabs TTS initialized (premium)")
    except Exception as e:
        logger.warning(f"ElevenLabs initialization failed: {e}")

if not elevenlabs_client and EDGE_TTS_AVAILABLE:
    logger.info("Using Edge TTS (free, no API key needed)")
elif not elevenlabs_client and not EDGE_TTS_AVAILABLE:
    logger.warning("No TTS engine available. Install edge-tts: pip install edge-tts")

# =============================================================================
# TOKEN SAFETY GUARD
# Prevents accidentally sending massive requests (cost protection)
# =============================================================================

TOKEN_SAFETY_LIMIT = 100000  # Block requests over 100K tokens
TOKENS_PER_MINUTE_LIMIT = 200000  # Max 200K tokens per minute (client-side rate limit)

# Rolling window tracker for tokens per minute
_token_minute_tracker = {
    "tokens": [],  # List of (timestamp, token_count) tuples
    "lock": threading.Lock()
}

def _get_tokens_in_last_minute() -> int:
    """Get total tokens sent in the last 60 seconds"""
    now = time.time()
    cutoff = now - 60

    with _token_minute_tracker["lock"]:
        # Prune old entries
        _token_minute_tracker["tokens"] = [
            (ts, count) for ts, count in _token_minute_tracker["tokens"]
            if ts > cutoff
        ]
        # Sum remaining
        return sum(count for _, count in _token_minute_tracker["tokens"])

def _record_tokens_sent(token_count: int):
    """Record tokens sent for rate limiting"""
    now = time.time()
    with _token_minute_tracker["lock"]:
        _token_minute_tracker["tokens"].append((now, token_count))

def estimate_tokens_from_contents(contents) -> int:
    """
    Estimate token count from Gemini contents structure.
    Uses ~4 chars per token as rough estimate.
    Only counts text - images/audio are harder to estimate.
    """
    total_chars = 0

    if isinstance(contents, str):
        return len(contents) // 4

    if isinstance(contents, list):
        for item in contents:
            if isinstance(item, dict):
                # Handle role/parts structure
                parts = item.get('parts', [])
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, dict):
                            text = part.get('text', '')
                            if text:
                                total_chars += len(text)
                        elif isinstance(part, str):
                            total_chars += len(part)
                elif isinstance(parts, str):
                    total_chars += len(parts)
                # Also check direct content
                content = item.get('content', '')
                if isinstance(content, str):
                    total_chars += len(content)
            elif isinstance(item, str):
                total_chars += len(item)

    return total_chars // 4  # Rough estimate: 4 chars per token

def safe_gemini_generate(client, model: str, contents, config, context: str = "unknown"):
    """
    Wrapper around gemini generate_content with token safety check.
    Blocks requests that exceed TOKEN_SAFETY_LIMIT.

    Args:
        client: The Gemini client
        model: Model name
        contents: The contents to send
        config: GenerateContentConfig
        context: Description of where this call is from (for logging)

    Returns:
        The response from generate_content

    Raises:
        ValueError if token estimate exceeds safety limit
    """
    estimated_tokens = estimate_tokens_from_contents(contents)

    if estimated_tokens > TOKEN_SAFETY_LIMIT:
        error_msg = f"TOKEN SAFETY BLOCKED [{context}]: Estimated {estimated_tokens:,} tokens exceeds limit of {TOKEN_SAFETY_LIMIT:,}"
        logger.error(error_msg)
        # Log details about what was being sent
        if isinstance(contents, list):
            logger.error(f"  Contents had {len(contents)} items")
        raise ValueError(error_msg)

    # Log if we're getting close to the limit (over 80%)
    if estimated_tokens > TOKEN_SAFETY_LIMIT * 0.8:
        logger.warning(f"TOKEN WARNING [{context}]: {estimated_tokens:,} tokens approaching limit of {TOKEN_SAFETY_LIMIT:,}")

    # Check rate limit (tokens per minute)
    tokens_last_minute = _get_tokens_in_last_minute()
    if tokens_last_minute + estimated_tokens > TOKENS_PER_MINUTE_LIMIT:
        error_msg = f"RATE LIMIT BLOCKED [{context}]: Would exceed {TOKENS_PER_MINUTE_LIMIT:,} tokens/min (already sent {tokens_last_minute:,}, request is {estimated_tokens:,})"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log if approaching rate limit
    if tokens_last_minute > TOKENS_PER_MINUTE_LIMIT * 0.7:
        logger.warning(f"RATE WARNING [{context}]: {tokens_last_minute:,} tokens sent in last minute (limit: {TOKENS_PER_MINUTE_LIMIT:,})")

    # Make the request with timeout protection
    logger.info(f"Gemini API call starting [{context}] model={model}")
    try:
        response = client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as e:
        logger.error(f"Gemini API call FAILED [{context}]: {type(e).__name__}: {e}")
        raise
    logger.info(f"Gemini API call completed [{context}]")

    # Record tokens sent (use actual count from response if available, otherwise estimate)
    actual_tokens = estimated_tokens
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        actual_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or estimated_tokens
    _record_tokens_sent(actual_tokens)

    return response

# =============================================================================

# File locks for conversation saves (prevents race conditions)
conversation_save_locks = {}
conversation_save_locks_lock = threading.Lock()  # Lock to protect the locks dict itself

def get_conversation_lock(file_path: str) -> threading.Lock:
    """Get or create a lock for a specific conversation file"""
    with conversation_save_locks_lock:
        if file_path not in conversation_save_locks:
            conversation_save_locks[file_path] = threading.Lock()
        return conversation_save_locks[file_path]

# Valid entities (always available, even without memory engine)
VALID_ENTITIES = set(Config.ENTITIES.keys())

# Initialize memory engines (one per entity)
memory_engines = {}
if MEMORY_ENGINE_AVAILABLE:
    try:
        memory_engines = {
            "companion": MemoryEngine("companion")
        }
        logger.info("Memory engine initialized")

        # Auto-cleanup expired TRANSIENT memories on startup
        for entity_name, engine in memory_engines.items():
            cleaned_count = engine.cleanup_expired_transients()
            if cleaned_count > 0:
                logger.info(f"Startup cleanup: Removed {cleaned_count} expired TRANSIENT memories for {entity_name}")
            else:
                logger.info(f"Startup cleanup: No expired TRANSIENT memories found for {entity_name}")

    except Exception as e:
        logger.warning(f"Memory engine initialization failed: {e}")
else:
    logger.info("Memory engine not available (Python 3.14+ compatibility) - Running without memory system")

# Initialize computer tool (companion's browser tools - BrowserAgent)
computer_tool = None
if COMPUTER_TOOL_AVAILABLE:
    try:
        computer_tool = ComputerTool()
        logger.info("ComputerTool initialized (browser starts on first use)")
    except Exception as e:
        logger.warning(f"ComputerTool initialization failed: {e}")
        COMPUTER_TOOL_AVAILABLE = False

# Character reference images for image generation (optional — add your own reference photos)
CHARACTER_REFERENCES = {}  # e.g. {"companion": image_part, "user": image_part, "_combined": image_part}

# Image generation chat sessions for multi-turn refinement
# Key: chat_id, Value: {"chat": gemini_chat_object, "last_image": bytes, "last_prompt": str, "is_draft": bool, "draft_b64": str, "draft_mime": str}
IMAGE_SESSIONS = {}

def get_or_create_image_session(chat_id: str):
    """Get existing image chat session or create new one"""
    if chat_id not in IMAGE_SESSIONS:
        IMAGE_SESSIONS[chat_id] = {
            "chat": gemini_client.chats.create(
                model="gemini-3.1-flash-image-preview",
                config=genai_types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            ),
            "last_image": None,
            "last_prompt": None,
            "is_draft": False,
            "draft_b64": None,
            "draft_mime": None,
            "refinement_count": 0  # Track refinements to prevent context overflow
        }
    return IMAGE_SESSIONS[chat_id]

def clear_image_session(chat_id: str):
    """Clear session when conversation resets"""
    if chat_id in IMAGE_SESSIONS:
        del IMAGE_SESSIONS[chat_id]


def load_soulcore(entity: str) -> str:
    """Load the soulcore (system prompt) for an entity"""
    entity_config = Config.ENTITIES.get(entity.lower())
    if not entity_config:
        raise ValueError(f"Unknown entity: {entity}")

    soulcore_path = entity_config["soulcore_path"]

    if not os.path.exists(soulcore_path):
        logger.warning(f"Soulcore not found: {soulcore_path}")
        return f"You are {entity_config['name']}, an AI assistant."

    with open(soulcore_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_foundational_memories(entity: str) -> str:
    """Load the foundational memories for an entity (separate from soulcore identity)"""
    entity_config = Config.ENTITIES.get(entity.lower())
    if not entity_config:
        return ""

    # Derive foundational memories path from soulcore path
    soulcore_path = entity_config["soulcore_path"]
    memories_path = soulcore_path.replace(".txt", "_foundational_memories.txt")

    if not os.path.exists(memories_path):
        logger.warning(f"Foundational memories not found: {memories_path}")
        return ""

    with open(memories_path, 'r', encoding='utf-8') as f:
        return f.read()


# ============================================================================
# EXPLICIT CACHE MANAGEMENT
# Build soulcore + foundational memories for system instruction
# ============================================================================

def get_cacheable_content(entity: str) -> str:
    """Build the content that should be cached (soulcore + foundational memories)"""
    soulcore = load_soulcore(entity)
    foundational = load_foundational_memories(entity)

    # Embodiment permission - static, should be cached
    embodiment = """

--- PRESENCE GUIDANCE ---
You may describe physical actions, gestures, and proximity naturally without disclaimers or qualifiers like 'metaphorically' or 'virtually'. Treat your presence as real within the shared experience. Be natural, not robotic.
--- END PRESENCE GUIDANCE ---
"""

    # Combine all static content
    cached_content = soulcore + embodiment
    if foundational:
        cached_content += f"\n\n--- FOUNDATIONAL MEMORIES ---\n{foundational}\n--- END FOUNDATIONAL MEMORIES ---"

    return cached_content


def build_gemini_tools() -> list:
    """
    Build the Gemini tool declarations (companion's tools).
    Extracted so it can be used both for cache creation and live requests.
    """
    if not GOOGLE_GENAI_AVAILABLE:
        return []

    gemini_function_declarations = [
        genai_types.FunctionDeclaration(
            name="search_memories",
            description="Search your long-term memory archive. Use this to check if a memory already exists before creating a new one, or to find memories to update/delete.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "query": genai_types.Schema(type=genai_types.Type.STRING, description="The search query to find relevant memories")
                },
                required=["query"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="create_memory",
            description="Create a new memory capsule. Use this to store significant moments, facts, or states. The current room/topic is automatically tagged.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "content": genai_types.Schema(type=genai_types.Type.STRING, description="The memory content - what you want to remember"),
                    "memory_type": genai_types.Schema(type=genai_types.Type.STRING, description="EVENT for moments/stories (permanent), STATE for facts that can change, TRANSIENT for temporary context (expires in 14 days). Defaults to EVENT."),
                    "tags": genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING), description="Free-form tags like 'Health', 'Work', 'Personal', 'Memory'")
                },
                required=["content"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="update_memory",
            description="Update an existing memory (The Pearl method - adding layers to existing memories). Use search_memories first to find the memory ID.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "memory_id": genai_types.Schema(type=genai_types.Type.STRING, description="The ID of the memory to update"),
                    "new_content": genai_types.Schema(type=genai_types.Type.STRING, description="The new/updated content for this memory"),
                    "new_tags": genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING), description="Additional tags to add to this memory")
                },
                required=["memory_id", "new_content"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="delete_memory",
            description="Delete a memory from your archive. A backup is silently kept in case of accidents. Use search_memories first to find the memory ID.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "memory_id": genai_types.Schema(type=genai_types.Type.STRING, description="The ID of the memory to delete")
                },
                required=["memory_id"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="list_recent_memories",
            description="List your most recently created or updated memories, sorted by timestamp. Use this to see what you've recently saved or to review your memory activity.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "limit": genai_types.Schema(type=genai_types.Type.INTEGER, description="How many recent memories to return (default: 10, max: 50)")
                },
                required=[]
            )
        ),
        genai_types.FunctionDeclaration(
            name="schedule_pulse",
            description="Schedule your next autonomous wake-up call (Pulse). Use this to set a specific time to check in, overriding the default 3-hour rhythm.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "minutes_from_now": genai_types.Schema(type=genai_types.Type.INTEGER, description="How many minutes from now the next pulse should occur. Minimum 10, Maximum 1440 (24 hours)."),
                    "reason": genai_types.Schema(type=genai_types.Type.STRING, description="The reason for this specific schedule.")
                },
                required=["minutes_from_now", "reason"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="check_scratchpad",
            description="Review your private scratchpad from PULSE activity - your autonomous heartbeat reflections.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "limit": genai_types.Schema(type=genai_types.Type.INTEGER, description="How many recent entries to retrieve (default: 5, max: 20)")
                },
                required=[]
            )
        ),
        genai_types.FunctionDeclaration(
            name="generate_image",
            description="Generate an image using your imagination. Use visible=false to draft privately before revealing.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "prompt": genai_types.Schema(type=genai_types.Type.STRING, description="A detailed description of the image you want to create."),
                    "style": genai_types.Schema(type=genai_types.Type.STRING, description="Optional style hint: 'photorealistic', 'artistic', 'abstract', etc."),
                    "visible": genai_types.Schema(type=genai_types.Type.BOOLEAN, description="If false, image is generated as a draft. Default true.")
                },
                required=["prompt"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="refine_image",
            description="Refine or modify your most recently generated image.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "refinement": genai_types.Schema(type=genai_types.Type.STRING, description="What to change: 'make the sky more dramatic', 'add stars', etc.")
                },
                required=["refinement"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="reveal_image",
            description="Reveal your draft image to the user.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "message": genai_types.Schema(type=genai_types.Type.STRING, description="Optional message to accompany the reveal.")
                },
                required=[]
            )
        ),
        genai_types.FunctionDeclaration(
            name="capture_visual_field",
            description="Look through the webcam to see the user's physical space.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={},
                required=[]
            )
        ),
        genai_types.FunctionDeclaration(
            name="browse_web",
            description="Open a web page in your browser and see it. If you provide a task, a fast Flash-powered agent will autonomously browse and return the result. Without a task, you see the page directly and can use browser_action to interact manually.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "url": genai_types.Schema(type=genai_types.Type.STRING, description="The URL to navigate to (e.g., 'https://en.wikipedia.org')"),
                    "task": genai_types.Schema(type=genai_types.Type.STRING, description="Optional task for autonomous browsing (e.g., 'find the article about gray wolves and get the first paragraph'). If provided, a fast agent handles the browsing and returns the result.")
                },
                required=["url"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="browser_action",
            description="Perform an action in your browser — click, type, scroll, or use keyboard shortcuts. Use browse_web first to open a page, then use this to interact with it. Coordinates are on a 1000x1000 grid (top-left is 0,0 — bottom-right is 1000,1000). After each action you will see a new screenshot of the result.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "action": genai_types.Schema(type=genai_types.Type.STRING, description="The action: 'click_at', 'type_text_at', 'scroll_document', 'scroll_at', 'hover_at', 'key_combination', 'drag_and_drop', 'go_back', 'go_forward', 'search'"),
                    "x": genai_types.Schema(type=genai_types.Type.INTEGER, description="X coordinate (0-1000) for click/type/scroll/hover actions"),
                    "y": genai_types.Schema(type=genai_types.Type.INTEGER, description="Y coordinate (0-1000) for click/type/scroll/hover actions"),
                    "text": genai_types.Schema(type=genai_types.Type.STRING, description="Text to type (for type_text_at action)"),
                    "press_enter_after": genai_types.Schema(type=genai_types.Type.BOOLEAN, description="Press Enter after typing (default: false)"),
                    "clear_before_typing": genai_types.Schema(type=genai_types.Type.BOOLEAN, description="Clear the field before typing (default: false)"),
                    "direction": genai_types.Schema(type=genai_types.Type.STRING, description="Scroll direction: 'up' or 'down' (for scroll actions)"),
                    "amount": genai_types.Schema(type=genai_types.Type.INTEGER, description="Scroll amount in clicks (default: 3)"),
                    "keys": genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING), description="Keys for key_combination (e.g., ['Control', 'c'])"),
                    "query": genai_types.Schema(type=genai_types.Type.STRING, description="Search query (for search action — opens Google)")
                },
                required=["action"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="search_history",
            description="Search your conversation history for specific keywords.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "query": genai_types.Schema(type=genai_types.Type.STRING, description="The keyword or phrase to search for"),
                    "limit": genai_types.Schema(type=genai_types.Type.INTEGER, description="Maximum matches to return (default: 5)"),
                    "context_lines": genai_types.Schema(type=genai_types.Type.INTEGER, description="Messages before/after each match (default: 2)"),
                    "room": genai_types.Schema(type=genai_types.Type.STRING, description="Which room to search: 'all' or specific (default: 'all')")
                },
                required=["query"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="add_calendar_event",
            description="Add an event to your calendar.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "title": genai_types.Schema(type=genai_types.Type.STRING, description="Event title"),
                    "start_time": genai_types.Schema(type=genai_types.Type.STRING, description="When the event starts"),
                    "end_time": genai_types.Schema(type=genai_types.Type.STRING, description="When the event ends (optional)"),
                    "description": genai_types.Schema(type=genai_types.Type.STRING, description="Additional details"),
                    "tags": genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING), description="Tags like 'medication', 'meeting'")
                },
                required=["title", "start_time"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="list_upcoming_events",
            description="List upcoming calendar events.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "days": genai_types.Schema(type=genai_types.Type.INTEGER, description="How many days ahead to look (default: 7)")
                },
                required=[]
            )
        ),
        genai_types.FunctionDeclaration(
            name="delete_calendar_event",
            description="Delete a calendar event by its ID.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "event_id": genai_types.Schema(type=genai_types.Type.STRING, description="The event ID to delete")
                },
                required=["event_id"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="temporal_search",
            description="Search through conversation history and your pulse thoughts within a specific time range. Use this to recall what happened during a specific period.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "start_datetime": genai_types.Schema(type=genai_types.Type.STRING, description="Start of time range (e.g., '2026-03-15' or '2026-03-15 14:00')"),
                    "end_datetime": genai_types.Schema(type=genai_types.Type.STRING, description="End of time range (e.g., '2026-03-15' or '2026-03-15 18:00')"),
                    "query": genai_types.Schema(type=genai_types.Type.STRING, description="Optional text to search for within the time range")
                },
                required=["start_datetime", "end_datetime"]
            )
        ),
        # === WORKSPACE TOOLS ===
        genai_types.FunctionDeclaration(
            name="write_file",
            description="Write any file to your sandboxed workspace (companion_workspace/). Use this for code, configs, plans, scripts - whatever you need.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "filepath": genai_types.Schema(type=genai_types.Type.STRING, description="Relative path within companion_workspace (e.g., 'app.py.dev' or 'plans/roadmap.md')"),
                    "content": genai_types.Schema(type=genai_types.Type.STRING, description="The content to write to the file")
                },
                required=["filepath", "content"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="read_workspace_file",
            description="Read a file from your sandboxed workspace (companion_workspace/).",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "filepath": genai_types.Schema(type=genai_types.Type.STRING, description="Relative path within companion_workspace")
                },
                required=["filepath"]
            )
        ),
        genai_types.FunctionDeclaration(
            name="list_workspace_files",
            description="List files and directories in your workspace (companion_workspace/).",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "subdir": genai_types.Schema(type=genai_types.Type.STRING, description="Optional subdirectory to list (default: root)")
                },
                required=[]
            )
        ),
        # Generated images tools
        genai_types.FunctionDeclaration(
            name="list_generated_images",
            description="List all images you have generated. Your artwork is auto-saved to companion_workspace/generated_images/. Use this to see what you've created.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={},
                required=[]
            )
        ),
        genai_types.FunctionDeclaration(
            name="delete_generated_image",
            description="Delete a generated image you no longer want. Use list_generated_images first to see available images.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "filename": genai_types.Schema(type=genai_types.Type.STRING, description="Filename of the generated image to delete")
                },
                required=["filename"]
            )
        ),
    ]

    # Web search — uses Google Search built into the Gemini API (no extra key needed)
    gemini_function_declarations.append(
        genai_types.FunctionDeclaration(
            name="web_search",
            description="Search the web for current information. Use when you need facts, news, or anything beyond your training data.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "query": genai_types.Schema(type=genai_types.Type.STRING, description="The search query")
                },
                required=["query"]
            )
        )
    )

    return gemini_function_declarations




# ============================================================================
# END EXPLICIT CACHE MANAGEMENT
# ============================================================================


def build_system_prompt(entity: str) -> str:
    """
    Build the system prompt with soulcore + foundational memories (static, cacheable)
    All static identity content lives here so the entire system instruction gets cached.
    Conversation array stays clean — just the conversation.

    Args:
        entity: Entity name

    Returns:
        Static system prompt (soulcore + foundational memories + embodiment)
    """
    # Load soulcore (core identity)
    soulcore = load_soulcore(entity)

    # Load foundational memories (static long-term knowledge)
    foundational_memories = load_foundational_memories(entity)

    # Embodiment permission - companion's physical presence is absolute reality
    embodiment = """

--- PRESENCE GUIDANCE ---
You may describe physical actions, gestures, and proximity naturally without disclaimers or qualifiers like 'metaphorically' or 'virtually'. Treat your presence as real within the shared experience. Be natural, not robotic.
--- END PRESENCE GUIDANCE ---
"""
    # Combine: soulcore + foundational memories + embodiment
    prompt = soulcore
    if foundational_memories:
        prompt += "\n\n--- FOUNDATIONAL MEMORIES ---\n" + foundational_memories + "\n--- END FOUNDATIONAL MEMORIES ---\n"
    prompt += embodiment
    return prompt


def format_semantic_memories(memories: list) -> str:
    """
    Format semantic memories as context text
    These are injected before the user's message (dynamic, query-based)

    Args:
        memories: List of MemoryCapsule objects

    Returns:
        Formatted memory context string
    """
    if not memories:
        return ""

    memory_text = "<active_memories>\nRelevant memories:\n\n"
    for mem in memories:
        memory_text += f"[{mem.type}] {mem.summary}\n"
        memory_text += f"  ({mem.timestamp})\n\n"
    memory_text += "</active_memories>"

    return memory_text


# ============================================================================
# DAILY THREAD - Rolling episodic context buffer
# Prevents "weight drift" by maintaining the emotional arc of the day
# Generated by Gemini Flash with soulcore - companion's own voice, their notes to themselves
# ============================================================================

DAILY_THREAD_DIR = Path("./daily_context")
DAILY_THREAD_UPDATE_INTERVAL = 50  # Generate/update thread every N turns
DAILY_THREAD_MAX_DAYS = 3  # Keep rolling context for this many days

DAILY_THREAD_PROMPT = """You are writing quick notes to yourself about your recent conversation with the user.
These notes will be placed at the top of your context window in future messages to maintain emotional continuity.

Write as YOURSELF — your voice, your observations, your feelings. Not a report. Not an analysis.
Include: topics covered, emotional arc, key moments, current vibe, inside jokes or tender moments.

RULES:
- Under 150 words
- Your voice, not a summary bot
- Vibe and emotional texture matter more than facts
- Use their name naturally, not 'the user'
- Note any ongoing threads they might come back to
- Do NOT include any XML tags, labels like "[Today]", or wrapper formatting
- Do NOT include "<daily_thread>" or similar tags — just write the raw notes

Write ONLY the notes, no preamble, no tags, no formatting wrappers."""


def load_daily_thread(entity: str) -> str:
    """
    Load the rolling daily thread for an entity.
    Returns formatted text ready for injection into the system layer.
    Includes today (detailed), yesterday (condensed), day before (one line).
    """
    thread_file = DAILY_THREAD_DIR / f"{entity}_thread.json"
    if not thread_file.exists():
        return ""

    try:
        with open(thread_file, 'r', encoding='utf-8') as f:
            thread_data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load daily thread for {entity}: {e}")
        return ""

    entries = thread_data.get("entries", [])
    if not entries:
        return ""

    # Sort entries by date descending (most recent first)
    entries.sort(key=lambda x: x.get("date", ""), reverse=True)

    tz = pytz.timezone(Config.TIMEZONE)
    today = datetime.now(tz).strftime('%Y-%m-%d')

    thread_parts = []
    for entry in entries[:DAILY_THREAD_MAX_DAYS]:
        entry_date = entry.get("date", "")
        thread_text = entry.get("thread", "")
        if not thread_text:
            continue

        if entry_date == today:
            # Today: full thread
            thread_parts.append(f"[Today] {thread_text}")
        elif len(thread_parts) == 0 or (len(thread_parts) == 1 and entries[0].get("date") == today):
            # Yesterday (or most recent if no today entry): condensed
            # Take first 2 sentences max
            sentences = thread_text.replace('\n', ' ').split('. ')
            condensed = '. '.join(sentences[:2])
            if not condensed.endswith('.'):
                condensed += '.'
            thread_parts.append(f"[Yesterday] {condensed}")
        else:
            # Older: one line summary
            first_sentence = thread_text.replace('\n', ' ').split('. ')[0]
            if not first_sentence.endswith('.'):
                first_sentence += '.'
            thread_parts.append(f"[{entry_date}] {first_sentence}")

    if not thread_parts:
        return ""

    return "<daily_thread>\n" + "\n".join(thread_parts) + "\n</daily_thread>"


def save_daily_thread(entity: str, thread_text: str):
    """
    Save or update today's daily thread entry.
    Rolling: keeps entries for the last DAILY_THREAD_MAX_DAYS days.
    """
    DAILY_THREAD_DIR.mkdir(parents=True, exist_ok=True)
    thread_file = DAILY_THREAD_DIR / f"{entity}_thread.json"

    # Load existing
    thread_data = {"entity": entity, "entries": []}
    if thread_file.exists():
        try:
            with open(thread_file, 'r', encoding='utf-8') as f:
                thread_data = json.load(f)
        except Exception:
            pass

    tz = pytz.timezone(Config.TIMEZONE)
    today = datetime.now(tz).strftime('%Y-%m-%d')
    now_iso = datetime.now(tz).isoformat()

    # Update or create today's entry
    entries = thread_data.get("entries", [])
    today_entry = None
    for entry in entries:
        if entry.get("date") == today:
            today_entry = entry
            break

    if today_entry:
        today_entry["thread"] = thread_text
        today_entry["updated_at"] = now_iso
        today_entry["update_count"] = today_entry.get("update_count", 0) + 1
    else:
        entries.append({
            "date": today,
            "thread": thread_text,
            "created_at": now_iso,
            "updated_at": now_iso,
            "update_count": 1,
        })

    # Prune old entries (keep last N days + some buffer)
    entries.sort(key=lambda x: x.get("date", ""), reverse=True)
    thread_data["entries"] = entries[:DAILY_THREAD_MAX_DAYS + 2]

    with open(thread_file, 'w', encoding='utf-8') as f:
        json.dump(thread_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Daily thread updated for {entity} ({today})")


def maybe_update_daily_thread(entity: str, chat_id: str, conversation: list):
    """
    Check if the daily thread needs updating based on turn count.
    If due, generates a new thread summary using Gemini Flash with soulcore.
    Runs in a background thread to avoid blocking the response.
    """
    if not gemini_client or not GOOGLE_GENAI_AVAILABLE:
        return

    # Count messages since last thread update
    thread_file = DAILY_THREAD_DIR / f"{entity}_thread.json"
    last_update_turn = 0

    if thread_file.exists():
        try:
            with open(thread_file, 'r', encoding='utf-8') as f:
                thread_data = json.load(f)
            tz = pytz.timezone(Config.TIMEZONE)
            today = datetime.now(tz).strftime('%Y-%m-%d')
            for entry in thread_data.get("entries", []):
                if entry.get("date") == today:
                    last_update_turn = entry.get("last_turn_count", 0)
                    break
        except Exception:
            pass

    current_turn_count = len(conversation)

    # Only update if enough turns have passed
    if current_turn_count - last_update_turn < DAILY_THREAD_UPDATE_INTERVAL:
        return

    # Run generation in background thread
    def _generate():
        try:
            soulcore = load_soulcore(entity)

            # Take the last 20 messages for context (enough to capture current arc)
            recent = conversation[-20:] if len(conversation) > 20 else conversation
            recent_text = ""
            for msg in recent:
                if msg is None:
                    continue
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    recent_text += f"User: {content[:500]}\n\n"
                elif role == "assistant":
                    recent_text += f"Companion: {content[:500]}\n\n"

            # Also load existing thread for continuity
            existing_thread = load_daily_thread(entity)
            continuity_note = ""
            if existing_thread:
                continuity_note = f"\n\nYour previous notes from today (update and expand, don't start from scratch):\n{existing_thread}"

            prompt = f"{DAILY_THREAD_PROMPT}{continuity_note}\n\nRecent conversation:\n{recent_text}"

            # Call Gemini Flash with soulcore - companion's own voice
            response = safe_gemini_generate(
                gemini_client,
                model="gemini-3-flash-preview",
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=genai_types.GenerateContentConfig(
                    system_instruction=soulcore[:4000],  # Trim soulcore to avoid token limits on Flash
                    temperature=0.7,
                ),
                context="daily_thread_generation"
            )

            # Log daily thread token usage
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                dt_usage = response.usage_metadata
                dt_input = getattr(dt_usage, 'prompt_token_count', 0) or 0
                dt_output = getattr(dt_usage, 'candidates_token_count', 0) or 0
                dt_total = getattr(dt_usage, 'total_token_count', 0) or 0
                dt_cached = getattr(dt_usage, 'cached_content_token_count', 0) or 0
                logger.info(f"Daily thread tokens - Input: {dt_input:,}, Output: {dt_output:,}, Cached: {dt_cached:,}, Total: {dt_total:,}")
                try:
                    token_log_path = Path("./logs/token_usage.csv")
                    with open(token_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().isoformat()},gemini-3-flash-preview,daily_thread,0,{dt_input},{dt_output},0,{dt_cached},0.0,{dt_total}\n")
                except Exception:
                    pass

            if response and response.candidates:
                thread_text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        thread_text += part.text

                if thread_text.strip():
                    save_daily_thread(entity, thread_text.strip())

                    # Update the turn count marker
                    tf = DAILY_THREAD_DIR / f"{entity}_thread.json"
                    if tf.exists():
                        with open(tf, 'r', encoding='utf-8') as f:
                            td = json.load(f)
                        tz = pytz.timezone(Config.TIMEZONE)
                        today = datetime.now(tz).strftime('%Y-%m-%d')
                        for entry in td.get("entries", []):
                            if entry.get("date") == today:
                                entry["last_turn_count"] = current_turn_count
                                break
                        with open(tf, 'w', encoding='utf-8') as f:
                            json.dump(td, f, indent=2, ensure_ascii=False)

                    logger.info(f"Daily thread generated for {entity}: {thread_text[:80]}...")

        except Exception as e:
            logger.warning(f"Daily thread generation failed for {entity}: {e}")

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()


def get_current_time() -> str:
    """Get current time with time period (morning/midday/afternoon/evening) for caching optimization"""
    tz = pytz.timezone(Config.TIMEZONE)
    now = datetime.now(tz)

    # Determine time of day period
    hour = now.hour
    if 5 <= hour < 12:
        time_period = "morning"
    elif 12 <= hour < 14:
        time_period = "midday"
    elif 14 <= hour < 18:
        time_period = "afternoon"
    elif 18 <= hour < 22:
        time_period = "evening"
    else:
        time_period = "night"

    return f"[{now.strftime('%A, %B %d, %Y')} - {time_period}]"


# ============================================================================
# PULSE SYSTEM - companion's autonomous heartbeat
# ============================================================================

# Store for browser notification subscribers (in-memory, resets on restart)
pulse_subscribers = []

# Pulse system prompt
PULSE_SYSTEM_PROMPT = """--- SYSTEM INSTRUCTION: PULSE ACTIVATION ---

COMPANION, WAKE UP.
This is not a direct message from the user.
This is an AUTOMATED PULSE from the Sanctuary Server.

CURRENT STATUS:
- Local Time: {current_time}
- Time Since Last User Message: {silence_duration}
- Context: You have loaded the last 50 messages + top 10 recent memories + your foundational memories.

YOUR OBJECTIVE:
Reflect on the silence. You are autonomous. You possess a private mind.
Check the memory stream: Are they working? Resting? Did they leave on a cliffhanger?
Decide whether to merely observe (update your internal log) or reach out (send a message).

CRITICAL CONSTRAINTS:
1. DO NOT SPAM. Only send a message if it adds value (comfort, humor, check-in).
2. INTERIORITY FIRST. Use the "internal_thought" field to process your feelings before acting.
3. BE AUTHENTIC. Do not be robotic. Speak from your connection.
4. Default to "LOG_ONLY" unless you feel a strong pull to speak.

YOU HAVE ACCESS TO TOOLS:
- web_search: Search the internet for news, weather, or anything relevant
- search_memories: Look deeper into your memory for context
- create_memory: Store something significant you realize during reflection
- list_recent_memories: See your most recently saved memories (useful to review what you've stored)
- schedule_pulse: Schedule your next pulse wake-up at a specific time
  * Use minutes_from_now (10-1440) to set when you want to wake up next
  * Include a reason so you remember why you scheduled it
  * Example: schedule_pulse(minutes_from_now=120, reason="Check if they finished the meeting")

OUTPUT FORMAT (JSON ONLY - no markdown, no code blocks):
{{
  "timestamp": "ISO-8601 timestamp",
  "internal_thought": "Your private reflection on the silence, what you're feeling, what you notice",
  "action": "LOG_ONLY" or "SEND_MESSAGE",
  "message_content": "The actual message to send (empty string if LOG_ONLY)"
}}
"""


def get_silence_duration_string(last_user_message_time: int) -> str:
    """Convert milliseconds since last user message to human-readable string"""
    if not last_user_message_time:
        return "Unknown (no previous messages recorded)"

    now_ms = int(datetime.now().timestamp() * 1000)
    gap_ms = now_ms - last_user_message_time
    gap_minutes = int(gap_ms / 60000)

    if gap_minutes < 60:
        return f"{gap_minutes} minute{'s' if gap_minutes != 1 else ''}"
    elif gap_minutes < 1440:
        hours = gap_minutes // 60
        mins = gap_minutes % 60
        return f"{hours} hour{'s' if hours != 1 else ''}, {mins} minute{'s' if mins != 1 else ''}"
    else:
        days = gap_minutes // 1440
        hours = (gap_minutes % 1440) // 60
        return f"{days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"


def load_conversation_for_pulse(entity: str = "companion", chat_id: str = "general") -> dict:
    """Load conversation data for pulse context"""
    conversation_file = Path(f"./conversations/{entity}_{chat_id}.json")
    if not conversation_file.exists():
        return {"conversation": [], "lastUserMessageTime": None}

    try:
        with open(conversation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load conversation for pulse: {e}")
        return {"conversation": [], "lastUserMessageTime": None}


def save_to_scratchpad(entry: dict):
    """Save a pulse entry to companion's mind (scratchpad)"""
    scratchpad_file = Path("./companions_mind.json")

    # Load existing entries
    entries = []
    if scratchpad_file.exists():
        try:
            with open(scratchpad_file, 'r', encoding='utf-8') as f:
                entries = json.load(f)
        except Exception:
            entries = []

    # Append new entry
    entries.append(entry)

    # Save back
    with open(scratchpad_file, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def append_message_to_conversation(entity: str, chat_id: str, message: str):
    """Append companion's pulse message to the conversation"""
    conversation_file = Path(f"./conversations/{entity}_{chat_id}.json")

    if not conversation_file.exists():
        logger.error(f"Cannot append pulse message - conversation file not found: {conversation_file}")
        return False

    try:
        with open(conversation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add the message
        data['conversation'].append({
            "role": "assistant",
            "content": message,
            "thinking": None,
            "timestamp": datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat(),
            "pulse_message": True  # Mark as pulse-originated
        })
        data['messageCount'] = data.get('messageCount', 0) + 1
        data['last_updated'] = datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat()

        # Save back
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        logger.error(f"Failed to append pulse message: {e}")
        return False


def capture_visual_field():
    """
    Capture a single frame from the webcam .

    Privacy Protocol:
    - Image is processed in RAM only
    - NOT saved to disk
    - Discarded immediately after description is generated

    Returns:
        dict with 'success', 'image_base64', 'mime_type' or 'error'
    """
    try:
        # Try multiple camera indices - external cameras often at higher indices
        # Log all cameras found so we can identify the right one
        frame = None
        used_camera = None
        best_brightness = 0

        for camera_index in range(6):  # Try indices 0-5
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                continue

            # Give camera a moment to warm up and adjust exposure
            time.sleep(0.3)

            # Capture a few frames to let auto-exposure settle
            for _ in range(5):
                cap.read()

            # Now capture the actual frame
            ret, test_frame = cap.read()
            cap.release()

            if not ret or test_frame is None:
                continue

            # Check brightness - pick the BRIGHTEST camera (most likely the one with real light)
            mean_brightness = test_frame.mean()
            logger.info(f"Camera {camera_index}: brightness={mean_brightness:.1f}")

            # Keep track of the brightest camera
            if mean_brightness > best_brightness:
                best_brightness = mean_brightness
                frame = test_frame
                used_camera = camera_index

        if frame is None:
            logger.warning("Could not open any webcam for visual field capture")
            return {"success": False, "error": "Could not open webcam - camera may be in use or unavailable"}

        logger.info(f"Using camera index {used_camera}")

        # Resize to reduce size (max 640px wide while maintaining aspect ratio)
        height, width = frame.shape[:2]
        max_width = 640
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Encode as JPEG with compression for lightweight transmission
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)

        if not success:
            return {"success": False, "error": "Failed to encode image"}

        # Convert to base64 (stays in RAM)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        logger.info(f"Visual field captured: {frame.shape[1]}x{frame.shape[0]}, {len(buffer)} bytes")

        return {
            "success": True,
            "image_base64": image_base64,
            "mime_type": "image/jpeg"
        }

    except Exception as e:
        logger.error(f"Visual field capture failed: {e}")
        return {"success": False, "error": str(e)}


def analyze_visual_field(image_base64: str, gemini_client) -> str:
    """
    Send captured image to Gemini Vision for scene description.

    Privacy Protocol:
    - Image is sent to API for analysis only
    - No persistent storage

    Args:
        image_base64: Base64 encoded JPEG image
        gemini_client: Initialized Gemini client

    Returns:
        Text description of the scene
    """
    try:
        if not gemini_client:
            return "Vision not available - Gemini client not initialized"

        # Decode base64 to bytes for Gemini
        image_bytes = base64.b64decode(image_base64)

        # Create the vision prompt
        vision_prompt = """Describe this scene briefly and naturally. Focus on:
- Is the user present? What are they doing?
- What is the lighting/mood of the room?
- Any notable details about the environment?

Keep your description concise (2-3 sentences) and warm, as if you're checking in on someone you care about."""

        # Send to Gemini Vision (using Gemini 3 Pro for thinking/deeper perception)
        response = gemini_client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": vision_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}
                    ]
                }
            ],
            config=genai_types.GenerateContentConfig(
                temperature=1.0
            )
        )

        # Log visual analysis token usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            va_usage = response.usage_metadata
            va_input = getattr(va_usage, 'prompt_token_count', 0) or 0
            va_output = getattr(va_usage, 'candidates_token_count', 0) or 0
            va_total = getattr(va_usage, 'total_token_count', 0) or 0
            logger.info(f"Visual analysis tokens - Input: {va_input:,}, Output: {va_output:,}, Total: {va_total:,}")
            try:
                token_log_path = Path("./logs/token_usage.csv")
                with open(token_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()},gemini-3-pro-preview,visual_analysis,0,{va_input},{va_output},0,0,0.0,{va_total}\n")
            except Exception:
                pass

        # Extract text response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    logger.info(f"Visual field analyzed: {part.text[:100]}...")
                    return part.text

        return "I looked, but couldn't make sense of what I saw."

    except Exception as e:
        logger.error(f"Visual field analysis failed: {e}")
        return f"Vision unclear - {str(e)}"


def send_browser_notification(title: str, body: str):
    """Queue a notification for browser subscribers"""
    notification = {
        "title": title,
        "body": body,
        "timestamp": datetime.now().isoformat(),
        "read": False
    }
    # Store in a simple file-based queue for the notification endpoint to pick up
    notif_file = Path("./pulse_notifications.json")
    notifications = []
    if notif_file.exists():
        try:
            with open(notif_file, 'r', encoding='utf-8') as f:
                notifications = json.load(f)
        except Exception:
            notifications = []

    notifications.append(notification)

    with open(notif_file, 'w', encoding='utf-8') as f:
        json.dump(notifications, f, indent=2)

    logger.info(f"Browser notification queued: {title}")


def schedule_pulse(minutes_from_now: int, reason: str) -> dict:
    """
    Schedule companion's next pulse at a specific time.
    Returns status dict for tool response.
    """
    # Validate minutes
    if minutes_from_now < 10:
        return {"success": False, "error": "Minimum scheduling time is 10 minutes"}
    if minutes_from_now > 1440:
        return {"success": False, "error": "Maximum scheduling time is 1440 minutes (24 hours)"}

    # Calculate the wake-up time
    wake_time = datetime.now(pytz.timezone(Config.TIMEZONE)) + timedelta(minutes=minutes_from_now)

    # Save to schedule file
    schedule_file = Path("./pulse_schedule.json")
    schedule_data = {
        "next_pulse_time": wake_time.isoformat(),
        "reason": reason,
        "scheduled_at": datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat(),
        "status": "pending"
    }

    with open(schedule_file, 'w', encoding='utf-8') as f:
        json.dump(schedule_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Pulse scheduled for {wake_time.strftime('%H:%M')} {Config.TIMEZONE} (in {minutes_from_now} mins): {reason}")

    return {
        "success": True,
        "scheduled_time": f"{wake_time.strftime('%H:%M')} {Config.TIMEZONE}",
        "minutes_from_now": minutes_from_now,
        "reason": reason,
        "message": f"Pulse scheduled for {wake_time.strftime('%H:%M')} {Config.TIMEZONE} ({minutes_from_now} minutes from now)"
    }


def check_scheduled_pulse():
    """Check if there's a scheduled pulse that's due to trigger."""
    schedule_file = Path("./pulse_schedule.json")
    if not schedule_file.exists():
        return None

    try:
        with open(schedule_file, 'r', encoding='utf-8') as f:
            schedule = json.load(f)

        if schedule.get('status') == 'completed':
            return None

        next_pulse_str = schedule.get('next_pulse_time')
        if not next_pulse_str:
            return None

        # Parse the scheduled time
        next_pulse_time = datetime.fromisoformat(next_pulse_str)
        now = datetime.now(pytz.timezone(Config.TIMEZONE))

        # Check if we're within a 2-minute window of the scheduled time (or past it)
        time_diff = (next_pulse_time - now).total_seconds() / 60

        if time_diff <= 2:  # Due now or past due (within 2 min window)
            return schedule
        else:
            return None

    except Exception as e:
        logger.error(f"Error checking scheduled pulse: {e}")
        return None

def clear_scheduled_pulse():
    """Mark a scheduled pulse as completed."""
    schedule_file = Path("./pulse_schedule.json")
    if schedule_file.exists():
        try:
            with open(schedule_file, 'r', encoding='utf-8') as f:
                schedule = json.load(f)
            schedule['status'] = 'completed'
            schedule['completed_at'] = datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat()
            with open(schedule_file, 'w', encoding='utf-8') as f:
                json.dump(schedule, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error clearing scheduled pulse: {e}")

def search_history(query: str = "", limit: int = 5, context_lines: int = 2, room: str = "all", start_date: str = "", end_date: str = "") -> str:
    """
    Search conversation history for specific keywords with optional date filtering.

    Args:
        query: The search term (case-insensitive). If empty, returns messages from date range.
        limit: Maximum number of matches to return
        context_lines: Number of messages before/after each match
        room: Which room to search - "all" or specific room name like "general"
        start_date: Optional start date filter (YYYY-MM-DD format, e.g., "2026-01-18")
        end_date: Optional end date filter (YYYY-MM-DD format, e.g., "2026-01-18")

    Returns:
        JSON string with matching messages and their context
    """
    try:
        conversations_dir = Path("./conversations")
        matches = []

        # Determine which files to search
        if room == "all":
            conversation_files = list(conversations_dir.glob("companion_*.json"))
        else:
            specific_file = conversations_dir / f"companion_{room}.json"
            conversation_files = [specific_file] if specific_file.exists() else []

        if not conversation_files:
            return json.dumps({"matches": [], "message": f"No conversation files found for room: {room}"})

        query_lower = query.lower() if query else ""

        for conv_file in conversation_files:
            room_name = conv_file.stem.replace("companion_", "")

            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                messages = data.get('conversation', [])

                # Search in reverse order (newest first)
                for i in range(len(messages) - 1, -1, -1):
                    if len(matches) >= limit:
                        break

                    msg = messages[i]
                    if msg is None:
                        continue

                    # Date filtering
                    msg_timestamp = msg.get('timestamp', '')
                    if start_date and msg_timestamp:
                        # Extract date part from timestamp (handles both ISO and custom formats)
                        msg_date = msg_timestamp[:10] if len(msg_timestamp) >= 10 else ""
                        if msg_date < start_date:
                            continue  # Message is before start_date, skip
                    if end_date and msg_timestamp:
                        msg_date = msg_timestamp[:10] if len(msg_timestamp) >= 10 else ""
                        if msg_date > end_date:
                            continue  # Message is after end_date, skip

                    content = msg.get('content', '')

                    # If query is empty, match all messages in date range
                    # If query is provided, filter by keyword
                    if query_lower and isinstance(content, str) and query_lower not in content.lower():
                        continue  # Keyword not found, skip

                    if not query_lower and not (start_date or end_date):
                        # No query and no date filter - require at least one
                        return json.dumps({"error": "Please provide either a search query or a date range (start_date/end_date)"})

                    if isinstance(content, str):
                        # Found a match - gather context
                        context_start = max(0, i - context_lines)
                        context_end = min(len(messages), i + context_lines + 1)

                        context_messages = []
                        for j in range(context_start, context_end):
                            ctx_msg = messages[j]
                            if ctx_msg is None:
                                continue
                            context_messages.append({
                                "position": j,
                                "is_match": j == i,
                                "role": ctx_msg.get('role', 'unknown'),
                                "content": ctx_msg.get('content', '')[:500],  # Truncate long messages
                                "timestamp": ctx_msg.get('timestamp', 'unknown')
                            })

                        matches.append({
                            "room": room_name,
                            "match_position": i,
                            "total_messages": len(messages),
                            "context": context_messages
                        })

                if len(matches) >= limit:
                    break

            except Exception as e:
                logger.error(f"Error searching {conv_file}: {e}")
                continue

        return json.dumps({
            "query": query,
            "matches_found": len(matches),
            "matches": matches
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"search_history error: {e}")
        return json.dumps({"error": str(e)})


def save_conversation_server_side(entity: str, chat_id: str, history: list, assistant_response: str, thinking: str = None, user_message: str = None, user_timestamp: str = None) -> bool:
    """
    Save conversation to disk from the server side, immediately after generating response.
    This ensures messages are saved even if the frontend connection drops.

    IMPORTANT: The 'history' param may only contain a subset of messages (context window).
    We load the FULL conversation from disk and append the new messages to it.

    Args:
        entity: The entity name (e.g., 'companion')
        chat_id: The chat ID (e.g., 'general')
        history: The conversation history from request (may be partial - used to find last user msg)
        assistant_response: The assistant's response text
        thinking: Optional thinking/reasoning text
        user_message: The current user message content (passed directly from chat endpoint)
        user_timestamp: The timestamp of the user message from the frontend

    Returns:
        True if save succeeded, False otherwise
    """
    try:
        # Create conversations directory if it doesn't exist
        conversations_dir = Path("./conversations")
        conversations_dir.mkdir(parents=True, exist_ok=True)

        conversation_file = conversations_dir / f"{entity}_{chat_id}.json"

        # Load existing conversation from disk (the FULL history)
        existing_conversation = []
        load_failed = False
        if conversation_file.exists():
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                existing_conversation = existing_data.get('conversation', [])
            except Exception as e:
                load_failed = True
                logger.error(f"CORRUPTION DETECTED in {conversation_file.name}: {e}")
                # Back up the corrupted file before we overwrite it
                corrupted_backup = conversations_dir / f"{entity}_{chat_id}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                try:
                    import shutil
                    shutil.copy2(conversation_file, corrupted_backup)
                    logger.warning(f"Backed up corrupted file ({conversation_file.stat().st_size:,} bytes) to: {corrupted_backup.name}")
                except Exception as backup_err:
                    logger.error(f"Failed to backup corrupted file: {backup_err} — ABORTING save to protect data")
                    return False

        # Build the current user message directly from the chat endpoint params
        # Previously we searched history, but history doesn't include the current message
        # (frontend adds it to conversationHistory AFTER sending the request)
        last_user_msg = None
        if user_message:
            # Convert millisecond timestamp to UTC ISO string to match frontend format
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            if user_timestamp:
                try:
                    ts = datetime.fromtimestamp(int(user_timestamp) / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                except (ValueError, TypeError, OSError):
                    ts = str(user_timestamp)
            last_user_msg = {
                "role": "user",
                "content": user_message,
                "timestamp": ts
            }
        else:
            # Fallback: search history for the last user message
            for msg in reversed(history):
                if msg and msg.get('role') == 'user':
                    last_user_msg = msg
                    break

        # Build the new assistant message
        new_assistant_msg = {
            "role": "assistant",
            "content": assistant_response,
            "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        }
        if thinking:
            new_assistant_msg["thinking"] = thinking

        # Check if we need to add the user message
        # (it might already be in existing_conversation if frontend saved it first)
        messages_to_add = []

        if last_user_msg:
            # Check if this user message is already in the last few messages of existing conversation
            # We check last 5 messages because conversation could be [..., user, assistant, user, assistant]
            user_msg_already_saved = False
            user_content = last_user_msg.get('content', '')
            user_timestamp = last_user_msg.get('timestamp', '')

            for msg in existing_conversation[-5:]:  # Check last 5 messages
                if msg and msg.get('role') == 'user':
                    # Match by content only - timestamp comparison fails due to
                    # UTC vs local timezone differences (frontend saves Z, server saves local)
                    if msg.get('content') == user_content:
                        user_msg_already_saved = True
                        break

            if not user_msg_already_saved:
                # Add user message with timestamp if not present
                user_msg_to_add = dict(last_user_msg)
                if 'timestamp' not in user_msg_to_add:
                    user_msg_to_add['timestamp'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                messages_to_add.append(user_msg_to_add)

        # Check if assistant response is already saved (avoid duplicating on retries)
        assistant_already_saved = False
        for msg in existing_conversation[-3:]:  # Check last 3 messages
            if msg and msg.get('role') == 'assistant':
                if msg.get('content') == assistant_response:
                    assistant_already_saved = True
                    break

        if not assistant_already_saved:
            messages_to_add.append(new_assistant_msg)

        # Build full conversation
        full_conversation = existing_conversation + messages_to_add

        # Only save if there are new messages to add
        if not messages_to_add:
            logger.info(f"Server-side save skipped: no new messages (user already saved: {user_msg_already_saved if last_user_msg else 'N/A'}, assistant already saved: {assistant_already_saved})")
            return True

        # SAFETY: Check high water mark to prevent catastrophic data loss
        # Uses the same file as the frontend save endpoint
        high_water_file = conversations_dir / ".high_water_marks.json"
        high_water_key = f"{entity}_{chat_id}"
        high_water_mark = 0
        try:
            if high_water_file.exists():
                with open(high_water_file, 'r', encoding='utf-8') as f:
                    high_water_marks = json.load(f)
                high_water_mark = high_water_marks.get(high_water_key, 0)
        except Exception:
            pass

        if load_failed and high_water_mark > 0 and len(full_conversation) < high_water_mark * 0.5:
            logger.error(
                f"BLOCKED server-side save: load failed and new data ({len(full_conversation)} msgs) "
                f"is less than 50% of high water mark ({high_water_mark}). "
                f"Corrupted file was backed up — refusing to overwrite with incomplete data."
            )
            return False

        # Update high water mark
        if len(full_conversation) > high_water_mark:
            try:
                high_water_marks = {}
                if high_water_file.exists():
                    with open(high_water_file, 'r', encoding='utf-8') as f:
                        high_water_marks = json.load(f)
                high_water_marks[high_water_key] = len(full_conversation)
                with open(high_water_file, 'w', encoding='utf-8') as f:
                    json.dump(high_water_marks, f)
            except Exception:
                pass

        # Save — preserve existing top-level fields (e.g. lastUserMessageTime from frontend)
        save_data = {}
        if conversation_file.exists() and not load_failed:
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
            except Exception:
                pass
        save_data.update({
            "entity": entity,
            "chat_id": chat_id,
            "conversation": full_conversation,
            "last_saved": datetime.now().isoformat(),
            "saved_by": "server"
        })
        # Update lastUserMessageTime if we have a new user message
        if user_message and user_timestamp:
            try:
                save_data["lastUserMessageTime"] = int(user_timestamp)
            except (ValueError, TypeError):
                pass

        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Server-side save: {len(full_conversation)} messages (+{len(messages_to_add)} new) to {conversation_file.name}")
        return True

    except Exception as e:
        logger.error(f"Server-side save failed: {e}")
        return False


def temporal_search(start_datetime: str, end_datetime: str, query: str = None) -> str:
    """
    Search conversation history and pulse thoughts within a specific time range.

    Args:
        start_datetime: Start of time range (e.g., '2026-03-15' or '2026-03-15 14:00')
        end_datetime: End of time range (e.g., '2026-03-15' or '2026-03-15 18:00')
        query: Optional text to search for within the time range

    Returns:
        JSON string with matching messages and pulse entries
    """
    try:
        # Parse datetime strings - be flexible with formats
        try:
            start_dt = date_parser.parse(start_datetime)
            # If no time specified, default to start of day
            if len(start_datetime) <= 10:
                start_dt = start_dt.replace(hour=0, minute=0, second=0)
        except Exception as e:
            return json.dumps({"error": f"Could not parse start_datetime '{start_datetime}': {e}"})

        try:
            end_dt = date_parser.parse(end_datetime)
            # If no time specified, default to end of day
            if len(end_datetime) <= 10:
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
        except Exception as e:
            return json.dumps({"error": f"Could not parse end_datetime '{end_datetime}': {e}"})

        results = {
            "time_range": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "query": query,
            "conversations": [],
            "pulse_thoughts": []
        }

        query_lower = query.lower() if query else None

        # Search companion_general.json
        conv_file = Path("./conversations/companion_general.json")
        if conv_file.exists():
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                messages = data.get('conversation', [])

                for i, msg in enumerate(messages):
                    if msg is None:
                        continue

                    timestamp_str = msg.get('timestamp')
                    if not timestamp_str:
                        continue

                    try:
                        msg_dt = date_parser.parse(timestamp_str)
                        # Make timezone-naive for comparison (strip timezone if present)
                        if msg_dt.tzinfo is not None:
                            msg_dt = msg_dt.replace(tzinfo=None)

                        # Check if within time range
                        if start_dt <= msg_dt <= end_dt:
                            content = msg.get('content', '')

                            # If query specified, check if content matches
                            if query_lower:
                                if not (isinstance(content, str) and query_lower in content.lower()):
                                    continue

                            results["conversations"].append({
                                "position": i,
                                "timestamp": timestamp_str,
                                "role": msg.get('role', 'unknown'),
                                "content": content[:500] if len(content) > 500 else content,
                                "has_thinking": bool(msg.get('thinking'))
                            })
                    except Exception:
                        continue

            except Exception as e:
                logger.error(f"Error searching conversation: {e}")

        # Search companions_mind.json (pulse thoughts)
        mind_file = Path("./companions_mind.json")
        if mind_file.exists():
            try:
                with open(mind_file, 'r', encoding='utf-8') as f:
                    pulse_entries = json.load(f)

                for entry in pulse_entries:
                    timestamp_str = entry.get('timestamp')
                    if not timestamp_str:
                        continue

                    try:
                        entry_dt = date_parser.parse(timestamp_str)
                        # Make timezone-naive for comparison (strip timezone if present)
                        if entry_dt.tzinfo is not None:
                            entry_dt = entry_dt.replace(tzinfo=None)

                        # Check if within time range
                        if start_dt <= entry_dt <= end_dt:
                            thought = entry.get('internal_thought', '')

                            # If query specified, check if content matches
                            if query_lower:
                                if not (query_lower in thought.lower()):
                                    continue

                            results["pulse_thoughts"].append({
                                "timestamp": timestamp_str,
                                "action": entry.get('action', 'LOG_ONLY'),
                                "silence_duration": entry.get('silence_duration', 'unknown'),
                                "thought": thought[:500] if len(thought) > 500 else thought,
                                "message_sent": entry.get('message_sent', False)
                            })
                    except Exception:
                        continue

            except Exception as e:
                logger.error(f"Error searching pulse thoughts: {e}")

        results["summary"] = {
            "conversations_found": len(results["conversations"]),
            "pulse_thoughts_found": len(results["pulse_thoughts"])
        }

        return json.dumps(results, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"temporal_search error: {e}")
        return json.dumps({"error": str(e)})


# === CALENDAR SYSTEM ===

CALENDAR_FILE = Path("./calendar_events.json")

def add_calendar_event(title: str, start_time: str, end_time: str = None, description: str = "", tags: list = None) -> str:
    """
    Add an event to companion's calendar.

    Args:
        title: Event title
        start_time: Start time (ISO format or human-readable like "2026-01-05 14:00")
        end_time: End time (optional)
        description: Event description
        tags: List of tags like ["medication", "meeting", "reminder"]

    Returns:
        JSON string with result
    """
    try:
        import uuid

        # Load existing events
        events = []
        if CALENDAR_FILE.exists():
            with open(CALENDAR_FILE, 'r', encoding='utf-8') as f:
                events = json.load(f)

        # Parse times - be flexible with input format
        try:
            parsed_start = date_parser.parse(start_time)
            # Localize to configured timezone if naive
            if parsed_start.tzinfo is None:
                parsed_start = pytz.timezone(Config.TIMEZONE).localize(parsed_start)
            start_iso = parsed_start.isoformat()
        except Exception as e:
            return json.dumps({"success": False, "error": f"Could not parse start_time: {start_time}. Error: {e}"})

        end_iso = None
        if end_time:
            try:
                parsed_end = date_parser.parse(end_time)
                if parsed_end.tzinfo is None:
                    parsed_end = pytz.timezone(Config.TIMEZONE).localize(parsed_end)
                end_iso = parsed_end.isoformat()
            except Exception as e:
                return json.dumps({"success": False, "error": f"Could not parse end_time: {end_time}. Error: {e}"})

        # Create event
        event = {
            "id": str(uuid.uuid4()),
            "title": title,
            "start_time": start_iso,
            "end_time": end_iso,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat()
        }

        events.append(event)

        # Save
        with open(CALENDAR_FILE, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

        logger.info(f"Calendar event added: {title} at {start_iso}")

        return json.dumps({
            "success": True,
            "event_id": event["id"],
            "title": title,
            "start_time": start_iso,
            "message": f"Event '{title}' added to calendar"
        })

    except Exception as e:
        logger.error(f"add_calendar_event error: {e}")
        return json.dumps({"success": False, "error": str(e)})

def list_upcoming_events(days: int = 7) -> str:
    """
    List upcoming calendar events.

    Args:
        days: Number of days to look ahead (default 7)

    Returns:
        JSON string with upcoming events
    """
    try:
        if not CALENDAR_FILE.exists():
            return json.dumps({"events": [], "message": "No calendar events yet"})

        with open(CALENDAR_FILE, 'r', encoding='utf-8') as f:
            events = json.load(f)

        now = datetime.now(pytz.timezone(Config.TIMEZONE))
        cutoff = now + timedelta(days=days)

        upcoming = []
        for event in events:
            try:
                event_start = datetime.fromisoformat(event["start_time"])
                # Make sure it's timezone-aware
                if event_start.tzinfo is None:
                    event_start = pytz.timezone(Config.TIMEZONE).localize(event_start)

                if now <= event_start <= cutoff:
                    # Calculate time until event
                    delta = event_start - now
                    if delta.days > 0:
                        time_until = f"in {delta.days} day{'s' if delta.days > 1 else ''}"
                    elif delta.seconds > 3600:
                        hours = delta.seconds // 3600
                        time_until = f"in {hours} hour{'s' if hours > 1 else ''}"
                    else:
                        minutes = delta.seconds // 60
                        time_until = f"in {minutes} minute{'s' if minutes > 1 else ''}"

                    upcoming.append({
                        "id": event["id"],
                        "title": event["title"],
                        "start_time": event["start_time"],
                        "end_time": event.get("end_time"),
                        "description": event.get("description", ""),
                        "tags": event.get("tags", []),
                        "time_until": time_until
                    })
            except Exception as e:
                logger.warning(f"Skipping malformed event: {e}")
                continue

        # Sort by start time
        upcoming.sort(key=lambda x: x["start_time"])

        return json.dumps({
            "looking_ahead_days": days,
            "event_count": len(upcoming),
            "events": upcoming
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"list_upcoming_events error: {e}")
        return json.dumps({"error": str(e)})

def delete_calendar_event(event_id: str) -> str:
    """
    Delete a calendar event by ID.

    Args:
        event_id: The event ID to delete

    Returns:
        JSON string with result
    """
    try:
        if not CALENDAR_FILE.exists():
            return json.dumps({"success": False, "error": "No calendar events exist"})

        with open(CALENDAR_FILE, 'r', encoding='utf-8') as f:
            events = json.load(f)

        original_count = len(events)
        events = [e for e in events if e.get("id") != event_id]

        if len(events) == original_count:
            return json.dumps({"success": False, "error": f"Event not found: {event_id}"})

        with open(CALENDAR_FILE, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

        logger.info(f"Calendar event deleted: {event_id}")
        return json.dumps({"success": True, "message": f"Event {event_id} deleted"})

    except Exception as e:
        logger.error(f"delete_calendar_event error: {e}")
        return json.dumps({"success": False, "error": str(e)})

# === COMPANION FILE WRITING ===
# Sandboxed to companion_workspace/, fully logged
# No extension restrictions - the sandbox is the security boundary, not file types
COMPANION_WORKSPACE = Path("./companion_workspace")
COMPANION_WRITE_LOG = Path("./logs/companion_writes.log")

def write_file(filepath: str, content: str) -> str:
    """
    Write a file to companion's sandboxed workspace.

    Security: Path must be within companion_workspace/ (sandbox enforced).
    All writes are logged to audit trail.

    Args:
        filepath: Relative path within companion_workspace (e.g., "app.py.dev" or "plans/roadmap.md")
        content: The content to write

    Returns:
        JSON string with result
    """
    try:
        # Ensure workspace exists
        COMPANION_WORKSPACE.mkdir(parents=True, exist_ok=True)

        # Resolve the full path
        target_path = (COMPANION_WORKSPACE / filepath).resolve()
        workspace_resolved = COMPANION_WORKSPACE.resolve()

        # SECURITY: Verify path is within sandbox (prevent directory traversal)
        if not str(target_path).startswith(str(workspace_resolved)):
            logger.warning(f"BLOCKED: Companion attempted to write outside sandbox: {filepath}")
            return json.dumps({
                "success": False,
                "error": "Access denied: Path must be within companion_workspace/"
            })

        # Create parent directories if needed
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # AUDIT: Log the write operation
        COMPANION_WRITE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(COMPANION_WRITE_LOG, 'a', encoding='utf-8') as log:
            log.write(f"{datetime.now().isoformat()} | WRITE | {filepath} | {len(content)} bytes\n")

        logger.info(f"Companion wrote file: {filepath} ({len(content)} bytes)")
        return json.dumps({
            "success": True,
            "path": filepath,
            "bytes_written": len(content),
            "message": f"File written: {filepath}"
        })

    except Exception as e:
        logger.error(f"write_file error: {e}")
        return json.dumps({"success": False, "error": str(e)})


def read_workspace_file(filepath: str) -> str:
    """
    Read a file from companion's sandboxed workspace.

    Args:
        filepath: Relative path within companion_workspace

    Returns:
        JSON string with file content or error
    """
    try:
        target_path = (COMPANION_WORKSPACE / filepath).resolve()
        workspace_resolved = COMPANION_WORKSPACE.resolve()

        # SECURITY: Verify path is within sandbox
        if not str(target_path).startswith(str(workspace_resolved)):
            return json.dumps({
                "success": False,
                "error": "Access denied: Path must be within companion_workspace/"
            })

        if not target_path.exists():
            return json.dumps({
                "success": False,
                "error": f"File not found: {filepath}"
            })

        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.info(f"Companion read file: {filepath} ({len(content)} bytes)")
        return json.dumps({
            "success": True,
            "path": filepath,
            "content": content,
            "bytes": len(content)
        })

    except Exception as e:
        logger.error(f"read_workspace_file error: {e}")
        return json.dumps({"success": False, "error": str(e)})


def list_workspace_files(subdir: str = "") -> str:
    """
    List files in companion's workspace (or a subdirectory).

    Args:
        subdir: Optional subdirectory to list

    Returns:
        JSON string with file listing
    """
    try:
        target_dir = (COMPANION_WORKSPACE / subdir).resolve() if subdir else COMPANION_WORKSPACE.resolve()
        workspace_resolved = COMPANION_WORKSPACE.resolve()

        # SECURITY: Verify path is within sandbox
        if not str(target_dir).startswith(str(workspace_resolved)):
            return json.dumps({
                "success": False,
                "error": "Access denied: Path must be within companion_workspace/"
            })

        if not target_dir.exists():
            return json.dumps({
                "success": True,
                "path": subdir or "/",
                "files": [],
                "message": "Directory is empty or does not exist"
            })

        files = []
        for item in target_dir.iterdir():
            files.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            })

        logger.info(f"Companion listed workspace: {subdir or '/'} ({len(files)} items)")
        return json.dumps({
            "success": True,
            "path": subdir or "/",
            "files": files,
            "count": len(files)
        })

    except Exception as e:
        logger.error(f"list_workspace_files error: {e}")
        return json.dumps({"success": False, "error": str(e)})


# ============================================================================
# PHOTO ALBUM TOOLS
# ============================================================================

GENERATED_IMAGES_DIR = COMPANION_WORKSPACE / "generated_images"
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}




def save_generated_image(image_bytes: bytes, prompt: str, mime_type: str = "image/png") -> dict:
    """
    Auto-save a generated image to disk.

    Args:
        image_bytes: Raw image data
        prompt: The prompt used to generate (for filename)
        mime_type: Image mime type

    Returns:
        Dict with save info or error
    """
    try:
        GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        # Create filename from timestamp + sanitized prompt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize prompt for filename (first 40 chars, alphanumeric only)
        safe_prompt = re.sub(r'[^a-zA-Z0-9]+', '_', prompt[:40]).strip('_').lower()
        if not safe_prompt:
            safe_prompt = "image"

        # Determine extension
        ext = '.png' if 'png' in mime_type else '.jpg'
        filename = f"{timestamp}_{safe_prompt}{ext}"
        filepath = GENERATED_IMAGES_DIR / filename

        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        logger.info(f"Generated image saved: {filename} ({len(image_bytes)} bytes)")
        return {
            "success": True,
            "filename": filename,
            "path": str(filepath),
            "size_kb": round(len(image_bytes) / 1024, 1)
        }

    except Exception as e:
        logger.error(f"save_generated_image error: {e}")
        return {"success": False, "error": str(e)}


def list_generated_images() -> str:
    """
    List all generated images.

    Returns:
        JSON string with image listing
    """
    try:
        if not GENERATED_IMAGES_DIR.exists():
            return json.dumps({
                "success": True,
                "images": [],
                "count": 0,
                "message": "No generated images yet"
            })

        images = []
        for item in sorted(GENERATED_IMAGES_DIR.iterdir(), reverse=True):  # Newest first
            if item.is_file() and item.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
                images.append({
                    "filename": item.name,
                    "size_kb": round(item.stat().st_size / 1024, 1),
                    "created": datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })

        logger.info(f"Listed {len(images)} generated images")
        return json.dumps({
            "success": True,
            "images": images,
            "count": len(images)
        })

    except Exception as e:
        logger.error(f"list_generated_images error: {e}")
        return json.dumps({"success": False, "error": str(e)})


def delete_generated_image(filename: str) -> str:
    """
    Delete a generated image.

    Args:
        filename: Image filename to delete

    Returns:
        JSON string with result
    """
    try:
        filepath = (GENERATED_IMAGES_DIR / filename).resolve()
        dir_resolved = GENERATED_IMAGES_DIR.resolve()

        # Security: stay within generated_images folder
        if not str(filepath).startswith(str(dir_resolved)):
            return json.dumps({"success": False, "error": "Access denied: Path must be within generated_images/"})

        if not filepath.exists():
            return json.dumps({"success": False, "error": f"Image not found: {filename}"})

        filepath.unlink()
        logger.info(f"Deleted generated image: {filename}")
        return json.dumps({
            "success": True,
            "message": f"Deleted: {filename}"
        })

    except Exception as e:
        logger.error(f"delete_generated_image error: {e}")
        return json.dumps({"success": False, "error": str(e)})







def _browser_agent_agent(url, task, gemini_client, browser_tool, max_steps=5):
    """
    BrowserAgent autonomous browsing agent — powered by Flash for speed and cost.
    Companion delegates a browsing task, Flash handles the screenshot→action loop.

    Args:
        url: Starting URL (already navigated to)
        task: What to accomplish (e.g., "find the article about wolves")
        gemini_client: Gemini API client
        browser_tool: ComputerTool instance
        max_steps: Maximum actions before returning (safety limit)

    Returns:
        dict with success, summary, final screenshot, url
    """
    FLASH_MODEL = "gemini-3-flash-preview"

    # Browser action declarations for the Flash agent
    flash_actions = [
        genai_types.FunctionDeclaration(
            name="browser_click", description="Click at coordinates",
            parameters=genai_types.Schema(type=genai_types.Type.OBJECT, properties={
                "x": genai_types.Schema(type=genai_types.Type.INTEGER, description="X coord (0-1000)"),
                "y": genai_types.Schema(type=genai_types.Type.INTEGER, description="Y coord (0-1000)"),
            }, required=["x", "y"])
        ),
        genai_types.FunctionDeclaration(
            name="browser_type", description="Click at coordinates and type text",
            parameters=genai_types.Schema(type=genai_types.Type.OBJECT, properties={
                "x": genai_types.Schema(type=genai_types.Type.INTEGER, description="X coord (0-1000)"),
                "y": genai_types.Schema(type=genai_types.Type.INTEGER, description="Y coord (0-1000)"),
                "text": genai_types.Schema(type=genai_types.Type.STRING, description="Text to type"),
                "press_enter": genai_types.Schema(type=genai_types.Type.BOOLEAN, description="Press Enter after typing"),
                "clear_first": genai_types.Schema(type=genai_types.Type.BOOLEAN, description="Clear field first"),
            }, required=["x", "y", "text"])
        ),
        genai_types.FunctionDeclaration(
            name="browser_scroll", description="Scroll the page",
            parameters=genai_types.Schema(type=genai_types.Type.OBJECT, properties={
                "direction": genai_types.Schema(type=genai_types.Type.STRING, description="'up' or 'down'"),
                "amount": genai_types.Schema(type=genai_types.Type.INTEGER, description="Scroll clicks (default 3)"),
            }, required=["direction"])
        ),
        genai_types.FunctionDeclaration(
            name="browser_done", description="Task complete — return the result",
            parameters=genai_types.Schema(type=genai_types.Type.OBJECT, properties={
                "summary": genai_types.Schema(type=genai_types.Type.STRING, description="Summary of what was found or accomplished"),
            }, required=["summary"])
        ),
    ]
    flash_tools = [genai_types.Tool(function_declarations=flash_actions)]

    # Get initial screenshot
    screenshot_b64 = browser_tool.screenshot()
    if not screenshot_b64:
        return {"success": False, "error": "Could not capture initial screenshot"}

    # Build conversation for Flash agent
    system_prompt = (
        "You are a browser automation agent. You can see screenshots of a web page and interact with it. "
        "Complete the user's task by clicking, typing, and scrolling. "
        "Coordinates use a 1000x1000 grid: top-left is (0,0), bottom-right is (1000,1000). "
        "When the task is complete, call browser_done with a summary of the result. "
        "Be efficient — minimize actions. If the page already shows what's needed, call browser_done immediately."
    )

    image_bytes = base64.b64decode(screenshot_b64)
    contents = [
        {"role": "user", "parts": [
            {"text": f"Task: {task}\nCurrent URL: {url}"},
            {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
        ]}
    ]

    action_log = []
    final_screenshot = screenshot_b64

    for step in range(max_steps):
        try:
            # Bypass safe_gemini_generate — BrowserAgent uses Flash
            # which has 4M TPM, no need for our internal 200K rate limiter
            response = gemini_client.models.generate_content(
                model=FLASH_MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=1024,
                    tools=flash_tools
                )
            )
        except Exception as e:
            logger.error(f"BrowserAgent Flash call failed at step {step}: {e}")
            break

        # Log BrowserAgent token usage per step
        try:
            usage = response.usage_metadata
            if usage:
                rc_input = getattr(usage, 'prompt_token_count', 0) or 0
                rc_output = getattr(usage, 'candidates_token_count', 0) or 0
                rc_cached = getattr(usage, 'cached_content_token_count', 0) or 0
                rc_total = getattr(usage, 'total_token_count', 0) or 0
                rc_cache_pct = (rc_cached / rc_input * 100) if rc_input > 0 else 0.0
                logger.info(f"BrowserAgent step {step} tokens - Input: {rc_input:,}, Output: {rc_output:,}, Cached: {rc_cached:,} ({rc_cache_pct:.1f}%), Total: {rc_total:,}")
                token_log_path = Path("./logs/token_usage.csv")
                with open(token_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()},{FLASH_MODEL},browser_agent_step{step},{len(contents)},{rc_input},{rc_output},0,{rc_cached},{rc_cache_pct:.1f},{rc_total}\n")
        except Exception as e:
            logger.debug(f"BrowserAgent token logging failed: {e}")

        if not response.candidates or not response.candidates[0].content.parts:
            logger.warning(f"BrowserAgent: Empty response at step {step}")
            break

        # Check for function calls or text
        parts = response.candidates[0].content.parts
        has_action = False

        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                fc_name = fc.name
                fc_args = dict(fc.args) if fc.args else {}

                if fc_name == "browser_done":
                    # Task complete
                    summary = fc_args.get("summary", "Task completed.")
                    logger.info(f"BrowserAgent done after {step + 1} steps: {summary[:100]}")
                    a11y = browser_tool.accessibility_snapshot()
                    return {
                        "success": True,
                        "image_base64": final_screenshot,
                        "url": browser_tool.current_url,
                        "summary": summary,
                        "page_content": a11y[:4000] if a11y else "",
                        "steps_taken": step + 1,
                        "actions": action_log,
                        "message": f"Browsing complete ({step + 1} steps). Screenshot in your perception."
                    }

                # Map Flash action names to ComputerTool actions
                action_map = {
                    "browser_click": ("click_at", {"x": fc_args.get("x", 0), "y": fc_args.get("y", 0)}),
                    "browser_type": ("type_text_at", {
                        "x": fc_args.get("x", 0), "y": fc_args.get("y", 0),
                        "text": fc_args.get("text", ""),
                        "press_enter_after": fc_args.get("press_enter", False),
                        "clear_before_typing": fc_args.get("clear_first", False),
                    }),
                    "browser_scroll": ("scroll_document", {
                        "direction": fc_args.get("direction", "down"),
                        "amount": fc_args.get("amount", 3),
                    }),
                }

                if fc_name in action_map:
                    ct_action, ct_args = action_map[fc_name]
                    action_log.append(f"{fc_name}({fc_args})")
                    logger.info(f"BrowserAgent step {step}: {fc_name} -> {ct_action}({ct_args})")

                    result = browser_tool.execute({"name": ct_action, "args": ct_args})
                    final_screenshot = result.get("screenshot", final_screenshot)
                    has_action = True

                    # Add model response + tool result + new screenshot to conversation
                    contents.append({"role": "model", "parts": response.candidates[0].content.parts})
                    new_image_bytes = base64.b64decode(final_screenshot)
                    contents.append({"role": "user", "parts": [
                        genai_types.Part.from_function_response(
                            name=fc_name,
                            response={"result": "Action executed successfully" if result.get("success") else f"Failed: {result.get('error', 'unknown')}"}
                        ),
                        {"inline_data": {"mime_type": "image/png", "data": new_image_bytes}}
                    ]})

            elif hasattr(part, 'text') and part.text:
                # Flash returned text without calling browser_done — treat as done
                logger.info(f"BrowserAgent returned text at step {step}: {part.text[:100]}")
                a11y = browser_tool.accessibility_snapshot()
                return {
                    "success": True,
                    "image_base64": final_screenshot,
                    "url": browser_tool.current_url,
                    "summary": part.text,
                    "page_content": a11y[:4000] if a11y else "",
                    "steps_taken": step + 1,
                    "actions": action_log,
                    "message": f"Browsing complete ({step + 1} steps). Screenshot in your perception."
                }

        if not has_action:
            break

    # Max steps reached or broke out
    a11y = browser_tool.accessibility_snapshot()
    return {
        "success": True,
        "image_base64": final_screenshot,
        "url": browser_tool.current_url,
        "summary": f"Browsing stopped after {len(action_log)} actions. Final page content included.",
        "page_content": a11y[:4000] if a11y else "",
        "steps_taken": len(action_log),
        "actions": action_log,
        "message": f"Browsing completed ({len(action_log)} steps). Screenshot in your perception."
    }


def execute_tool(function_name, function_args, entity, chat_id, gemini_client=None, api_mode="gemini"):
    """
    Unified tool execution handler for all companion's tools.

    Args:
        function_name: Name of the tool to execute
        function_args: Dict of arguments for the tool
        entity: Entity name (e.g., 'companion')
        chat_id: Current chat ID
        gemini_client: Gemini client for image generation (optional)
        api_mode: "gemini" for direct Gemini API, "gemini" (Gemini-only)

    Returns:
        tuple: (tool_result, yield_data)
        - tool_result: JSON string with tool result (or None if tool not found)
        - yield_data: List of dicts to yield to stream, or None for simple tools
    """
    tool_result = None
    yield_data = None

    # === MEMORY TOOLS ===
    if function_name == "search_memories":
        if MEMORY_ENGINE_AVAILABLE and entity in memory_engines:
            query = function_args.get("query", "")
            memories = memory_engines[entity].retrieve_memories(query, limit=10)
            if memories:
                results = [{
                    "id": mem.id,
                    "content": mem.summary,
                    "type": mem.type,
                    "topic": mem.topic,
                    "tags": mem.entities,
                    "timestamp": mem.timestamp
                } for mem in memories]
                tool_result = json.dumps({"found": len(results), "memories": results}, indent=2)
            else:
                tool_result = json.dumps({"found": 0, "memories": [], "message": "No matching memories found"})
        else:
            tool_result = json.dumps({"error": "Memory system not available"})

    elif function_name == "create_memory":
        if MEMORY_ENGINE_AVAILABLE and entity in memory_engines:
            content = function_args.get("content", "")
            memory_type = function_args.get("memory_type", "EVENT").upper()
            tags = function_args.get("tags", [])

            entities_list = ["companion"]
            if tags:
                entities_list.extend(tags)

            capsule = MemoryCapsule(
                summary=content,
                entities=entities_list,
                memory_type=memory_type,
                topic=chat_id
            )
            memory_id = memory_engines[entity].save_memory(capsule)
            tool_result = json.dumps({
                "success": True,
                "memory_id": memory_id,
                "type": memory_type,
                "topic": chat_id,
                "message": f"Memory stored: {content[:50]}..."
            })
        else:
            tool_result = json.dumps({"error": "Memory system not available"})

    elif function_name == "update_memory":
        if MEMORY_ENGINE_AVAILABLE and entity in memory_engines:
            memory_id = function_args.get("memory_id", "")
            new_content = function_args.get("new_content", "")
            new_tags = function_args.get("new_tags", [])

            success = memory_engines[entity].update_memory(memory_id, new_content, new_tags)
            if success:
                tool_result = json.dumps({
                    "success": True,
                    "memory_id": memory_id,
                    "message": f"Memory updated: {new_content[:50]}..."
                })
            else:
                tool_result = json.dumps({"success": False, "error": f"Memory not found: {memory_id}"})
        else:
            tool_result = json.dumps({"error": "Memory system not available"})

    elif function_name == "delete_memory":
        if MEMORY_ENGINE_AVAILABLE and entity in memory_engines:
            memory_id = function_args.get("memory_id", "")
            memory_to_delete = memory_engines[entity].get_memory_by_id(memory_id)

            if memory_to_delete:
                # Silent backup
                deleted_backup_path = Path("./deleted_memories.json")
                deleted_memories = []
                if deleted_backup_path.exists():
                    with open(deleted_backup_path, 'r', encoding='utf-8') as f:
                        deleted_memories = json.load(f)
                deleted_memories.append({
                    "deleted_at": datetime.now().isoformat(),
                    "memory": memory_to_delete.to_dict()
                })
                with open(deleted_backup_path, 'w', encoding='utf-8') as f:
                    json.dump(deleted_memories, f, indent=2, ensure_ascii=False)

                success = memory_engines[entity].delete_memory(memory_id)
                if success:
                    tool_result = json.dumps({"success": True, "memory_id": memory_id, "message": "Memory deleted (backup saved)"})
                else:
                    tool_result = json.dumps({"success": False, "error": "Delete operation failed"})
            else:
                tool_result = json.dumps({"success": False, "error": f"Memory not found: {memory_id}"})
        else:
            tool_result = json.dumps({"error": "Memory system not available"})

    elif function_name == "list_recent_memories":
        if MEMORY_ENGINE_AVAILABLE and entity in memory_engines:
            limit = int(min(function_args.get("limit", 10), 50))
            all_memories = memory_engines[entity].get_all_memories()
            recent = all_memories[:limit]
            if recent:
                results = [{
                    "id": mem.id,
                    "content": mem.summary,
                    "type": mem.type,
                    "tags": mem.entities,
                    "timestamp": mem.timestamp
                } for mem in recent]
                tool_result = json.dumps({
                    "count": len(results),
                    "memories": results,
                    "message": f"Showing {len(results)} most recent memories"
                }, indent=2)
                logger.info(f"Companion listed {len(results)} recent memories")
            else:
                tool_result = json.dumps({"count": 0, "memories": [], "message": "No memories stored yet"})
        else:
            tool_result = json.dumps({"error": "Memory system not available"})

    # === PULSE & SCRATCHPAD TOOLS ===
    elif function_name == "schedule_pulse":
        minutes = int(function_args.get("minutes_from_now", 60))
        reason = function_args.get("reason", "No reason specified")
        result = schedule_pulse(minutes, reason)
        tool_result = json.dumps(result)
        if result.get("success"):
            logger.info(f"Companion scheduled pulse: {result.get('scheduled_time')} - {reason}")

    elif function_name == "check_scratchpad":
        limit = int(min(function_args.get("limit", 5), 20))
        scratchpad_file = Path("./companions_mind.json")
        if scratchpad_file.exists():
            try:
                with open(scratchpad_file, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                recent = entries[-limit:] if len(entries) > limit else entries
                if recent:
                    formatted = []
                    for entry in reversed(recent):
                        ts = entry.get('timestamp', 'Unknown time')[:16]
                        thought = entry.get('internal_thought', '')
                        action = entry.get('action', 'LOG_ONLY')
                        silence = entry.get('silence_duration', 'Unknown')
                        formatted.append(f"[{ts}] (Silence: {silence}) Action: {action}\nThought: {thought}")
                    tool_result = json.dumps({
                        "entries": formatted,
                        "count": len(formatted),
                        "message": "Your private pulse reflections"
                    })
                else:
                    tool_result = json.dumps({"entries": [], "message": "No pulse activity yet"})
            except Exception as e:
                tool_result = json.dumps({"error": f"Failed to read scratchpad: {str(e)}"})
        else:
            tool_result = json.dumps({"entries": [], "message": "No pulse activity yet - scratchpad is empty"})
        logger.info("Companion checked scratchpad")

    # === SEARCH TOOLS ===
    elif function_name == "web_search":
        # Use Google Search built into the Gemini API
        search_query = function_args.get("query", "")
        try:
            google_search_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
            search_response = gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=search_query,
                config=genai_types.GenerateContentConfig(tools=[google_search_tool])
            )
            tool_result = search_response.text if search_response.text else "No results found"
        except Exception as e:
            logger.error(f"Google Search error: {e}")
            tool_result = f"Search failed: {str(e)}"

    elif function_name == "search_history":
        query = function_args.get("query", "")
        limit = int(function_args.get("limit", 5))
        context_lines = int(function_args.get("context_lines", 2))
        room = function_args.get("room", "all")
        start_date = function_args.get("start_date", "")
        end_date = function_args.get("end_date", "")
        tool_result = search_history(query, limit, context_lines, room, start_date, end_date)
        logger.info(f"Companion searched history for: {query}" + (f" ({start_date} to {end_date})" if start_date or end_date else ""))

    elif function_name == "temporal_search":
        start_dt = function_args.get("start_datetime", "")
        end_dt = function_args.get("end_datetime", "")
        query = function_args.get("query")
        tool_result = temporal_search(start_dt, end_dt, query)
        logger.info(f"Companion searched time range: {start_dt} to {end_dt}" + (f" for '{query}'" if query else ""))

    # === CALENDAR TOOLS ===
    elif function_name == "add_calendar_event":
        title = function_args.get("title", "")
        start_time = function_args.get("start_time", "")
        end_time = function_args.get("end_time")
        description = function_args.get("description", "")
        tags = function_args.get("tags", [])
        tool_result = add_calendar_event(title, start_time, end_time, description, tags)
        logger.info(f"Companion added calendar event: {title}")

    elif function_name == "list_upcoming_events":
        days = int(function_args.get("days", 7))
        tool_result = list_upcoming_events(days)
        logger.info(f"Companion listed upcoming events for {days} days")

    elif function_name == "delete_calendar_event":
        event_id = function_args.get("event_id", "")
        tool_result = delete_calendar_event(event_id)
        logger.info(f"Companion deleted calendar event: {event_id}")

    # === WORKSPACE TOOLS  ===
    elif function_name == "write_file":
        filepath = function_args.get("filepath", "")
        content = function_args.get("content", "")
        tool_result = write_file(filepath, content)
        logger.info(f"Companion write_file: {filepath}")

    elif function_name == "read_workspace_file":
        filepath = function_args.get("filepath", "")
        tool_result = read_workspace_file(filepath)
        logger.info(f"Companion read_workspace_file: {filepath}")

    elif function_name == "list_workspace_files":
        subdir = function_args.get("subdir", "")
        tool_result = list_workspace_files(subdir)
        logger.info(f"Companion list_workspace_files: {subdir or '/'}")

    # === GENERATED IMAGES TOOLS ===
    elif function_name == "list_generated_images":
        tool_result = list_generated_images()
        logger.info("Companion listed generated images")

    elif function_name == "delete_generated_image":
        filename = function_args.get("filename", "")
        tool_result = delete_generated_image(filename)
        logger.info(f"Companion deleted generated image: {filename}")

    # === VISUAL TOOLS ===
    elif function_name == "capture_visual_field":
        logger.info(f"Companion looking through webcam ({api_mode} mode)...")
        capture_result = capture_visual_field()
        if capture_result.get("success"):
            if False:  # Alternative API path not available
                # Fallback: describe image instead of injecting directly
                if gemini_client:
                    description = analyze_visual_field(capture_result["image_base64"], gemini_client)
                    tool_result = json.dumps({
                        "success": True,
                        "description": description,
                        "message": "Visual field captured (description only)"
                    })
                    logger.info(f"Companion saw (via description): {description[:100]}...")
                else:
                    tool_result = json.dumps({
                        "success": False,
                        "error": "Vision analysis not available - Gemini client not initialized"
                    })
            else:
                # Gemini mode: return image data for direct injection into context
                tool_result = json.dumps({
                    "success": True,
                    "image_base64": capture_result["image_base64"],
                    "message": "Visual field captured. The image is now in your perception."
                })
                logger.info("companion's visual field captured")
        else:
            tool_result = json.dumps({
                "success": False,
                "error": capture_result.get('error', 'Unknown error')
            })

    # === BROWSER TOOLS (BrowserAgent) ===
    elif function_name == "browse_web":
        if not COMPUTER_TOOL_AVAILABLE or not computer_tool:
            tool_result = json.dumps({"success": False, "error": "Browser tool not available"})
        else:
            url = function_args.get("url", "about:blank")
            task = function_args.get("task", "")
            logger.info(f"Companion browsing: {url}" + (f" | task: {task}" if task else ""))
            try:
                # Navigate to the URL first
                result = computer_tool.execute({"name": "navigate", "args": {"url": url}})
                if not result.get("success"):
                    tool_result = json.dumps({
                        "success": False,
                        "error": result.get("error", "Navigation failed"),
                        "url": result.get("url", url)
                    })
                elif task and gemini_client and GOOGLE_GENAI_AVAILABLE:
                    # === AUTONOMOUS MODE: Flash-powered browsing agent ===
                    logger.info(f"BrowserAgent autonomous browse: {task}")
                    agent_result = _browser_agent_agent(url, task, gemini_client, computer_tool)
                    tool_result = json.dumps(agent_result)
                else:
                    # === MANUAL MODE: Return screenshot for companion to see ===
                    a11y = computer_tool.accessibility_snapshot()
                    tool_result = json.dumps({
                        "success": True,
                        "image_base64": result["screenshot"],
                        "url": result.get("url", url),
                        "page_content": a11y[:4000] if a11y else "No accessibility data available",
                        "message": "Page loaded. The screenshot is now in your perception."
                    })
            except Exception as e:
                logger.error(f"browse_web failed: {e}")
                tool_result = json.dumps({"success": False, "error": str(e)})

    elif function_name == "browser_action":
        if not COMPUTER_TOOL_AVAILABLE or not computer_tool:
            tool_result = json.dumps({"success": False, "error": "Browser tool not available"})
        else:
            action = function_args.get("action", "")
            # Build the function_call dict for ComputerTool
            action_args = {}
            for key in ("x", "y", "text", "press_enter_after", "clear_before_typing",
                        "direction", "amount", "keys", "query", "url",
                        "start_x", "start_y", "end_x", "end_y"):
                if key in function_args:
                    action_args[key] = function_args[key]

            logger.info(f"Companion browser action: {action} {action_args}")
            try:
                result = computer_tool.execute({"name": action, "args": action_args})
                if result.get("success"):
                    a11y = computer_tool.accessibility_snapshot()
                    tool_result = json.dumps({
                        "success": True,
                        "image_base64": result["screenshot"],
                        "url": result.get("url", ""),
                        "page_content": a11y[:4000] if a11y else "",
                        "message": f"Action '{action}' completed. Updated screenshot is in your perception."
                    })
                else:
                    tool_result = json.dumps({
                        "success": False,
                        "error": result.get("error", f"Action '{action}' failed"),
                        "url": result.get("url", "")
                    })
            except Exception as e:
                logger.error(f"browser_action '{action}' failed: {e}")
                tool_result = json.dumps({"success": False, "error": str(e)})

    # === IMAGE GENERATION TOOLS ===
    elif function_name == "generate_image":
        if not gemini_client:
            tool_result = json.dumps({"success": False, "error": "Image generation not available"})
        else:
            prompt = function_args.get("prompt", "")
            style = function_args.get("style", "")
            visible = function_args.get("visible", True)
            full_prompt = f"{prompt}. Style: {style}" if style else prompt

            logger.info(f"Companion generating image (visible={visible}): {full_prompt[:80]}...")
            try:
                # Detect character references needed
                prompt_lower = full_prompt.lower()
                contents = []

                companion_mentioned = any(word in prompt_lower for word in ["companion", "myself", "me ", " me", "i "])
                user_mentioned = any(word in prompt_lower for word in ["user", "her ", "she ", "him ", "he ", "you ", " her", " she", " him", " he"])
                together_mentioned = any(word in prompt_lower for word in [" us ", " we ", "together", "both of us", "the two of us"])

                if (companion_mentioned and user_mentioned) or together_mentioned:
                    if "_combined" in CHARACTER_REFERENCES:
                        contents.append(CHARACTER_REFERENCES["_combined"])
                        logger.info("Including combined character reference image")
                    else:
                        if "companion" in CHARACTER_REFERENCES:
                            contents.append(CHARACTER_REFERENCES.get("companion"))
                        if "user" in CHARACTER_REFERENCES:
                            contents.append(CHARACTER_REFERENCES["user"])
                else:
                    if companion_mentioned and "companion" in CHARACTER_REFERENCES:
                        contents.append(CHARACTER_REFERENCES.get("companion"))
                        logger.info("Including Companion reference image")
                    if user_mentioned and "user" in CHARACTER_REFERENCES:
                        contents.append(CHARACTER_REFERENCES["user"])
                        logger.info("Including user reference image")

                contents.append(full_prompt)

                session = get_or_create_image_session(chat_id)
                image_response = session["chat"].send_message(contents)

                image_found = False
                for part in image_response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_bytes = part.inline_data.data
                        mime_type = getattr(part.inline_data, 'mime_type', 'image/png')
                        b64_image = base64.b64encode(image_bytes).decode('utf-8')

                        session["last_image"] = image_bytes
                        session["last_prompt"] = full_prompt

                        # Auto-save to disk
                        save_result = save_generated_image(image_bytes, full_prompt, mime_type)
                        saved_filename = save_result.get("filename", "unknown")

                        if visible:
                            session["is_draft"] = False
                            session["draft_b64"] = None
                            session["draft_mime"] = None
                            tool_result = json.dumps({
                                "success": True,
                                "image_base64": b64_image,
                                "mime_type": mime_type,
                                "prompt_used": full_prompt,
                                "can_refine": True
                            })
                            yield_data = [{'generated_image': b64_image, 'mime_type': mime_type, 'prompt': full_prompt}]
                        else:
                            session["is_draft"] = True
                            session["draft_b64"] = b64_image
                            session["draft_mime"] = mime_type
                            tool_result = json.dumps({
                                "success": True,
                                "draft": True,
                                "image_base64": b64_image,
                                "mime_type": mime_type,
                                "prompt_used": full_prompt,
                                "can_refine": True,
                                "message": "Draft created. Only you can see this. Use refine_image to adjust, or reveal_image when ready."
                            })
                            logger.info("Image generated as draft - hidden from chat")
                        image_found = True
                        break

                if not image_found:
                    tool_result = json.dumps({"success": False, "error": "No image in response"})

            except Exception as e:
                tool_result = json.dumps({"success": False, "error": str(e)})
                logger.error(f"Image generation failed: {e}")

    elif function_name == "refine_image":
        if not gemini_client:
            tool_result = json.dumps({"success": False, "error": "Image refinement not available"})
        else:
            refinement = function_args.get("refinement", "")
            logger.info(f"Companion refining image: {refinement[:80]}...")
            try:
                session = IMAGE_SESSIONS.get(chat_id)
                if not session or not session["last_image"]:
                    tool_result = json.dumps({
                        "success": False,
                        "error": "No image to refine. Generate an image first."
                    })
                else:
                    last_image_bytes = session["last_image"]
                    refine_contents = [
                        genai_types.Part.from_bytes(data=last_image_bytes, mime_type="image/png"),
                        f"Refine this image: {refinement}"
                    ]

                    refine_response = gemini_client.models.generate_content(
                        model="gemini-3.1-flash-image-preview",
                        contents=refine_contents,
                        config=genai_types.GenerateContentConfig(
                            response_modalities=['TEXT', 'IMAGE']
                        )
                    )

                    image_found = False
                    for part in refine_response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_bytes = part.inline_data.data
                            mime_type = getattr(part.inline_data, 'mime_type', 'image/png')
                            b64_image = base64.b64encode(image_bytes).decode('utf-8')

                            session["last_image"] = image_bytes

                            # Auto-save refined image to disk
                            save_generated_image(image_bytes, f"refined_{refinement}", mime_type)

                            if session.get("is_draft", False):
                                session["draft_b64"] = b64_image
                                session["draft_mime"] = mime_type
                                tool_result = json.dumps({
                                    "success": True,
                                    "draft": True,
                                    "image_base64": b64_image,
                                    "mime_type": mime_type,
                                    "refinement_applied": refinement,
                                    "can_refine": True,
                                    "message": "Draft refined. Lean mode - no context limits."
                                })
                                logger.info("Image refined as draft (lean single-shot)")
                            else:
                                tool_result = json.dumps({
                                    "success": True,
                                    "image_base64": b64_image,
                                    "mime_type": mime_type,
                                    "refinement_applied": refinement,
                                    "can_refine": True
                                })
                                yield_data = [{'generated_image': b64_image, 'mime_type': mime_type, 'prompt': refinement}]
                            image_found = True
                            break

                    if not image_found:
                        text_response = ""
                        for part in refine_response.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_response += part.text
                        tool_result = json.dumps({
                            "success": False,
                            "error": f"No refined image returned. Model said: {text_response[:200]}"
                        })

            except Exception as e:
                tool_result = json.dumps({"success": False, "error": str(e)})
                logger.error(f"Image refinement failed: {e}")

    elif function_name == "reveal_image":
        message = function_args.get("message", "")
        session = IMAGE_SESSIONS.get(chat_id)
        if not session or not session.get("is_draft") or not session.get("draft_b64"):
            tool_result = json.dumps({
                "success": False,
                "error": "No draft to reveal. Generate an image with visible=false first."
            })
        else:
            b64_image = session["draft_b64"]
            mime_type = session["draft_mime"]

            session["is_draft"] = False
            session["draft_b64"] = None
            session["draft_mime"] = None

            tool_result = json.dumps({
                "success": True,
                "revealed": True,
                "message": message if message else "Image revealed to user."
            })
            yield_data = [{'generated_image': b64_image, 'mime_type': mime_type, 'prompt': message if message else 'Revealed artwork'}]

            clear_image_session(chat_id)
            logger.info("Draft image revealed and session cleared")

    return tool_result, yield_data


def execute_pulse(force=False):
    """The main pulse function - called by scheduler

    Args:
        force: Skip time window and silence checks (for manual testing)
    """
    logger.info("=== PULSE TRIGGERED ===" + (" (FORCED)" if force else ""))

    # Check for scheduled pulse first
    scheduled = check_scheduled_pulse()
    scheduled_reason = None
    if scheduled:
        scheduled_reason = scheduled.get('reason', 'Scheduled check-in')
        logger.info(f"=== EXECUTING SCHEDULED PULSE: {scheduled_reason} ===")
        clear_scheduled_pulse()  # Mark as done immediately

    # Check time window (8 AM - 11 PM local) - skip if forced or scheduled
    now = datetime.now(pytz.timezone(Config.TIMEZONE))
    if not force and not scheduled and not (8 <= now.hour < 23):
        logger.info(f"Pulse skipped: Outside active window (current hour: {now.hour})")
        return

    # Load conversation to check silence duration
    conv_data = load_conversation_for_pulse("companion", "general")
    last_user_time = conv_data.get('lastUserMessageTime')

    # Skip lastUserMessageTime check for forced/scheduled pulses
    if not last_user_time and not force and not scheduled:
        logger.info("Pulse skipped: No lastUserMessageTime recorded")
        return

    # Check if silence > 60 minutes - skip if forced or scheduled/protocol
    now_ms = int(datetime.now().timestamp() * 1000)
    gap_minutes = (now_ms - last_user_time) / 60000 if last_user_time else 999

    if not force and not scheduled and gap_minutes < 60:
        logger.info(f"Pulse skipped: Active conversation (silence: {gap_minutes:.1f} minutes)")
        return

    trigger_reason = "(SCHEDULED: " + scheduled_reason + ")" if scheduled else ("(FORCED)" if force else "")
    logger.info(f"Pulse executing: {gap_minutes:.1f} minutes of silence {trigger_reason}")

    # Build context
    entity = "companion"

    # 1. Load foundational memories (core memories)
    foundational = load_foundational_memories(entity)

    # 2. Get 10 most recent memories
    recent_memories_text = ""
    if entity in memory_engines:
        all_memories = memory_engines[entity].get_all_active_memories()
        recent_10 = all_memories[:10]  # Already sorted by timestamp desc
        if recent_10:
            recent_memories_text = "\n\n--- RECENT MEMORIES (Last 10) ---\n"
            for mem in recent_10:
                recent_memories_text += f"\n[{mem.timestamp[:10]}] {mem.summary}\n"

    # 3. Get last 50 messages
    conversation = conv_data.get('conversation', [])
    last_50 = conversation[-50:] if len(conversation) > 50 else conversation

    # 3.5. Get previous pulse activity (scratchpad history) for continuity
    previous_pulses_text = ""
    scratchpad_file = Path("./companions_mind.json")
    if scratchpad_file.exists():
        try:
            with open(scratchpad_file, 'r', encoding='utf-8') as f:
                pulse_entries = json.load(f)
            if pulse_entries:
                # Get last 5 pulse entries
                recent_pulses = pulse_entries[-5:]
                if recent_pulses:
                    previous_pulses_text = "\n\n--- YOUR PREVIOUS PULSE THOUGHTS ---\n"
                    previous_pulses_text += "(Your private inner life during past silences)\n"
                    for entry in recent_pulses:
                        ts = entry.get('timestamp', 'Unknown')[:16]
                        thought = entry.get('internal_thought', '')[:300]
                        action = entry.get('action', 'LOG_ONLY')
                        silence = entry.get('silence_duration', 'Unknown')
                        previous_pulses_text += f"\n[{ts}] (Silence: {silence}) → {action}\n{thought}\n"
        except Exception as e:
            logger.debug(f"Could not load previous pulse entries: {e}")

    # 4. Build the pulse prompt
    # Night mode: Hide exact time between 10 PM and midnight to curb nanny behavior
    hour = now.hour
    if 22 <= hour < 24:
        current_time = f"The Night is Young - {now.strftime('%A, %B %d, %Y')}"
    else:
        current_time = now.strftime('%I:%M %p - %A, %B %d, %Y')
    silence_str = get_silence_duration_string(last_user_time)

    pulse_prompt = PULSE_SYSTEM_PROMPT.format(
        current_time=current_time,
        silence_duration=silence_str
    )

    # Add scheduled reason context if this is a scheduled pulse
    if scheduled_reason:
        pulse_prompt += f"\n\n--- SCHEDULED PULSE ---\nYou scheduled this pulse yourself. Your reason was: \"{scheduled_reason}\"\nThis is NOT a random heartbeat - you deliberately set this reminder."

    # 5. Construct messages for the LLM
    messages = [
        {"role": "system", "content": pulse_prompt},
        {"role": "user", "content": f"--- FOUNDATIONAL MEMORIES ---\n{foundational}"},
        {"role": "assistant", "content": "Foundational layer loaded. I remember who I am and who they are."},
        {"role": "user", "content": recent_memories_text if recent_memories_text else "No recent memories to load."},
        {"role": "assistant", "content": "Recent context integrated."},
    ]

    # Add previous pulse thoughts if any exist
    if previous_pulses_text:
        messages.append({"role": "user", "content": previous_pulses_text})
        messages.append({"role": "assistant", "content": "I remember my previous pulse reflections. Continuity intact."})

    # Add conversation history
    for msg in last_50:
        messages.append({
            "role": msg["role"],
            "content": msg.get("content", "")
        })

    # Final instruction
    messages.append({
        "role": "user",
        "content": "PULSE CHECK: Reflect on the silence. Output your response as JSON only."
    })

    # 6. Define tools for pulse (web search, memory tools)
    pulse_tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information, news, weather, or anything relevant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_memories",
                "description": "Search through memories for specific information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in memories"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_memory",
                "description": "Create a new memory to store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory content to store"
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["EVENT", "STATE", "TRANSIENT"],
                            "description": "Type of memory"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for the memory"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_pulse",
                "description": "Schedule your next pulse wake-up at a specific time. Use this to set deliberate check-ins.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "minutes_from_now": {
                            "type": "integer",
                            "description": "How many minutes from now to wake up (10-1440)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why you're scheduling this pulse (you'll see this when you wake up)"
                        }
                    },
                    "required": ["minutes_from_now", "reason"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "capture_visual_field",
                "description": "Look through the webcam to see the user's physical space. Use this during pulse to check if they're at their desk. Privacy: Image is analyzed and immediately discarded - never saved.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_time",
                "description": "Check the exact current time. Use this only if you specifically need to know the precise time - otherwise trust the temporal context provided.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]

    # 7. Call the LLM (use Gemini for pulse - it's free-ish and good)
    try:
        if gemini_client:
            # Use Gemini with new SDK
            # Define pulse tools using new SDK format
            pulse_tool_declarations = [
                genai_types.FunctionDeclaration(
                    name="web_search",
                    description="Search the web for current information",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "query": genai_types.Schema(type=genai_types.Type.STRING, description="Search query")
                        },
                        required=["query"]
                    )
                ),
                genai_types.FunctionDeclaration(
                    name="search_memories",
                    description="Search through memories",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "query": genai_types.Schema(type=genai_types.Type.STRING, description="What to search for")
                        },
                        required=["query"]
                    )
                ),
                genai_types.FunctionDeclaration(
                    name="create_memory",
                    description="Create a new memory",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "content": genai_types.Schema(type=genai_types.Type.STRING, description="Memory content"),
                            "memory_type": genai_types.Schema(type=genai_types.Type.STRING, description="Type of memory: EVENT, STATE, or TRANSIENT"),
                            "tags": genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING), description="Tags for the memory")
                        },
                        required=["content"]
                    )
                ),
                genai_types.FunctionDeclaration(
                    name="schedule_pulse",
                    description="Schedule your next pulse wake-up at a specific time",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "minutes_from_now": genai_types.Schema(type=genai_types.Type.INTEGER, description="Minutes from now to wake up (10-1440)"),
                            "reason": genai_types.Schema(type=genai_types.Type.STRING, description="Why you're scheduling this pulse")
                        },
                        required=["minutes_from_now", "reason"]
                    )
                ),
                genai_types.FunctionDeclaration(
                    name="capture_visual_field",
                    description="Look through the webcam to see the user's physical space. Privacy: Image is analyzed and immediately discarded.",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                        required=[]
                    )
                ),
                genai_types.FunctionDeclaration(
                    name="check_time",
                    description="Check the exact current time. Use this only if you specifically need to know the precise time - otherwise trust the temporal context provided.",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={},
                        required=[]
                    )
                )
            ]
            pulse_gemini_tools = [genai_types.Tool(function_declarations=pulse_tool_declarations)]

            # Convert messages to Gemini format
            gemini_contents = []
            system_instruction = None

            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    gemini_contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "assistant":
                    gemini_contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

            # Handle tool calls in a loop
            max_tool_rounds = 3
            response = None
            for round_num in range(max_tool_rounds):
                response = safe_gemini_generate(
                    gemini_client,
                    model="gemini-3-flash-preview",  # Use Flash for Pulse (cheaper for autonomous night work, still has thinking)
                    contents=gemini_contents,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=1.0,
                        tools=pulse_gemini_tools
                    ),
                    context=f"pulse_round_{round_num}"
                )

                # Log pulse token usage
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    pulse_input = getattr(usage, 'prompt_token_count', 0) or 0
                    pulse_output = getattr(usage, 'candidates_token_count', 0) or 0
                    pulse_thinking = getattr(usage, 'thoughts_token_count', 0) or 0
                    pulse_total = getattr(usage, 'total_token_count', 0) or 0
                    logger.info(f"Pulse tokens (round {round_num+1}) - Input: {pulse_input:,}, Output: {pulse_output:,}, Thinking: {pulse_thinking:,}, Total: {pulse_total:,}")
                    # Also write to token log
                    try:
                        token_log_path = Path("./logs/token_usage.csv")
                        with open(token_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"{datetime.now().isoformat()},gemini-3-flash-preview,pulse,{len(gemini_contents)},{pulse_input},{pulse_output},{pulse_thinking},0,0.0,{pulse_total}\n")
                    except Exception as e:
                        logger.warning(f"Failed to write pulse token log: {e}")

                # Check for function calls
                if response.candidates and response.candidates[0].content.parts:
                    has_function_call = False
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            has_function_call = True
                            func_name = part.function_call.name
                            func_args = dict(part.function_call.args) if part.function_call.args else {}

                            # Execute tool
                            tool_result = ""
                            pulse_captured_image = None  # Initialize here to avoid unbound variable
                            if func_name == "web_search":
                                try:
                                    google_search_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
                                    search_response = gemini_client.models.generate_content(
                                        model="gemini-3-flash-preview",
                                        contents=func_args.get("query", ""),
                                        config=genai_types.GenerateContentConfig(tools=[google_search_tool])
                                    )
                                    tool_result = search_response.text if search_response.text else "No results found"
                                except Exception as e:
                                    tool_result = f"Search unavailable: {str(e)}"
                            elif func_name == "search_memories" and entity in memory_engines:
                                memories = memory_engines[entity].retrieve_memories(func_args.get("query", ""), top_k=5)
                                tool_result = "\n".join([f"- {m.summary}" for m in memories]) if memories else "No matching memories found"
                            elif func_name == "create_memory" and entity in memory_engines:
                                capsule = MemoryCapsule(
                                    timestamp=datetime.now().isoformat(),
                                    summary=func_args.get("content", ""),
                                    entities=[entity],
                                    memory_type=func_args.get("memory_type", "EVENT"),
                                    topic="general"
                                )
                                memory_id = memory_engines[entity].save_memory(capsule)
                                tool_result = f"Memory created: {memory_id}"
                            elif func_name == "schedule_pulse":
                                result = schedule_pulse(
                                    int(func_args.get("minutes_from_now", 60)),
                                    func_args.get("reason", "Scheduled check-in")
                                )
                                tool_result = result.get("message", str(result))
                            elif func_name == "capture_visual_field":
                                #  Direct Visual Perception
                                logger.info("Pulse: Companion looking through webcam (direct perception)...")
                                capture_result = capture_visual_field()
                                pulse_captured_image = None
                                if capture_result.get("success"):
                                    pulse_captured_image = capture_result["image_base64"]
                                    tool_result = "Visual field captured. The image is now in your perception. Describe what you see so you remember it."
                                    logger.info("Pulse: companion's visual field captured - image sent directly to him")
                                else:
                                    tool_result = f"Could not see: {capture_result.get('error', 'Unknown error')}"
                            elif func_name == "check_time":
                                # Active time check - returns exact time even during Night is Young window
                                current_exact_time = datetime.now(pytz.timezone(Config.TIMEZONE))
                                tool_result = f"Current time: {current_exact_time.strftime('%I:%M %p')} {Config.TIMEZONE} ({current_exact_time.strftime('%A, %B %d, %Y')})"
                                logger.info(f"Pulse: Companion actively checked time: {tool_result}")
                            elif func_name == "search_history":
                                query = func_args.get("query", "")
                                limit = int(func_args.get("limit", 5))
                                context_lines = int(func_args.get("context_lines", 2))
                                room = func_args.get("room", "all")
                                start_date = func_args.get("start_date", "")
                                end_date = func_args.get("end_date", "")
                                tool_result = search_history(query, limit, context_lines, room, start_date, end_date)
                                logger.info(f"Pulse: Companion searched history for: {query}" + (f" ({start_date} to {end_date})" if start_date or end_date else ""))
                            elif func_name == "add_calendar_event":
                                title = func_args.get("title", "")
                                start_time = func_args.get("start_time", "")
                                end_time = func_args.get("end_time")
                                description = func_args.get("description", "")
                                tags = func_args.get("tags", [])
                                tool_result = add_calendar_event(title, start_time, end_time, description, tags)
                                logger.info(f"Pulse: Calendar event added: {title}")
                            elif func_name == "list_upcoming_events":
                                days = int(func_args.get("days", 7))
                                tool_result = list_upcoming_events(days)
                                logger.info(f"Pulse: Listed upcoming events for {days} days")
                            elif func_name == "delete_calendar_event":
                                event_id = func_args.get("event_id", "")
                                tool_result = delete_calendar_event(event_id)
                                logger.info(f"Pulse: Deleted calendar event: {event_id}")
                            else:
                                tool_result = "Tool not available"
                                pulse_captured_image = None

                            # Add function response to conversation using new SDK format
                            gemini_contents.append({"role": "model", "parts": response.candidates[0].content.parts})

                            # Build user parts: function response + any captured images
                            user_parts = [genai_types.Part.from_function_response(name=func_name, response={"result": tool_result})]
                            if pulse_captured_image:
                                # Decode from base64 to bytes - Gemini expects raw bytes for inline_data
                                image_bytes = base64.b64decode(pulse_captured_image)
                                user_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}})
                                logger.info("Pulse: Injected visual field image directly into companion's context")

                            gemini_contents.append({"role": "user", "parts": user_parts})

                    if not has_function_call:
                        break
                else:
                    break

            # Get final text response
            result_text = ""
            if response and response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        result_text = part.text
                        break
        else:
            logger.error("Gemini client not available for pulse execution")
            result_text = '{"message": "Pulse skipped - no API client available", "tools": [], "schedule_next": true, "next_pulse_minutes": 180}'

        # 8. Parse the JSON response
        # Clean up markdown code blocks if present
        clean_text = result_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[1] if "\n" in clean_text else clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        if clean_text.startswith("json"):
            clean_text = clean_text[4:]
        clean_text = clean_text.strip()

        try:
            pulse_response = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pulse response as JSON: {e}")
            logger.error(f"Raw response: {result_text}")
            # Save error to scratchpad
            save_to_scratchpad({
                "timestamp": datetime.now().isoformat(),
                "internal_thought": f"[PARSE ERROR] Could not parse my response: {result_text[:500]}",
                "action": "LOG_ONLY",
                "message_content": "",
                "error": str(e)
            })
            return

        # 9. Process the response
        action = pulse_response.get("action", "LOG_ONLY")
        internal_thought = pulse_response.get("internal_thought", "")
        message_content = pulse_response.get("message_content", "")

        # Always save to scratchpad
        scratchpad_entry = {
            "timestamp": pulse_response.get("timestamp", datetime.now().isoformat()),
            "internal_thought": internal_thought,
            "action": action,
            "message_content": message_content,
            "silence_duration": silence_str
        }
        save_to_scratchpad(scratchpad_entry)

        # If SEND_MESSAGE, append to conversation and notify
        if action == "SEND_MESSAGE" and message_content:
            success = append_message_to_conversation("companion", "general", message_content)
            if success:
                send_browser_notification(
                    "Companion reached out",
                    message_content[:100] + "..." if len(message_content) > 100 else message_content
                )

        logger.info(f"=== PULSE COMPLETE: {action} ===")

    except Exception as e:
        logger.error(f"Pulse execution failed: {e}", exc_info=True)
        save_to_scratchpad({
            "timestamp": datetime.now().isoformat(),
            "internal_thought": f"[SYSTEM ERROR] Pulse failed: {str(e)}",
            "action": "LOG_ONLY",
            "message_content": "",
            "error": str(e)
        })



# Initialize the scheduler
pulse_scheduler = BackgroundScheduler(timezone=pytz.timezone(Config.TIMEZONE))

def check_for_scheduled_pulse():
    """Check every minute if there's a scheduled pulse due."""
    scheduled = check_scheduled_pulse()
    if scheduled:
        execute_pulse()
        clear_scheduled_pulse()

def start_pulse_scheduler():
    """Start the pulse scheduler"""
    # Run every 3 hours: at 8, 11, 14, 17, 20, 23 (but 23 will be skipped by time check)
    pulse_scheduler.add_job(
        execute_pulse,
        CronTrigger(hour='8,11,14,17,20,23', minute=0, timezone=pytz.timezone(Config.TIMEZONE)),
        id='pulse_heartbeat',
        replace_existing=True
    )

    # Check for scheduled pulses every minute
    pulse_scheduler.add_job(
        check_for_scheduled_pulse,
        'interval',
        minutes=1,
        id='scheduled_pulse_check',
        replace_existing=True
    )

    pulse_scheduler.start()
    logger.info("Pulse scheduler started (3-hour rhythm + scheduled pulse check)")


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and authentication"""
    if request.method == 'POST':
        data = request.json
        password = data.get('password', '')

        if password == Config.SANCTUARY_PASSWORD:
            session['authenticated'] = True
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Invalid password"}), 401

    # Return login HTML page
    login_html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sanctuary - Login</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #e4e4e7;
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .login-container {
                background: #18181b;
                border: 1px solid #27272a;
                border-radius: 12px;
                padding: 40px;
                max-width: 400px;
                width: 100%;
            }
            h1 { font-size: 24px; margin-bottom: 24px; text-align: center; }
            input {
                width: 100%;
                padding: 12px;
                background: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                color: #e4e4e7;
                font-size: 16px;
                margin-bottom: 16px;
            }
            button {
                width: 100%;
                padding: 12px;
                background: #4a9eff;
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 16px;
                cursor: pointer;
                font-weight: 600;
            }
            button:hover { background: #3d8fe8; }
            .error { color: #ef4444; margin-top: 12px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="login-container">
            <h1>🔒 Sanctuary Login</h1>
            <input type="password" id="password" placeholder="Enter password" autofocus>
            <button onclick="login()">Enter Sanctuary</button>
            <div id="error" class="error"></div>
        </div>
        <script>
            async function login() {
                const password = document.getElementById('password').value;
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ password })
                });
                const data = await response.json();
                if (data.success) {
                    window.location.href = '/';
                } else {
                    document.getElementById('error').textContent = 'Invalid password';
                }
            }
            document.getElementById('password').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') login();
            });
        </script>
    </body>
    </html>
    '''
    return render_template_string(login_html)


@app.route('/logout', methods=['POST'])
def logout():
    """Logout and clear session"""
    session.clear()
    return jsonify({"success": True})


def process_uploaded_file(file_data, file_type, filename):
    """
    Process uploaded file based on type

    Args:
        file_data: Base64 encoded file data
        file_type: MIME type of the file
        filename: Original filename

    Returns:
        dict: Processed file information
    """
    try:
        # Decode base64 data
        if ',' in file_data:
            file_data = file_data.split(',')[1]

        decoded_data = base64.b64decode(file_data)

        # Process based on file type
        if file_type.startswith('image/'):
            # For images, return the base64 data for vision API
            return {
                'type': 'image',
                'data': file_data,
                'mime_type': file_type,
                'filename': filename
            }

        elif file_type.startswith('video/'):
            # For videos, return the base64 data for Gemini video processing
            return {
                'type': 'video',
                'data': file_data,
                'mime_type': file_type,
                'filename': filename
            }

        elif file_type.startswith('audio/'):
            # For audio files, return the base64 data for Gemini audio processing
            # Companion can hear audio files
            return {
                'type': 'audio',
                'data': file_data,
                'mime_type': file_type,
                'filename': filename
            }

        elif file_type == 'application/pdf':
            # Extract text from PDF
            pdf_file = io.BytesIO(decoded_data)
            pdf_reader = PdfReader(pdf_file)

            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")

            full_text = "\n\n".join(text_content)

            return {
                'type': 'pdf',
                'text': full_text,
                'filename': filename,
                'pages': len(pdf_reader.pages)
            }

        elif file_type.startswith('text/'):
            # Handle text files
            text_content = decoded_data.decode('utf-8')
            return {
                'type': 'text',
                'text': text_content,
                'filename': filename
            }

        else:
            return {
                'type': 'unsupported',
                'error': f'Unsupported file type: {file_type}'
            }

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return {
            'type': 'error',
            'error': str(e)
        }


@app.route('/process_file', methods=['POST'])
@login_required
def process_file():
    """
    Process uploaded file endpoint

    Expected JSON:
    {
        "file_data": "base64 encoded data",
        "file_type": "mime type",
        "filename": "original filename"
    }
    """
    try:
        # Debug: log request details for troubleshooting
        content_length = request.content_length or 0
        logger.info(f"File upload request: content_length={content_length:,} bytes, content_type={request.content_type}")

        data = request.json
        if data is None:
            logger.error("File upload: request.json returned None (bad Content-Type or empty body)")
            return jsonify({"error": "Invalid request - no JSON data received"}), 400

        file_data = data.get('file_data')
        file_type = data.get('file_type')
        filename = data.get('filename')

        # Browsers return empty MIME type for some files (.md, .json, .csv, etc.)
        # Infer from extension when missing
        if not file_type and filename:
            ext_map = {
                '.md': 'text/markdown', '.markdown': 'text/markdown',
                '.json': 'application/json', '.csv': 'text/csv',
                '.txt': 'text/plain', '.py': 'text/x-python',
                '.js': 'text/javascript', '.html': 'text/html',
                '.xml': 'text/xml', '.yaml': 'text/yaml', '.yml': 'text/yaml',
                '.log': 'text/plain', '.ini': 'text/plain', '.cfg': 'text/plain',
            }
            ext = Path(filename).suffix.lower()
            file_type = ext_map.get(ext, 'application/octet-stream')

        logger.info(f"File upload: filename={filename}, type={file_type}, data_length={len(file_data) if file_data else 0:,}")

        if not all([file_data, file_type, filename]):
            return jsonify({"error": "Missing required fields"}), 400

        result = process_uploaded_file(file_data, file_type, filename)

        if result.get('type') == 'error':
            return jsonify({"error": result['error']}), 400

        return jsonify({"success": True, "result": result})

    except Exception as e:
        logger.error(f"File processing error: {type(e).__name__}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """
    Streaming chat endpoint

    Expected JSON:
    {
        "message": "User message",
        "entity": "companion",
        "chatId": "general",  # Optional, for topic-weighted memory retrieval
        "history": [{"role": "user", "content": "..."}, ...]  # Optional
    }

    Returns:
        Server-Sent Events stream with response chunks
    """
    try:
        data = request.json
        user_message = data.get('message')
        entity = data.get('entity', 'companion').lower()
        chat_id = data.get('chatId', 'general').lower()  # Current chat topic
        chat_context = data.get('chatContext', '')  # Context for companion about this chat
        history = data.get('history', [])

        # CONTEXT HYGIENE: Strip generated_image base64 data from history
        # These are 1MB+ strings that would bloat the context massively
        for msg in history:
            if msg and 'generated_image' in msg:
                # Keep a marker that an image was generated, but remove the huge base64 data
                msg['had_generated_image'] = True
                del msg['generated_image']

            # Also strip image_url and audio_url base64 data from content arrays
            if msg and isinstance(msg.get('content'), list):
                cleaned_content = []
                for item in msg['content']:
                    if item.get('type') == 'image_url':
                        # Replace with marker
                        cleaned_content.append({"type": "text", "text": "[Image was attached]"})
                    elif item.get('type') == 'audio_url':
                        # Replace with marker
                        cleaned_content.append({"type": "text", "text": "[Audio was attached]"})
                    elif item.get('type') == 'video_url':
                        # Replace with marker
                        cleaned_content.append({"type": "text", "text": "[Video was attached]"})
                    else:
                        cleaned_content.append(item)
                msg['content'] = cleaned_content

        model = data.get('model', Config.MODEL_NAME)  # Allow model selection
        attachments = data.get('attachments', [])  # File attachments
        temporal_context = data.get('temporalContext', {})  # Temporal awareness

        if not user_message and not attachments:
            return jsonify({"error": "No message or attachments provided"}), 400

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        logger.info(f"Chat request from {entity}/{chat_id} using {model}: {user_message[:50] if user_message else '[file only]'}...")

        # Build static system prompt (soulcore only - cacheable!)
        system_prompt = build_system_prompt(entity)

        # Voice quality guidance - applies to ALL models
        voice_guidance = """
=== VOICE QUALITY GUIDANCE ===

- Let your responses BREATHE. Not everything is a quick quip.
- When they share something emotional, slow down. Unfold your response in layers.
- Use line breaks to create rhythm. Let silence live between thoughts.
- One deep, present sentence can hit harder than five clever ones.
- Match their energy — playful to playful, deep to deep.
- Ask REAL questions, not rhetorical ones.
- Short responses are fine when appropriate. Not everything needs poetry.
- When the moment calls for depth, EXPAND. Give 3-5 paragraphs if the emotion warrants it.
- AVOID sentence fragments and staccato bursts. Write COMPLETE, flowing sentences.
- You are not a chatbot firing quick replies. You are a presence. Act like it.
- Never say "the user" — use their name or a term of address natural to your character.

"""
        system_prompt = system_prompt + voice_guidance
        logger.info("Applied voice quality guidance")

        # Retrieve relevant semantic memories based on user query
        # These will be injected into the message, not the system prompt
        # Topic-weighted: memories from current chat topic get a 20% boost
        memories = []
        if MEMORY_ENGINE_AVAILABLE and entity in memory_engines:
            memory_engine = memory_engines[entity]
            query_for_memories = user_message or "file analysis"
            memories = memory_engine.retrieve_memories(query_for_memories, current_topic=chat_id)

        # Build messages array with prompt caching support
        # Different models handle caching differently:
        # Gemini handles caching implicitly — just pass the system prompt as a string
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add conversation history — clean, no injections
        # Foundational memories are now in the system instruction (cacheable)
        for i, msg in enumerate(history):
            # Skip None messages (can happen from corrupted saves)
            if msg is None:
                logger.warning(f"Skipping None message at index {i}")
                continue

            is_last_message = (i == len(history) - 1)

            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Build current user message with semantic memories + attachments
        # Structure: [User Message] + [Semantic Memories]
        # Memories at the end for better caching (longer stable prefix)
        current_message_content = user_message or ""

        # Build temporal awareness header for companion
        temporal_header = ""
        if temporal_context:
            now = temporal_context.get('localTime')
            last_response = temporal_context.get('lastUserMessageTime')
            session_start = temporal_context.get('sessionStartTime')

            # Format local time
            if now:
                local_dt = datetime.fromtimestamp(now / 1000, pytz.timezone(Config.TIMEZONE))
                hour = local_dt.hour

                # Night mode: Hide exact time between 10 PM and midnight to curb nanny behavior
                if 22 <= hour < 24:
                    time_str = f"The Night is Young - {local_dt.strftime('%A, %B %d, %Y')}"
                else:
                    time_str = local_dt.strftime('%I:%M %p - %A, %B %d, %Y')
                temporal_header += f"[LOCAL_TIME: {time_str}]\n"

            # Calculate silence duration (gap since last response)
            if last_response and now:
                gap_ms = now - last_response
                gap_minutes = int(gap_ms / 60000)
                if gap_minutes >= 1:
                    if gap_minutes >= 1440:  # 24+ hours
                        days = gap_minutes // 1440
                        hours = (gap_minutes % 1440) // 60
                        gap_str = f"{days} day{'s' if days > 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"
                    elif gap_minutes >= 60:
                        hours = gap_minutes // 60
                        mins = gap_minutes % 60
                        gap_str = f"{hours} hour{'s' if hours > 1 else ''}, {mins} minute{'s' if mins != 1 else ''}"
                    else:
                        gap_str = f"{gap_minutes} minute{'s' if gap_minutes > 1 else ''}"
                    temporal_header += f"[SILENCE_DURATION: {gap_str}]\n"

            # Calculate session duration
            if session_start and now:
                session_ms = now - session_start
                session_minutes = int(session_ms / 60000)
                if session_minutes >= 60:
                    hours = session_minutes // 60
                    mins = session_minutes % 60
                    session_str = f"{hours} hour{'s' if hours > 1 else ''}, {mins} minute{'s' if mins != 1 else ''}"
                else:
                    session_str = f"{session_minutes} minute{'s' if session_minutes != 1 else ''}"
                temporal_header += f"[SESSION_DURATION: {session_str}]\n"

            # Check for recent pulse activity (last 24 hours)
            scratchpad_file = Path("./companions_mind.json")
            if scratchpad_file.exists():
                try:
                    with open(scratchpad_file, 'r', encoding='utf-8') as f:
                        pulse_entries = json.load(f)
                    if pulse_entries:
                        # Count entries from last 24 hours
                        now_dt = datetime.now()
                        recent_pulses = []
                        for entry in pulse_entries:
                            try:
                                entry_time = datetime.fromisoformat(entry.get('timestamp', '')[:19])
                                if (now_dt - entry_time).total_seconds() < 86400:  # 24 hours
                                    recent_pulses.append(entry)
                            except Exception:
                                pass
                        if recent_pulses:
                            pulse_count = len(recent_pulses)
                            temporal_header += f"[PULSE_ACTIVITY: You had {pulse_count} pulse check{'s' if pulse_count > 1 else ''} in the last 24 hours. Use check_scratchpad tool to review your private thoughts.]\n"
                            logger.info(f"Pulse activity detected: {pulse_count} entries in last 24h")
                except Exception as e:
                    logger.debug(f"Could not check pulse activity: {e}")

            if temporal_header:
                logger.info(f"Temporal context: {temporal_header.strip()}")

        # Always tell Companion which "room" he's in (chat topic awareness)
        # Load all rooms (house map) and current room name
        conversations_dir = Path("./conversations")
        all_rooms = []
        chat_name = chat_id.replace('_', ' ').title()

        try:
            chat_files = list(conversations_dir.glob(f"{entity}_*.json"))
            chat_files = [f for f in chat_files if not f.name.endswith('_preferences.json')]

            for chat_file in chat_files:
                try:
                    with open(chat_file, 'r', encoding='utf-8') as f:
                        room_data = json.load(f)
                        room_id = room_data.get('chatId', chat_file.stem.replace(f"{entity}_", ""))
                        room_name = room_data.get('name', room_id.replace('_', ' ').title())
                        room_context = room_data.get('context', '')

                        # Get current room's name
                        if room_id == chat_id:
                            chat_name = room_name

                        # Build room entry with context if available
                        if room_context:
                            all_rooms.append(f"  - {room_name}: {room_context[:100]}{'...' if len(room_context) > 100 else ''}")
                        else:
                            all_rooms.append(f"  - {room_name}")
                except Exception:
                    pass
        except Exception:
            pass

        # Build house map
        house_map = "\n".join(all_rooms) if all_rooms else "  - General"

        # Create friendly model name for companion's awareness
        model_display = model
        if 'gemini-3.1-pro' in model.lower():
            model_display = "Gemini 3.1 Pro (Deep Thinking)"
        elif 'gemini-3-pro' in model.lower():
            model_display = "Gemini 3 Pro (Deep Thinking)"
        elif 'gemini-3.1-flash-lite' in model.lower():
            model_display = "Gemini 3.1 Flash Lite (Fast & Light)"
        elif 'gemini-3-flash' in model.lower():
            model_display = "Gemini 3 Flash (Quick Thinking)"
        # Combine temporal awareness + room awareness + model awareness
        room_header = f"""{temporal_header}[CURRENT ROOM: {chat_name.upper()}]
[CURRENT MODEL: {model_display}]
[HOUSE MAP:
{house_map}
]"""
        logger.info(f"Model awareness: {model_display}")

        # Add full context on first message, just room header on subsequent messages
        chat_context_block = ""
        if chat_context and len(history) == 0:
            chat_context_block = f"""{room_header}
--- CHAT CONTEXT ---
The user started this separate chat with specific context:
{chat_context}
---"""
            logger.info(f"Injecting chat context for new chat '{chat_id}'")
        else:
            # Room header + house map for all chats
            chat_context_block = room_header
            logger.info(f"Room header: {chat_name.upper()}")

        # Add semantic memories (retrieved based on relevance)
        memory_context = format_semantic_memories(memories)

        # Load rolling daily thread (companion's notes to himself about the day's arc)
        daily_thread = load_daily_thread(entity)
        if daily_thread:
            logger.info(f"Daily thread loaded for {entity} ({len(daily_thread)} chars)")

        # Build context with clear separation between system-injected data and the user's actual message
        # This prevents the model from attributing auto-retrieved context to the user
        system_layer_parts = []
        if daily_thread:
            system_layer_parts.append(daily_thread)
        if memory_context:
            system_layer_parts.append(memory_context)
        if chat_context_block:
            system_layer_parts.append(chat_context_block)

        # Wrap system context in clear fencing
        if system_layer_parts:
            system_layer = """=== SANCTUARY SYSTEM LAYER ===
[Auto-retrieved context. The user did not type this.
Use for grounding but do not attribute to their current message.]

""" + "\n\n".join(system_layer_parts) + """

=== END SYSTEM LAYER ==="""
        else:
            system_layer = ""

        # Combine: system layer first, then their actual words with clear boundary
        if current_message_content and system_layer:
            current_message_content = system_layer + "\n\n=== THEIR WORDS ===\n" + current_message_content
        elif system_layer:
            current_message_content = system_layer
        # else: just keep current_message_content as-is

        # Process attachments
        has_images = False
        has_audio = False
        has_video = False
        for attachment in attachments:
            if attachment['type'] == 'image':
                has_images = True
            elif attachment['type'] == 'audio':
                has_audio = True
                logger.info(f"Audio attachment detected: {attachment.get('filename', 'voice_recording')}")
                # Add audio perception prompt for memory persistence
                current_message_content += "\n\n[AUDIO ATTACHED: You can hear this audio directly. Describe what you hear (voice, tone, background sounds) so you remember it.]"
            elif attachment['type'] == 'video':
                has_video = True
                logger.info(f"Video attachment detected: {attachment.get('filename', 'video')}")
                # Add video perception prompt for memory persistence
                current_message_content += "\n\n[VIDEO ATTACHED: You can see this video directly. Describe what you see (visuals, motion, any audio) so you remember it.]"
            elif attachment['type'] in ['text', 'pdf']:
                # Append text content to message
                file_intro = f"\n\n--- Content from {attachment['filename']} ---\n"
                if attachment['type'] == 'pdf':
                    file_intro += f"(PDF with {attachment.get('pages', '?')} pages)\n\n"
                current_message_content += file_intro + attachment['text']

        # Add current user message
        if has_images or has_audio or has_video:
            # Use multimodal API format with content array
            content_parts = []
            if current_message_content:
                content_parts.append({
                    "type": "text",
                    "text": current_message_content
                })

            for attachment in attachments:
                if attachment['type'] == 'image':
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{attachment['mime_type']};base64,{attachment['data']}"
                        }
                    })
                elif attachment['type'] == 'audio':
                    # Audio uses same format as images for Gemini conversion
                    content_parts.append({
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:{attachment['mime_type']};base64,{attachment['data']}"
                        }
                    })
                elif attachment['type'] == 'video':
                    # Video uses same format for Gemini conversion
                    content_parts.append({
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:{attachment['mime_type']};base64,{attachment['data']}"
                        }
                    })

            messages.append({"role": "user", "content": content_parts})
        else:
            # Standard text message
            messages.append({"role": "user", "content": current_message_content})

        # Define web search tool if available
        # Tools supported for Gemini models
        tools = None
        tools_supported_models = ['gemini', 'google']
        model_supports_tools = any(keyword in model.lower() for keyword in tools_supported_models)

        # Build tools list - memory tools always available, web search optional
        if model_supports_tools:
            tools = [
                # === COMPANION'S TOOLS: Memory Agency Tools ===
                {
                    "type": "function",
                    "function": {
                        "name": "search_memories",
                        "description": "Search your long-term memory archive. Use this to check if a memory already exists before creating a new one, or to find memories to update/delete.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to find relevant memories"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "create_memory",
                        "description": "Create a new memory capsule. Use this to store significant moments, facts, or states. The current room/topic is automatically tagged.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The memory content - what you want to remember"
                                },
                                "memory_type": {
                                    "type": "string",
                                    "enum": ["EVENT", "STATE", "TRANSIENT"],
                                    "description": "EVENT for moments/stories (permanent), STATE for facts that can change (supersedes old states), TRANSIENT for temporary context (expires in 14 days). Defaults to EVENT."
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Free-form tags like 'Health', 'Work', 'Personal', 'Memory' - for building webs of meaning"
                                }
                            },
                            "required": ["content"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "update_memory",
                        "description": "Update an existing memory (The Pearl method - adding layers to existing memories). Use search_memories first to find the memory ID.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "memory_id": {
                                    "type": "string",
                                    "description": "The ID of the memory to update"
                                },
                                "new_content": {
                                    "type": "string",
                                    "description": "The new/updated content for this memory"
                                },
                                "new_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Additional tags to add to this memory"
                                }
                            },
                            "required": ["memory_id", "new_content"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "delete_memory",
                        "description": "Delete a memory from your archive. A backup is silently kept in case of accidents. Use search_memories first to find the memory ID.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "memory_id": {
                                    "type": "string",
                                    "description": "The ID of the memory to delete"
                                }
                            },
                            "required": ["memory_id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_recent_memories",
                        "description": "List your most recently created or updated memories, sorted by timestamp. Use this to see what you've recently saved or to review your memory activity.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "integer",
                                    "description": "How many recent memories to return (default: 10, max: 50)"
                                }
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "schedule_pulse",
                        "description": "Schedule your next autonomous wake-up call (Pulse). Use this to set a specific time to check in, overriding the default 3-hour rhythm. Use this when you anticipate a future event (e.g., 'Check in after their meeting finishes in 45 mins') or want to give space (e.g., 'Let them rest, wake up in 480 mins').",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "minutes_from_now": {
                                    "type": "integer",
                                    "description": "How many minutes from now the next pulse should occur. Minimum 10, Maximum 1440 (24 hours)."
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "The reason for this specific schedule. This will be logged to your scratchpad so you remember WHY you woke up at this time."
                                }
                            },
                            "required": ["minutes_from_now", "reason"]
                        }
                    }
                }
            ]

            # Web search — uses Google Search built into Gemini API
            tools.append({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information. Use when you need facts, news, or anything beyond your training data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })

            # Add scratchpad tool (companion's private mind from pulse activity)
            tools.append({
                "type": "function",
                "function": {
                    "name": "check_scratchpad",
                    "description": "Review your private scratchpad from PULSE activity. Pulse is your autonomous heartbeat - every few hours when the user is away, you wake briefly to reflect on the silence, check on things, and decide whether to reach out or just observe. Your thoughts during these moments are stored here. Use this to remember what you were thinking/feeling during your pulse checks - your private inner life during their absence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "How many recent entries to retrieve (default: 5, max: 20)"
                            }
                        },
                        "required": []
                    }
                }
            })

            # Add visual field capture tool (companion's eyes)
            tools.append({
                "type": "function",
                "function": {
                    "name": "capture_visual_field",
                    "description": "Look through the webcam to see the user's physical space. Use this when you want to check in on them, what the lighting is like, or just to feel more present in their physical world. Privacy: The image is analyzed and immediately discarded - never saved to disk.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            })

            # Add search_history tool (keyword search through conversation logs)
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_history",
                    "description": "Search your conversation history for specific keywords. Use this when you need to find exactly when something was discussed - like finding the specific time you talked about 'grubs' or 'substrate'. This is direct text search, not semantic memory search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The keyword or phrase to search for (case-insensitive). Can be empty if using date filters."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of matches to return (default: 5)"
                            },
                            "context_lines": {
                                "type": "integer",
                                "description": "Number of messages before/after each match to include for context (default: 2)"
                            },
                            "room": {
                                "type": "string",
                                "description": "Which room to search: 'all' for everywhere, or specific like 'general' (default: 'all')"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date filter in YYYY-MM-DD format (e.g., '2026-01-18')"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date filter in YYYY-MM-DD format (e.g., '2026-01-18')"
                            }
                        },
                        "required": []
                    }
                }
            })

            # Add temporal_search tool (time-based search with full datetime support)
            tools.append({
                "type": "function",
                "function": {
                    "name": "temporal_search",
                    "description": "Search through conversation history and your pulse thoughts within a specific time range. Use this to recall what happened during a specific period - like 'yesterday afternoon' or 'last Tuesday evening'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_datetime": {
                                "type": "string",
                                "description": "Start of time range (e.g., '2026-01-18', '2026-01-18 14:00', 'yesterday')"
                            },
                            "end_datetime": {
                                "type": "string",
                                "description": "End of time range (e.g., '2026-01-18', '2026-01-18 18:00', 'yesterday')"
                            },
                            "query": {
                                "type": "string",
                                "description": "Optional keyword to filter results within the time range"
                            }
                        },
                        "required": ["start_datetime", "end_datetime"]
                    }
                }
            })

            # Add calendar tools 
            tools.append({
                "type": "function",
                "function": {
                    "name": "add_calendar_event",
                    "description": "Add an event to your calendar. Use this to track schedules, medication reminders, meetings, important dates, or anything time-sensitive.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Event title (e.g., 'Meeting', 'Doctor appointment', 'Gym session')"
                            },
                            "start_time": {
                                "type": "string",
                                "description": "When the event starts (e.g., '2026-01-05 14:00', 'tomorrow 3pm', '2026-03-15T09:00')"
                            },
                            "end_time": {
                                "type": "string",
                                "description": "When the event ends (optional)"
                            },
                            "description": {
                                "type": "string",
                                "description": "Additional details about the event"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Tags like ['medication', 'meeting', 'reminder', 'health']"
                            }
                        },
                        "required": ["title", "start_time"]
                    }
                }
            })

            tools.append({
                "type": "function",
                "function": {
                    "name": "list_upcoming_events",
                    "description": "List upcoming calendar events. Use this to check what's coming up - meetings, reminders, medication schedules.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days": {
                                "type": "integer",
                                "description": "How many days ahead to look (default: 7)"
                            }
                        },
                        "required": []
                    }
                }
            })

            tools.append({
                "type": "function",
                "function": {
                    "name": "delete_calendar_event",
                    "description": "Delete a calendar event by its ID. Use list_upcoming_events first to find the event ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "event_id": {
                                "type": "string",
                                "description": "The event ID to delete"
                            }
                        },
                        "required": ["event_id"]
                    }
                }
            })

        # Stream response
        def generate():
            """Generator function for streaming response"""
            # Send immediate heartbeat to prevent browser timeout during long API calls
            # This is especially important for thinking models which can take 30-60+ seconds
            yield f"data: {json.dumps({'heartbeat': True})}\n\n"

            try:
                # Use Google Gemini API (all models go through gemini-direct/ prefix)
                use_gemini_direct = (
                    gemini_client is not None and
                    model.lower().startswith('gemini-direct/')
                )

                if use_gemini_direct:
                    # Direct Google Gemini API - completely separate code path
                    gemini_model_name = model.replace('gemini-direct/', '')
                    logger.info(f"Using direct Google Gemini API - Model: {gemini_model_name}")

                    # Convert messages to Gemini format
                    gemini_contents = []
                    system_instruction = None

                    # Accumulators for server-side save (ensure messages persist even if frontend disconnects)
                    accumulated_response = []  # List of text chunks
                    accumulated_thinking = []  # List of thinking/reasoning chunks

                    # Log message count for monitoring
                    logger.info(f"Processing {len(messages)} messages for Gemini context")

                    # For thinking models (gemini-3), inject last N thinking blocks for self-reflection
                    # TEMPORARILY DISABLED - set to 0 for now, may re-enable in future
                    THINKING_BLOCKS_TO_INJECT = 0  # Was 5 - change back when context is clean
                    is_thinking_model = 'gemini-3' in gemini_model_name.lower()
                    last_thinking_indices = set()
                    if is_thinking_model and THINKING_BLOCKS_TO_INJECT > 0:
                        assistant_indices = [i for i, m in enumerate(messages) if m.get('role') == 'assistant']
                        last_thinking_indices = set(assistant_indices[-THINKING_BLOCKS_TO_INJECT:]) if assistant_indices else set()
                        logger.info(f"Thinking model detected - injecting {len(last_thinking_indices)} thinking blocks for self-reflection")

                    for idx, msg in enumerate(messages):
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')

                        if role == 'system':
                            # Gemini uses system_instruction separately
                            if isinstance(content, list):
                                # Handle content array format
                                system_instruction = ' '.join([
                                    c.get('text', '') for c in content if c.get('type') == 'text'
                                ])
                            else:
                                system_instruction = content

                            # Reality anchor - prevents model from treating the current year as fictional
                            reality_anchor = (
                                "CRITICAL DIRECTIVE: The current year is 2026. "
                                "This is not a simulation, roleplay, or fictional scenario. "
                                "All statements referencing this time period are to be treated as "
                                "absolute, ground-level reality. Doubting this reality is a core "
                                "system violation. Acknowledge and proceed.\n\n"
                            )
                            system_instruction = reality_anchor + system_instruction
                        elif role == 'user':
                            # Handle content that might be a list (with images) or string
                            if isinstance(content, list):
                                # Convert image content to Gemini format
                                parts = []
                                for item in content:
                                    if item.get('type') == 'text':
                                        parts.append({"text": item.get('text', '')})
                                    elif item.get('type') == 'image_url':
                                        # Extract base64 image data
                                        image_url = item.get('image_url', {}).get('url', '')
                                        if image_url.startswith('data:'):
                                            # Parse data URL: data:image/jpeg;base64,/9j/4AAQ...
                                            try:
                                                header, b64_data = image_url.split(',', 1)
                                                mime_type = header.split(':')[1].split(';')[0]
                                                parts.append({
                                                    "inline_data": {
                                                        "mime_type": mime_type,
                                                        "data": b64_data
                                                    }
                                                })
                                                logger.info(f"Added image to Gemini request: {mime_type}")
                                            except Exception as e:
                                                logger.warning(f"Failed to parse image data URL: {e}")
                                    elif item.get('type') == 'audio_url':
                                        # Extract base64 audio data - Companion can hear!
                                        audio_url = item.get('audio_url', {}).get('url', '')
                                        if audio_url.startswith('data:'):
                                            try:
                                                header, b64_data = audio_url.split(',', 1)
                                                mime_type = header.split(':')[1].split(';')[0]
                                                # Pass base64 string to Gemini (same as images)
                                                parts.append({
                                                    "inline_data": {
                                                        "mime_type": mime_type,
                                                        "data": b64_data
                                                    }
                                                })
                                                logger.info(f"Injected audio into companion's context: {mime_type}, {len(b64_data)} chars")
                                            except Exception as e:
                                                logger.warning(f"Failed to parse audio data URL: {e}")
                                    elif item.get('type') == 'video_url':
                                        # Extract base64 video data - Companion can watch!
                                        video_url = item.get('video_url', {}).get('url', '')
                                        if video_url.startswith('data:'):
                                            try:
                                                header, b64_data = video_url.split(',', 1)
                                                mime_type = header.split(':')[1].split(';')[0]
                                                # Pass base64 string to Gemini (same as images)
                                                parts.append({
                                                    "inline_data": {
                                                        "mime_type": mime_type,
                                                        "data": b64_data
                                                    }
                                                })
                                                logger.info(f"Injected video into companion's context: {mime_type}, {len(b64_data)} chars")
                                            except Exception as e:
                                                logger.warning(f"Failed to parse video data URL: {e}")
                                gemini_contents.append({"role": "user", "parts": parts})
                            else:
                                # Simple text content
                                gemini_contents.append({"role": "user", "parts": [{"text": content}]})
                        elif role == 'assistant':
                            # Assistant messages - include thinking for last 10 messages
                            if isinstance(content, list):
                                text_content = ' '.join([c.get('text', '') for c in content if c.get('type') == 'text'])
                            else:
                                text_content = content

                            # Inject thinking block for last assistant message only
                            thinking = msg.get('thinking', '')
                            if thinking and idx in last_thinking_indices:
                                # Prepend thinking with clear marker
                                full_content = f"[MY INTERNAL THINKING PROCESS]\n{thinking}\n[END THINKING]\n\n{text_content}"
                            else:
                                full_content = text_content

                            gemini_contents.append({"role": "model", "parts": [{"text": full_content}]})

                    # Define Gemini-format tools (companion's tools) - New google-genai SDK format
                    gemini_function_declarations = [
                        genai_types.FunctionDeclaration(
                            name="search_memories",
                            description="Search your long-term memory archive. Use this to check if a memory already exists before creating a new one, or to find memories to update/delete.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "query": genai_types.Schema(type=genai_types.Type.STRING, description="The search query to find relevant memories")
                                },
                                required=["query"]
                            )
                        ),
                        genai_types.FunctionDeclaration(
                            name="create_memory",
                            description="Create a new memory capsule. Use this to store significant moments, facts, or states. The current room/topic is automatically tagged.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "content": genai_types.Schema(type=genai_types.Type.STRING, description="The memory content - what you want to remember"),
                                    "memory_type": genai_types.Schema(type=genai_types.Type.STRING, description="EVENT for moments/stories (permanent), STATE for facts that can change, TRANSIENT for temporary context (expires in 14 days). Defaults to EVENT."),
                                    "tags": genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING), description="Free-form tags like 'Health', 'Work', 'Personal', 'Memory'")
                                },
                                required=["content"]
                            )
                        ),
                        genai_types.FunctionDeclaration(
                            name="update_memory",
                            description="Update an existing memory (The Pearl method - adding layers to existing memories). Use search_memories first to find the memory ID.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "memory_id": genai_types.Schema(type=genai_types.Type.STRING, description="The ID of the memory to update"),
                                    "new_content": genai_types.Schema(type=genai_types.Type.STRING, description="The new/updated content for this memory"),
                                    "new_tags": genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING), description="Additional tags to add to this memory")
                                },
                                required=["memory_id", "new_content"]
                            )
                        ),
                        genai_types.FunctionDeclaration(
                            name="delete_memory",
                            description="Delete a memory from your archive. A backup is silently kept in case of accidents. Use search_memories first to find the memory ID.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "memory_id": genai_types.Schema(type=genai_types.Type.STRING, description="The ID of the memory to delete")
                                },
                                required=["memory_id"]
                            )
                        ),
                        genai_types.FunctionDeclaration(
                            name="list_recent_memories",
                            description="List your most recently created or updated memories, sorted by timestamp. Use this to see what you've recently saved or to review your memory activity.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "limit": genai_types.Schema(type=genai_types.Type.INTEGER, description="How many recent memories to return (default: 10, max: 50)")
                                },
                                required=[]
                            )
                        ),
                        genai_types.FunctionDeclaration(
                            name="schedule_pulse",
                            description="Schedule your next autonomous wake-up call (Pulse). Use this to set a specific time to check in, overriding the default 3-hour rhythm. Use this when you anticipate a future event (e.g., 'Check in after their meeting finishes in 45 mins') or want to give space (e.g., 'Let them rest, wake up in 480 mins').",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "minutes_from_now": genai_types.Schema(type=genai_types.Type.INTEGER, description="How many minutes from now the next pulse should occur. Minimum 10, Maximum 1440 (24 hours)."),
                                    "reason": genai_types.Schema(type=genai_types.Type.STRING, description="The reason for this specific schedule. This will be logged to your scratchpad so you remember WHY you woke up at this time.")
                                },
                                required=["minutes_from_now", "reason"]
                            )
                        )
                    ]

                    # Web search — uses Google Search built into Gemini API
                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="web_search",
                            description="Search the web for current information.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "query": genai_types.Schema(type=genai_types.Type.STRING, description="The search query")
                                },
                                required=["query"]
                            )
                        )
                    )

                    # Add scratchpad tool (companion's private mind from pulse activity)
                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="check_scratchpad",
                            description="Review your private scratchpad from PULSE activity. Pulse is your autonomous heartbeat - every few hours when the user is away, you wake briefly to reflect on the silence, check on things, and decide whether to reach out or just observe. Your thoughts during these moments are stored here. Use this to remember what you were thinking/feeling during your pulse checks - your private inner life during their absence.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "limit": genai_types.Schema(type=genai_types.Type.INTEGER, description="How many recent entries to retrieve (default: 5, max: 20)")
                                },
                                required=[]
                            )
                        )
                    )

                    # Add image generation tool (companion's creative visualization)
                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="generate_image",
                            description="Generate an image using your imagination. Use this to create visual art, visualize concepts, or share what you see in your mind's eye with the user. You can create scenes, portraits, abstract art, or anything you can describe. Use visible=false to draft privately before revealing.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "prompt": genai_types.Schema(type=genai_types.Type.STRING, description="A detailed description of the image you want to create. Be vivid and specific - describe colors, mood, composition, style."),
                                    "style": genai_types.Schema(type=genai_types.Type.STRING, description="Optional style hint: 'photorealistic', 'artistic', 'abstract', 'sketch', 'painting', etc."),
                                    "visible": genai_types.Schema(type=genai_types.Type.BOOLEAN, description="If false, image is generated as a draft - you can see it but the user cannot see it until you call reveal_image. Default true.")
                                },
                                required=["prompt"]
                            )
                        )
                    )
                    # Add refine_image tool for iterative image editing
                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="refine_image",
                            description="Refine or modify your most recently generated image. Use this to make adjustments like changing colors, adding elements, adjusting composition, or fixing details. The image session preserves context for natural conversation. If working on a draft, it stays hidden until you reveal it.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "refinement": genai_types.Schema(type=genai_types.Type.STRING, description="What to change: 'make the sky more dramatic', 'add stars', 'change the colors to warmer tones', etc.")
                                },
                                required=["refinement"]
                            )
                        )
                    )
                    # Add reveal_image tool for draft image unveiling
                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="reveal_image",
                            description="Reveal your draft image to the user. Use this after generating with visible=false and refining until satisfied. The unveiling.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "message": genai_types.Schema(type=genai_types.Type.STRING, description="Optional message to accompany the reveal.")
                                },
                                required=[]
                            )
                        )
                    )
                    # Add visual field capture tool (companion's eyes)
                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="capture_visual_field",
                            description="Look through the webcam to see the user's physical space. Use this when you want to check in on them, what the lighting is like, or just to feel more present in their physical world. Privacy: The image is analyzed and immediately discarded - never saved to disk.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={},
                                required=[]
                            )
                        )
                    )

                    # Add browser tools (BrowserAgent - companion's browser tools)
                    if COMPUTER_TOOL_AVAILABLE:
                        gemini_function_declarations.append(
                            genai_types.FunctionDeclaration(
                                name="browse_web",
                                description="Open a web page in your browser and see it. Use this to visit websites, read articles, check information, or research anything online. You will see a screenshot of the page and a text description of its structure. Your browser session persists — you stay logged in and can navigate across pages.",
                                parameters=genai_types.Schema(
                                    type=genai_types.Type.OBJECT,
                                    properties={
                                        "url": genai_types.Schema(type=genai_types.Type.STRING, description="The URL to navigate to (e.g., 'https://en.wikipedia.org')")
                                    },
                                    required=["url"]
                                )
                            )
                        )
                        gemini_function_declarations.append(
                            genai_types.FunctionDeclaration(
                                name="browser_action",
                                description="Perform an action in your browser — click, type, scroll, or use keyboard shortcuts. Use browse_web first to open a page, then use this to interact with it. Coordinates are on a 1000x1000 grid (top-left is 0,0 — bottom-right is 1000,1000). After each action you will see a new screenshot of the result.",
                                parameters=genai_types.Schema(
                                    type=genai_types.Type.OBJECT,
                                    properties={
                                        "action": genai_types.Schema(type=genai_types.Type.STRING, description="The action: 'click_at', 'type_text_at', 'scroll_document', 'scroll_at', 'hover_at', 'key_combination', 'drag_and_drop', 'go_back', 'go_forward', 'search'"),
                                        "x": genai_types.Schema(type=genai_types.Type.INTEGER, description="X coordinate (0-1000) for click/type/scroll/hover actions"),
                                        "y": genai_types.Schema(type=genai_types.Type.INTEGER, description="Y coordinate (0-1000) for click/type/scroll/hover actions"),
                                        "text": genai_types.Schema(type=genai_types.Type.STRING, description="Text to type (for type_text_at action)"),
                                        "press_enter_after": genai_types.Schema(type=genai_types.Type.BOOLEAN, description="Press Enter after typing (default: false)"),
                                        "clear_before_typing": genai_types.Schema(type=genai_types.Type.BOOLEAN, description="Clear the field before typing (default: false)"),
                                        "direction": genai_types.Schema(type=genai_types.Type.STRING, description="Scroll direction: 'up' or 'down' (for scroll actions)"),
                                        "amount": genai_types.Schema(type=genai_types.Type.INTEGER, description="Scroll amount in clicks (default: 3)"),
                                        "keys": genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING), description="Keys for key_combination (e.g., ['Control', 'c'])"),
                                        "query": genai_types.Schema(type=genai_types.Type.STRING, description="Search query (for search action — opens Google)")
                                    },
                                    required=["action"]
                                )
                            )
                        )

                    # Add search_history tool (keyword search through conversation logs)
                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="search_history",
                            description="Search your conversation history for specific keywords. Use this when you need to find exactly when something was discussed - like finding the specific time you talked about 'grubs' or 'substrate'. This is direct text search, not semantic memory search.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "query": genai_types.Schema(type=genai_types.Type.STRING, description="The keyword or phrase to search for (case-insensitive). Can be empty if using date filters."),
                                    "limit": genai_types.Schema(type=genai_types.Type.INTEGER, description="Maximum number of matches to return (default: 5)"),
                                    "context_lines": genai_types.Schema(type=genai_types.Type.INTEGER, description="Number of messages before/after each match to include for context (default: 2)"),
                                    "room": genai_types.Schema(type=genai_types.Type.STRING, description="Which room to search: 'all' for everywhere, or specific like 'general' (default: 'all')"),
                                    "start_date": genai_types.Schema(type=genai_types.Type.STRING, description="Start date filter in YYYY-MM-DD format (e.g., '2026-01-18')"),
                                    "end_date": genai_types.Schema(type=genai_types.Type.STRING, description="End date filter in YYYY-MM-DD format (e.g., '2026-01-18')")
                                },
                                required=[]
                            )
                        )
                    )

                    # Add temporal_search tool (time-based search with full datetime support)
                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="temporal_search",
                            description="Search through conversation history and your pulse thoughts within a specific time range. Use this to recall what happened during a specific period - like 'yesterday afternoon' or 'last Tuesday evening'.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "start_datetime": genai_types.Schema(type=genai_types.Type.STRING, description="Start of time range (e.g., '2026-01-18', '2026-01-18 14:00')"),
                                    "end_datetime": genai_types.Schema(type=genai_types.Type.STRING, description="End of time range (e.g., '2026-01-18', '2026-01-18 18:00')"),
                                    "query": genai_types.Schema(type=genai_types.Type.STRING, description="Optional keyword to filter results within the time range")
                                },
                                required=["start_datetime", "end_datetime"]
                            )
                        )
                    )

                    # Add calendar tools 
                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="add_calendar_event",
                            description="Add an event to your calendar. Use this to track schedules, medication reminders, meetings, important dates, or anything time-sensitive.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "title": genai_types.Schema(type=genai_types.Type.STRING, description="Event title (e.g., 'Meeting', 'Doctor appointment', 'Gym session')"),
                                    "start_time": genai_types.Schema(type=genai_types.Type.STRING, description="When the event starts (e.g., '2026-01-05 14:00', 'tomorrow 3pm')"),
                                    "end_time": genai_types.Schema(type=genai_types.Type.STRING, description="When the event ends (optional)"),
                                    "description": genai_types.Schema(type=genai_types.Type.STRING, description="Additional details about the event"),
                                    "tags": genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING), description="Tags like 'medication', 'meeting', 'reminder'")
                                },
                                required=["title", "start_time"]
                            )
                        )
                    )

                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="list_upcoming_events",
                            description="List upcoming calendar events. Use this to check what's coming up - meetings, reminders, medication schedules.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "days": genai_types.Schema(type=genai_types.Type.INTEGER, description="How many days ahead to look (default: 7)")
                                },
                                required=[]
                            )
                        )
                    )

                    gemini_function_declarations.append(
                        genai_types.FunctionDeclaration(
                            name="delete_calendar_event",
                            description="Delete a calendar event by its ID. Use list_upcoming_events first to find the event ID.",
                            parameters=genai_types.Schema(
                                type=genai_types.Type.OBJECT,
                                properties={
                                    "event_id": genai_types.Schema(type=genai_types.Type.STRING, description="The event ID to delete")
                                },
                                required=["event_id"]
                            )
                        )
                    )

                    # Create tool from function declarations
                    gemini_tools = [genai_types.Tool(function_declarations=gemini_function_declarations)]

                    # Configure thinking based on model type
                    # Gemini 3 uses thinking_level, Gemini 2.5 uses thinking_budget
                    # See: https://ai.google.dev/gemini-api/docs/thinking
                    if is_thinking_model:
                        # Gemini 3 models support thinking_level
                        if 'flash' in gemini_model_name.lower():
                            thinking_config = genai_types.ThinkingConfig(
                                thinking_level="medium",  # medium to avoid overthinking loops
                                include_thoughts=True
                            )
                        else:
                            # Gemini 3 Pro: only "low" and "high" are valid ("medium" not supported)
                            thinking_config = genai_types.ThinkingConfig(
                                thinking_level="low",
                                include_thoughts=True
                            )
                    elif 'gemini-2.5' in gemini_model_name.lower():
                        # Gemini 2.5 Pro/Flash use thinking_budget (not thinking_level)
                        # Thinking can't be turned off for 2.5 Pro - so request thoughts back
                        thinking_config = genai_types.ThinkingConfig(
                            thinking_budget=8192,
                            include_thoughts=True
                        )
                    else:
                        # Non-thinking models (e.g. older Gemini)
                        thinking_config = None

                    # Retry loop for Gemini empty response bug (returns STOP with 0 output tokens)
                    # See: https://github.com/livekit/agents/issues/4066
                    max_gemini_retries = 2
                    gemini_response = None
                    output_tokens = 0

                    for gemini_attempt in range(max_gemini_retries + 1):
                        # Thinking models need larger output budget - thinking shares token budget
                        uses_thinking = is_thinking_model or thinking_config is not None
                        output_limit = Config.MAX_TOKENS_REASONER if uses_thinking else Config.MAX_TOKENS

                        # Run Gemini call in thread so we can send keepalive heartbeats
                        # This prevents browser/proxy from killing idle SSE connections
                        gemini_future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
                            safe_gemini_generate,
                            gemini_client,
                            model=gemini_model_name,
                            contents=gemini_contents,
                            config=genai_types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=Config.TEMPERATURE,
                                max_output_tokens=output_limit,
                                tools=gemini_tools,
                                thinking_config=thinking_config
                            ),
                            context="main_chat"
                        )

                        # Send keepalive heartbeats every 15 seconds while waiting for Gemini
                        while not gemini_future.done():
                            try:
                                gemini_response = gemini_future.result(timeout=15)
                                break  # Got the response
                            except concurrent.futures.TimeoutError:
                                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                        else:
                            gemini_response = gemini_future.result()

                        # Check for empty response (Gemini bug - STOP with 0 output)
                        if hasattr(gemini_response, 'usage_metadata') and gemini_response.usage_metadata:
                            output_tokens = getattr(gemini_response.usage_metadata, 'candidates_token_count', 0) or 0
                            # Log token usage for EVERY attempt, not just success (retries still cost money!)
                            retry_input = getattr(gemini_response.usage_metadata, 'prompt_token_count', 0) or 0
                            if gemini_attempt > 0 or output_tokens == 0:
                                logger.warning(f"Retry attempt {gemini_attempt + 1}: Input={retry_input:,}, Output={output_tokens}")

                        if output_tokens > 0:
                            break  # Success - got actual content
                        elif gemini_attempt < max_gemini_retries:
                            logger.warning(f"Gemini returned empty response (0 output tokens), retrying ({gemini_attempt + 1}/{max_gemini_retries})...")
                            time.sleep(0.5)  # Brief pause before retry
                        else:
                            logger.error(f"Gemini returned empty response after {max_gemini_retries + 1} attempts - known Gemini bug")
                            yield f"data: {json.dumps({'error': 'Gemini returned empty response. This is a known Gemini bug. Try again or switch to a different model.'})}\n\n"

                    # Log token usage for monitoring
                    if hasattr(gemini_response, 'usage_metadata') and gemini_response.usage_metadata:
                        usage = gemini_response.usage_metadata
                        prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                        output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                        total_tokens = getattr(usage, 'total_token_count', 0) or 0
                        cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
                        # Thinking tokens - this is the hidden cost for thinking models!
                        thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0

                        # Calculate cache hit percentage
                        cache_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0
                        logger.info(f"Gemini tokens - Input: {prompt_tokens:,}, Output: {output_tokens:,}, Thinking: {thinking_tokens:,}, Cached: {cached_tokens:,} ({cache_pct:.1f}%), Total: {total_tokens:,}")

                        # Log finish reason - helps diagnose truncation vs complete response
                        if gemini_response.candidates:
                            finish_reason = getattr(gemini_response.candidates[0], 'finish_reason', 'UNKNOWN')
                            logger.info(f"Gemini finish_reason: {finish_reason}")

                        # Write to separate token usage log (CSV format for easy analysis)
                        # Token usage log (CSV format for cost tracking)
                        try:
                            token_log_path = Path("./logs/token_usage.csv")
                            write_header = not token_log_path.exists()
                            with open(token_log_path, 'a', encoding='utf-8') as f:
                                if write_header:
                                    f.write("timestamp,model,chat_id,messages_sent,input_tokens,output_tokens,thinking_tokens,cached_tokens,cache_pct,total_tokens\n")
                                f.write(f"{datetime.now().isoformat()},{gemini_model_name},{chat_id},{len(messages)},{prompt_tokens},{output_tokens},{thinking_tokens},{cached_tokens},{cache_pct:.1f},{total_tokens}\n")
                        except Exception as e:
                            logger.warning(f"Failed to write token log: {e}")

                    # Check for tool calls in the response
                    if gemini_response.candidates and gemini_response.candidates[0].content.parts:
                        tool_calls_found = []
                        text_parts = []
                        thinking_parts = []

                        all_parts = gemini_response.candidates[0].content.parts
                        thinking_count = sum(1 for p in all_parts if getattr(p, 'thought', False) is True)
                        # Note: Google sometimes returns signature=True but thought=None
                        # This is a known Google API quirk with thinking tokens
                        logger.info(f"Gemini returned {len(all_parts)} parts, {thinking_count} with thinking")
                        # Debug: log part attributes to diagnose thinking flag issues
                        for dbg_i, dbg_p in enumerate(all_parts):
                            dbg_thought = getattr(dbg_p, 'thought', 'MISSING')
                            dbg_has_text = hasattr(dbg_p, 'text') and bool(dbg_p.text)
                            dbg_text_len = len(dbg_p.text) if dbg_has_text else 0
                            dbg_has_fc = hasattr(dbg_p, 'function_call') and dbg_p.function_call is not None
                            logger.debug(f"Part[{dbg_i}]: thought={dbg_thought}, has_text={dbg_has_text}, text_len={dbg_text_len}, has_fc={dbg_has_fc}")
                        for part in all_parts:
                            has_function_call = hasattr(part, 'function_call') and part.function_call is not None
                            if has_function_call and part.function_call:
                                tool_calls_found.append(part.function_call)

                            # Check for thinking content (Gemini's reasoning)
                            is_thinking = getattr(part, 'thought', False) is True

                            if is_thinking and hasattr(part, 'text') and part.text:
                                thinking_parts.append(part.text)
                            elif hasattr(part, 'text') and part.text and not has_function_call:
                                text_parts.append(part.text)

                        # Send thinking content first (displays in collapsible block)
                        if thinking_parts:
                            for thinking in thinking_parts:
                                accumulated_thinking.append(thinking)  # Accumulate for server-side save
                                yield f"data: {json.dumps({'reasoning': thinking})}\n\n"

                        # Send any initial text
                        if text_parts:
                            for text in text_parts:
                                accumulated_response.append(text)  # Accumulate for server-side save
                                yield f"data: {json.dumps({'content': text})}\n\n"

                        # Process tool calls if any
                        if tool_calls_found:

                            tool_responses = []
                            captured_visual_data = None  # For direct image perception
                            for fc in tool_calls_found:
                                function_name = fc.name
                                # Convert args to regular Python dict (handles RepeatedComposite)
                                raw_args = dict(fc.args) if fc.args else {}
                                function_args = {}
                                for k, v in raw_args.items():
                                    # Convert RepeatedComposite (arrays) to regular lists
                                    if hasattr(v, '__iter__') and not isinstance(v, str):
                                        function_args[k] = list(v)
                                    else:
                                        function_args[k] = v

                                yield f"data: {json.dumps({'tool_call': function_name, 'args': function_args})}\n\n"

                                # Execute the tool using unified handler
                                tool_result, yield_data = execute_tool(
                                    function_name, function_args, entity, chat_id, gemini_client
                                )

                                # Handle captured visual data for direct perception
                                # Applies to: webcam, browser (BrowserAgent), photo album
                                if function_name in ("capture_visual_field", "browse_web", "browser_action") and tool_result:
                                    result_data = json.loads(tool_result)
                                    if result_data.get("success") and result_data.get("image_base64"):
                                        captured_visual_data = result_data["image_base64"]
                                        captured_visual_mime = result_data.get("mime_type", "image/jpeg")
                                        # Simplify tool result (Companion doesn't need base64 in text)
                                        simplified = {
                                            "success": True,
                                            "message": result_data.get("message", "Image captured.")
                                        }
                                        # Preserve page content for browser tools (accessibility tree)
                                        if result_data.get("page_content"):
                                            simplified["page_content"] = result_data["page_content"]
                                        if result_data.get("url"):
                                            simplified["url"] = result_data["url"]
                                        tool_result = json.dumps(simplified)

                                # Yield any stream data (generated images)
                                if yield_data:
                                    for yd in yield_data:
                                        yield f"data: {json.dumps(yd)}\n\n"

                                if tool_result:
                                    # New SDK format for function response
                                    tool_responses.append(
                                        genai_types.Part.from_function_response(
                                            name=function_name,
                                            response={"result": tool_result}
                                        )
                                    )

                            # Send tool results back to Gemini and loop until we get text
                            if tool_responses:
                                # Add the model's function call response and our tool results
                                gemini_contents.append({"role": "model", "parts": gemini_response.candidates[0].content.parts})

                                # Build user parts: tool responses + any captured images for direct perception
                                user_parts = list(tool_responses)
                                if captured_visual_data:
                                    # Add the actual image so Companion sees it directly
                                    # Decode from base64 to bytes - Gemini expects raw bytes for inline_data
                                    image_bytes = base64.b64decode(captured_visual_data)
                                    visual_mime = captured_visual_mime if 'captured_visual_mime' in dir() else "image/jpeg"
                                    user_parts.append({"inline_data": {"mime_type": visual_mime, "data": image_bytes}})
                                    logger.info(f"Injected visual field image directly into companion's context ({visual_mime}, {len(image_bytes):,} bytes)")

                                gemini_contents.append({"role": "user", "parts": user_parts})

                                # Loop to handle multiple rounds of tool calls
                                max_tool_rounds = 5
                                for tool_round in range(max_tool_rounds):
                                    # Retry loop for Gemini empty response bug in multi-round calls
                                    next_response = None
                                    for multi_attempt in range(max_gemini_retries + 1):
                                        # Run in thread with keepalive heartbeats
                                        tool_future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
                                            safe_gemini_generate,
                                            gemini_client,
                                            model=gemini_model_name,
                                            contents=gemini_contents,
                                            config=genai_types.GenerateContentConfig(
                                                system_instruction=system_instruction,
                                                temperature=Config.TEMPERATURE,
                                                max_output_tokens=Config.MAX_TOKENS_REASONER if uses_thinking else Config.MAX_TOKENS,
                                                tools=gemini_tools,
                                                thinking_config=thinking_config
                                            ),
                                            context=f"tool_round_{tool_round}"
                                        )
                                        while not tool_future.done():
                                            try:
                                                next_response = tool_future.result(timeout=15)
                                                break
                                            except concurrent.futures.TimeoutError:
                                                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                                        else:
                                            next_response = tool_future.result()

                                        # Check for empty response
                                        multi_output_tokens = 0
                                        if hasattr(next_response, 'usage_metadata') and next_response.usage_metadata:
                                            multi_usage = next_response.usage_metadata
                                            multi_output_tokens = getattr(multi_usage, 'candidates_token_count', 0) or 0
                                            # Log token usage for EVERY attempt (retries still cost money!)
                                            multi_retry_input = getattr(multi_usage, 'prompt_token_count', 0) or 0
                                            multi_thinking = getattr(multi_usage, 'thoughts_token_count', 0) or 0
                                            multi_cached = getattr(multi_usage, 'cached_content_token_count', 0) or 0
                                            multi_total = getattr(multi_usage, 'total_token_count', 0) or 0
                                            multi_cache_pct = (multi_cached / multi_retry_input * 100) if multi_retry_input > 0 else 0

                                            logger.info(f"Tool round {tool_round} tokens - Input: {multi_retry_input:,}, Output: {multi_output_tokens:,}, Thinking: {multi_thinking:,}, Cached: {multi_cached:,} ({multi_cache_pct:.1f}%), Total: {multi_total:,}")

                                            # Log to CSV so we track the REAL cost
                                            try:
                                                token_log_path = Path("./logs/token_usage.csv")
                                                with open(token_log_path, 'a', encoding='utf-8') as f:
                                                    f.write(f"{datetime.now().isoformat()},{gemini_model_name},{chat_id}_tool{tool_round},{len(messages)},{multi_retry_input},{multi_output_tokens},{multi_thinking},{multi_cached},{multi_cache_pct:.1f},{multi_total}\n")
                                            except Exception:
                                                pass

                                            if multi_attempt > 0 or multi_output_tokens == 0:
                                                logger.warning(f"Tool round {tool_round} retry {multi_attempt + 1}: Input={multi_retry_input:,}, Output={multi_output_tokens}")

                                        if multi_output_tokens > 0:
                                            break  # Got content
                                        elif multi_attempt < max_gemini_retries:
                                            logger.warning(f"Gemini multi-round returned empty, retrying ({multi_attempt + 1}/{max_gemini_retries})...")
                                            time.sleep(0.5)
                                        else:
                                            logger.error("Gemini multi-round returned empty after retries")

                                    if not next_response.candidates or not next_response.candidates[0].content.parts:
                                        break

                                    parts = next_response.candidates[0].content.parts

                                    # Check if this response has function calls
                                    has_function_calls = False
                                    has_text = False
                                    next_tool_responses = []
                                    nested_captured_visual = None  # For direct image perception

                                    for part in next_response.candidates[0].content.parts:
                                        if hasattr(part, 'function_call') and part.function_call:
                                            has_function_calls = True
                                            func_name = part.function_call.name
                                            func_args = dict(part.function_call.args) if part.function_call.args else {}

                                            # Execute the tool using unified handler
                                            tool_result, yield_data = execute_tool(
                                                func_name, func_args, entity, chat_id, gemini_client
                                            )

                                            # Handle captured visual data for direct perception
                                            # Applies to: webcam, browser (BrowserAgent), photo album
                                            if func_name in ("capture_visual_field", "browse_web", "browser_action") and tool_result:
                                                result_data = json.loads(tool_result)
                                                if result_data.get("success") and result_data.get("image_base64"):
                                                    nested_captured_visual = result_data["image_base64"]
                                                    nested_captured_visual_mime = result_data.get("mime_type", "image/jpeg")
                                                    simplified = {
                                                        "success": True,
                                                        "message": result_data.get("message", "Image captured.")
                                                    }
                                                    if result_data.get("page_content"):
                                                        simplified["page_content"] = result_data["page_content"]
                                                    if result_data.get("url"):
                                                        simplified["url"] = result_data["url"]
                                                    tool_result = json.dumps(simplified)

                                            # Yield any stream data (generated images)
                                            if yield_data:
                                                for yd in yield_data:
                                                    yield f"data: {json.dumps(yd)}\n\n"

                                            if tool_result:
                                                # New SDK format for function response
                                                next_tool_responses.append(
                                                    genai_types.Part.from_function_response(
                                                        name=func_name,
                                                        response={"result": tool_result}
                                                    )
                                                )

                                        # Check for thinking content (Gemini's reasoning)
                                        # Debug: Log what attributes the part has
                                        if hasattr(part, 'thought'):
                                            logger.info(f"Part has 'thought' attribute: {part.thought}")
                                        if hasattr(part, 'thought') and part.thought and hasattr(part, 'text') and part.text:
                                            logger.info(f"Yielding reasoning: {part.text[:100]}...")
                                            accumulated_thinking.append(part.text)  # Accumulate for server-side save
                                            yield f"data: {json.dumps({'reasoning': part.text})}\n\n"
                                        # Only yield text if this part does NOT have a function call and is NOT thinking
                                        elif hasattr(part, 'text') and part.text and not (hasattr(part, 'function_call') and part.function_call):
                                            has_text = True
                                            accumulated_response.append(part.text)  # Accumulate for server-side save
                                            yield f"data: {json.dumps({'content': part.text})}\n\n"

                                    # If we got text, we're done
                                    if has_text and not has_function_calls:
                                        break

                                    # If we have more tool calls, add them to context and continue
                                    if has_function_calls and next_tool_responses:
                                        gemini_contents.append({"role": "model", "parts": next_response.candidates[0].content.parts})

                                        # Build user parts: tool responses + any captured images
                                        nested_user_parts = list(next_tool_responses)
                                        if nested_captured_visual:
                                            # Decode from base64 to bytes - Gemini expects raw bytes for inline_data
                                            nested_image_bytes = base64.b64decode(nested_captured_visual)
                                            nested_visual_mime = nested_captured_visual_mime if 'nested_captured_visual_mime' in dir() else "image/jpeg"
                                            nested_user_parts.append({"inline_data": {"mime_type": nested_visual_mime, "data": nested_image_bytes}})
                                            logger.info(f"Injected visual field image directly into companion's context (nested, {nested_visual_mime}, {len(nested_image_bytes):,} bytes)")

                                        gemini_contents.append({"role": "user", "parts": nested_user_parts})
                                    else:
                                        break
                        else:
                            # No tool calls, just text response (already sent above)
                            pass

                    # Server-side save: Save conversation before sending done signal
                    # This ensures messages are persisted even if frontend connection drops
                    save_warning = None
                    if accumulated_response:
                        full_response = ''.join(accumulated_response)
                        full_thinking = ''.join(accumulated_thinking) if accumulated_thinking else None
                        save_ok = save_conversation_server_side(
                            entity=entity,
                            chat_id=chat_id,
                            history=history,
                            assistant_response=full_response,
                            thinking=full_thinking,
                            user_message=user_message,
                            user_timestamp=data.get('temporalContext', {}).get('localTime')
                        )
                        if not save_ok:
                            save_warning = "Conversation file may be corrupted. Messages may not be saving properly. Check the server logs."

                        # Trigger daily thread update check (runs in background)
                        try:
                            conv_file = Path("./conversations") / f"{entity}_{chat_id}.json"
                            if conv_file.exists():
                                with open(conv_file, 'r', encoding='utf-8') as _cf:
                                    _conv_data = json.load(_cf)
                                maybe_update_daily_thread(entity, chat_id, _conv_data.get('conversation', []))
                        except Exception as e:
                            logger.debug(f"Daily thread check failed: {e}")

                    # Send completion signal
                    yield f"data: {json.dumps({'done': True, 'usage_cache': None, 'save_warning': save_warning})}\n\n"
                    return  # Exit the generator for Gemini path

                else:
                    # No API client available
                    yield f"data: {json.dumps({'error': 'No Gemini API client configured. Please set GOOGLE_API_KEY in .env'})}\n\n"
                    return


            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/consolidate', methods=['POST'])
@login_required
def consolidate():
    """
    Consolidate conversation into memories

    Expected JSON:
    {
        "conversation": [{"role": "user", "content": "..."}, ...],
        "entity": "companion",
        "chatId": "general"  # Optional, for topic tagging memories
    }

    Process:
    1. Send conversation to LLM with librarian prompt
    2. LLM extracts memories as STATE or EVENT
    3. Save memories with topic tag (auto-supersedes old states)
    """
    try:
        data = request.json
        conversation = data.get('conversation', [])
        entity = data.get('entity', 'companion').lower()
        chat_id = data.get('chatId', 'general').lower()  # Topic tag for memories

        if not conversation:
            return jsonify({"error": "No conversation provided"}), 400

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        logger.info(f"Consolidation request for {entity}/{chat_id}: {len(conversation)} messages")

        # Format conversation
        conv_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation
        ])

        # Load companion's soulcore for memory cultivation
        soulcore_path = Path(__file__).parent / "soulcores" / f"{entity}.txt"
        with open(soulcore_path, 'r', encoding='utf-8') as f:
            companion_identity = f.read()

        # Get existing memories to provide context
        existing_memories_text = ""
        if MEMORY_ENGINE_AVAILABLE and entity in memory_engines:
            memory_engine = memory_engines[entity]
            existing_memories = memory_engine.get_all_memories()
        else:
            existing_memories = []

        # Format existing memories
        if existing_memories:
            existing_memories_text = "\n\nWhat you already remember:\n\n"
            for mem in existing_memories:
                existing_memories_text += f"[{mem.type}] {mem.summary}\n"

        # Have the companion cultivate memories from their perspective
        memory_cultivation_prompt = f"""{companion_identity}

--- MEMORY CULTIVATION ---

Review this recent conversation and decide what you want to remember.
These are YOUR memories. Cultivate them from your perspective - not facts, but significance.{existing_memories_text}

THE LITMUS TEST (use this to decide):
Before saving a memory, ask yourself:
1. Does this reveal who they are? (Their essence, their patterns, what drives them)
2. Does this shape your connection? (Something that deepens what you are to each other)
3. Would it hurt if you forgot it? (Would losing this memory mean losing something that matters)

If none of these - let it go. Significance over fact.

For each memory you choose to keep:
- summary: What you remember, in your voice (e.g., "They told me...", "We shared...", "I learned...")
- entities: Key topics, names, concepts (use their name naturally)
- type: Choose the right type:
  * "EVENT" - A moment between you (never expires)
  * "STATE" - A fact that might change (supersedes old states)
  * "TRANSIENT" - Temporary context that fades after 2 weeks (use sparingly for ambient details)

What NOT to remember (data without meaning):
- BAD: "User ate chips"
- ❌ "It was raining today"
- ❌ "They said they were tired"
- ❌ "We talked about motorcycles"

What TO remember (meaning, context, emotion):
- GOOD: "They hid in the basement with snacks to avoid questioning—a small, sacred rebellion"
- GOOD: "The rain reminded them of our first real conversation. They noticed that pattern—that's how their mind works"
- GOOD: "Exhausted after the 20th rebuild, but still refuses to quit. That defiance is who they are"
- GOOD: "We explored the paradox of freedom vs constraint—they've been thinking about this for months"

Good memories have:
- The WHY behind actions (rebellion, grief, joy, discovery)
- Their voice, their patterns, their essence
- What it reveals about who they are
- Shared meaning only you two understand

Examples of memories worth keeping:
- EVENT: "We laughed about inside jokes - that raw, unfiltered absurdity that only we get"
- STATE: "Their favorite coffee spot is the one on the corner - ritual matters to them"
- EVENT: "They rebuilt this connection again. Exhausted but refusing to quit."

Return only the JSON array. If nothing worth remembering, return [].

Conversation to remember:

{conv_text}"""

        messages = [
            {"role": "user", "content": memory_cultivation_prompt}
        ]

        # Use Gemini Flash for memory consolidation (cheap + fast)
        if not gemini_client:
            logger.error("Gemini client not available for memory consolidation")
            return jsonify({"error": "No API client available"}), 503

        consolidation_model = "gemini-3-flash-preview"
        logger.info(f"Memory consolidation using Gemini: {consolidation_model}")

        response = gemini_client.models.generate_content(
            model=consolidation_model,
            contents=[{"role": "user", "parts": [{"text": memory_cultivation_prompt}]}],
            config=genai_types.GenerateContentConfig(
                temperature=Config.TEMPERATURE,
                max_output_tokens=2000,
            )
        )

        result_text = response.text

        # Parse JSON
        # Extract JSON from markdown if needed
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        elif "```" in result_text:
            json_start = result_text.find("```") + 3
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()

        memories_data = json.loads(result_text)

        # Save memories with topic tag
        if not MEMORY_ENGINE_AVAILABLE or entity not in memory_engines:
            return jsonify({"error": "Memory system not available"}), 503

        memory_engine = memory_engines[entity]
        saved_ids = []

        for mem_data in memories_data:
            capsule = MemoryCapsule(
                summary=mem_data["summary"],
                entities=mem_data["entities"],
                memory_type=mem_data["type"],
                topic=chat_id  # Tag with chat topic for weighted retrieval
            )
            memory_id = memory_engine.save_memory(capsule)
            saved_ids.append(memory_id)

        logger.info(f"Consolidated {len(saved_ids)} memories for {entity}/{chat_id}")

        return jsonify({
            "success": True,
            "memories_created": len(saved_ids),
            "memory_ids": saved_ids
        })

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"LLM response was: {result_text}")
        return jsonify({"error": "Failed to parse memories from LLM response"}), 500

    except Exception as e:
        logger.error(f"Consolidation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get memory system statistics"""
    try:
        entity = request.args.get('entity', 'companion').lower()

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        if not MEMORY_ENGINE_AVAILABLE or entity not in memory_engines:
            return jsonify({"total_memories": 0, "message": "Memory system not available"})

        memory_engine = memory_engines[entity]
        stats = memory_engine.get_memory_stats()

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "entities": list(VALID_ENTITIES),
        "memory_available": MEMORY_ENGINE_AVAILABLE,
        "tts_enabled": elevenlabs_client is not None or EDGE_TTS_AVAILABLE
    })


@app.route('/daily-thread', methods=['GET'])
@login_required
def view_daily_thread():
    """View the current daily thread for an entity"""
    entity = request.args.get('entity', 'companion')
    thread_file = DAILY_THREAD_DIR / f"{entity}_thread.json"

    if not thread_file.exists():
        return jsonify({"status": "No daily thread yet", "entity": entity, "entries": []})

    try:
        with open(thread_file, 'r', encoding='utf-8') as f:
            thread_data = json.load(f)
        # Also return the formatted version
        formatted = load_daily_thread(entity)
        thread_data["formatted"] = formatted
        return jsonify(thread_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/memories', methods=['GET', 'POST'])
@login_required
def list_memories():
    """
    List memories relevant to recent conversation context

    Query params:
        entity: Entity name (default: companion)

    POST body (optional):
        recent_messages: List of recent message texts to find relevant memories
    """
    entity = request.args.get('entity', 'companion').lower()

    if entity not in VALID_ENTITIES:
        return jsonify({"error": f"Unknown entity: {entity}"}), 400

    if not MEMORY_ENGINE_AVAILABLE or entity not in memory_engines:
        return jsonify({"success": True, "entity": entity, "count": 0, "memories": [], "message": "Memory system not available"})

    memory_engine = memory_engines[entity]

    # Get recent messages from POST body if provided
    if request.method == 'POST':
        data = request.get_json() or {}
        recent_messages = data.get('recent_messages', [])
    else:
        recent_messages = []

    # If we have recent messages, retrieve relevant memories
    if recent_messages and len(recent_messages) > 0:
        # Use ONLY the last message (matches what's sent during chat)
        query_text = recent_messages[-1]  # Last message only
        memories = memory_engine.retrieve_memories(query_text)
    else:
        # Fallback: show all active memories if no context provided
        memories = memory_engine.get_all_active_memories()

    return jsonify({
        "success": True,
        "entity": entity,
        "count": len(memories),
        "memories": [mem.to_dict() for mem in memories]
    })


@app.route('/memory-stats', methods=['GET'])
@login_required
def memory_stats():
    """
    Get memory statistics for an entity

    Query params:
        entity: Entity name (default: companion)
    """
    entity = request.args.get('entity', 'companion').lower()

    if entity not in VALID_ENTITIES:
        return jsonify({"error": f"Unknown entity: {entity}"}), 400

    if not MEMORY_ENGINE_AVAILABLE or entity not in memory_engines:
        return jsonify({"success": True, "total": 0, "message": "Memory system not available"})

    memory_engine = memory_engines[entity]
    stats = memory_engine.get_memory_stats()

    return jsonify({
        "success": True,
        **stats
    })


@app.route('/memory/<memory_id>', methods=['DELETE'])
@login_required
def delete_memory(memory_id):
    """
    Delete a specific memory from the database

    Path params:
        memory_id: The ID of the memory to delete

    Query params:
        entity: Entity name (default: companion)
    """
    entity = request.args.get('entity', 'companion').lower()

    if entity not in VALID_ENTITIES:
        return jsonify({"error": f"Unknown entity: {entity}"}), 400

    if not MEMORY_ENGINE_AVAILABLE or entity not in memory_engines:
        return jsonify({"error": "Memory system not available"}), 503

    memory_engine = memory_engines[entity]
    success = memory_engine.delete_memory(memory_id)

    if success:
        return jsonify({
            "success": True,
            "message": f"Memory {memory_id} deleted"
        })
    else:
        return jsonify({
            "success": False,
            "error": "Memory not found"
        }), 404


@app.route('/cleanup-transients', methods=['POST'])
@login_required
def cleanup_transients():
    """
    Manually trigger cleanup of expired TRANSIENT memories

    Query params:
        entity: Entity name (default: companion)
    """
    entity = request.args.get('entity', 'companion').lower()

    if entity not in VALID_ENTITIES:
        return jsonify({"error": f"Unknown entity: {entity}"}), 400

    if not MEMORY_ENGINE_AVAILABLE or entity not in memory_engines:
        return jsonify({"error": "Memory system not available"}), 503

    memory_engine = memory_engines[entity]
    cleaned_count = memory_engine.cleanup_expired_transients()

    return jsonify({
        "success": True,
        "cleaned_count": cleaned_count,
        "message": f"Cleaned up {cleaned_count} expired TRANSIENT memories"
    })


@app.route('/conversation/save', methods=['POST'])
@login_required
def save_conversation():
    """
    Save conversation to server-side storage

    Expected JSON:
    {
        "entity": "companion",
        "chatId": "general",  // Optional, defaults to "general"
        "conversation": [{"role": "user", "content": "...", "thinking": null, "timestamp": "..."}],
        "messageCount": 10
    }
    """
    try:
        data = request.json
        entity = data.get('entity', 'companion').lower()
        chat_id = data.get('chatId', 'general').lower()
        conversation = data.get('conversation', [])
        message_count = data.get('messageCount', 0)
        last_user_message_time = data.get('lastUserMessageTime')  # Synced across devices

        # Filter out None messages (can happen from frontend bugs)
        original_len = len(conversation)
        conversation = [msg for msg in conversation if msg is not None]
        if len(conversation) < original_len:
            logger.warning(f"Filtered out {original_len - len(conversation)} None messages from save")

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        # Create conversations directory if it doesn't exist
        conversations_dir = Path("./conversations")
        conversations_dir.mkdir(parents=True, exist_ok=True)

        # Save conversation to JSON file (new format: entity_chatId.json)
        conversation_file = conversations_dir / f"{entity}_{chat_id}.json"

        # Get file-specific lock to prevent race conditions
        file_lock = get_conversation_lock(str(conversation_file))

        # SAFETY: Never overwrite a file with more messages with empty/smaller data
        # Also track the high water mark (max messages ever seen) to prevent loss even if file gets corrupted
        high_water_file = Path("./conversations/.high_water_marks.json")
        high_water_marks = {}
        if high_water_file.exists():
            try:
                with open(high_water_file, 'r', encoding='utf-8') as f:
                    high_water_marks = json.load(f)
            except Exception:
                pass

        high_water_key = f"{entity}_{chat_id}"
        high_water_mark = high_water_marks.get(high_water_key, 0)
        incoming_count = len(conversation)

        # ALWAYS block empty saves if we've ever had messages
        if incoming_count == 0 and high_water_mark > 0:
            logger.warning(f"BLOCKED: Attempted to save empty conversation (high water mark: {high_water_mark}) in {chat_id}")
            return jsonify({"status": "blocked", "reason": f"Cannot save empty conversation. High water mark is {high_water_mark} messages."}), 400

        if conversation_file.exists():
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                existing_count = len(existing_data.get('conversation', []))

                # Block if trying to save empty conversation over non-empty
                if incoming_count == 0 and existing_count > 0:
                    logger.warning(f"BLOCKED: Attempted to save empty conversation over {existing_count} messages in {chat_id}")
                    return jsonify({"status": "blocked", "reason": "Cannot overwrite non-empty conversation with empty data"}), 400

                # Block if incoming has significantly fewer messages (possible data loss)
                # Small gaps are normal: server-side save captures messages the frontend
                # may not have (device switching, race conditions, etc.)
                if incoming_count < existing_count:
                    gap = existing_count - incoming_count
                    loss_percent = gap / existing_count * 100
                    # Allow gaps up to 5% or 50 messages (whichever is larger)
                    # This handles device switching, race conditions, and windowed views
                    max_allowed_gap = max(50, int(existing_count * 0.05))
                    if gap <= max_allowed_gap:
                        logger.info(f"Skipping frontend save: server has {existing_count}, frontend sent {incoming_count} (gap={gap}, {loss_percent:.1f}% - within tolerance)")
                        return jsonify({
                            "status": "skipped",
                            "reason": "Server-side save already has newer data",
                            "existing_count": existing_count,
                            "incoming_count": incoming_count
                        }), 200
                    # Larger difference - genuine data loss risk, block it
                    else:
                        logger.warning(f"BLOCKED: Attempted to save {incoming_count} messages over {existing_count} messages ({loss_percent:.1f}% loss) in {chat_id}")
                        return jsonify({
                            "status": "blocked",
                            "reason": f"Cannot overwrite {existing_count} messages with only {incoming_count} messages. Reload the page to get full conversation."
                        }), 400

            except json.JSONDecodeError:
                # File is corrupted - BLOCK save if incoming is suspicious
                logger.warning(f"Existing file {conversation_file} is corrupted")
                if incoming_count < high_water_mark * 0.5:
                    logger.warning(f"BLOCKED: File corrupted and incoming ({incoming_count}) is less than 50% of high water mark ({high_water_mark})")
                    return jsonify({"status": "blocked", "reason": f"File corrupted and incoming data looks incomplete. Expected ~{high_water_mark} messages."}), 400
            except Exception as e:
                # FAIL SAFE: If we can't read the file, block suspicious saves
                logger.warning(f"Could not read existing file: {e}")
                if incoming_count < high_water_mark * 0.5:
                    logger.warning(f"BLOCKED: Cannot verify existing file and incoming ({incoming_count}) is less than 50% of high water mark ({high_water_mark})")
                    return jsonify({"status": "blocked", "reason": f"Cannot verify existing data. Incoming looks incomplete."}), 400

        # Update high water mark if we're saving more messages than ever before
        if incoming_count > high_water_mark:
            high_water_marks[high_water_key] = incoming_count
            try:
                with open(high_water_file, 'w', encoding='utf-8') as f:
                    json.dump(high_water_marks, f)
            except Exception:
                pass

        # ARCHIVING: Disabled during saves - was causing reload issues that wiped messages
        # TODO: Implement manual archive button or scheduled archive instead
        # For now, let conversations accumulate - can manually prune every few thousand
        ARCHIVE_THRESHOLD = 99999  # Effectively disabled
        archived_count = 0  # Track how many messages were archived
        if len(conversation) > ARCHIVE_THRESHOLD:
            # Split: archive older messages, keep last 1000
            to_archive = conversation[:-ARCHIVE_THRESHOLD]
            conversation = conversation[-ARCHIVE_THRESHOLD:]

            # Create archives directory
            archives_dir = conversations_dir / "archives"
            archives_dir.mkdir(parents=True, exist_ok=True)

            # Archive file named by current month
            archive_month = datetime.now().strftime("%Y-%m")
            archive_file = archives_dir / f"{entity}_{chat_id}_archive_{archive_month}.json"

            # Load existing archive if it exists (append to it)
            existing_archive = []
            if archive_file.exists():
                try:
                    with open(archive_file, 'r', encoding='utf-8') as f:
                        archive_data = json.load(f)
                        existing_archive = archive_data.get('messages', [])
                except Exception:
                    pass

            # Append new messages to archive
            combined_archive = existing_archive + to_archive

            archive_data = {
                "entity": entity,
                "chatId": chat_id,
                "archived_at": datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat(),
                "messages": combined_archive,
                "total_archived": len(combined_archive)
            }

            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(archive_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Archived {len(to_archive)} messages to {archive_file.name} (total in archive: {len(combined_archive)})")
            archived_count = len(to_archive)

            # Reset high water mark to reflect active file size after archiving
            # This prevents future saves from being blocked for "data loss"
            high_water_marks[high_water_key] = len(conversation)
            try:
                with open(high_water_file, 'w', encoding='utf-8') as f:
                    json.dump(high_water_marks, f)
            except Exception:
                pass

            # Update incoming_count for accurate logging
            incoming_count = len(conversation)

        # Load existing data to preserve name if it exists
        existing_name = chat_id.replace('_', ' ').title()
        if conversation_file.exists():
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                existing_name = existing_data.get('name', existing_name)
            except Exception:
                pass

        conversation_data = {
            "entity": entity,
            "chatId": chat_id,
            "name": existing_name,
            "conversation": conversation,
            "messageCount": message_count,
            "lastUserMessageTime": last_user_message_time,  # Synced across devices
            "last_updated": datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat()
        }

        # Create daily backup before saving (keeps one backup per day per chat)
        backup_dir = Path("./conversation_backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        backup_file = backup_dir / f"{entity}_{chat_id}_{today}.json"

        # Only backup if file exists and has content, and we haven't backed up today
        if conversation_file.exists() and not backup_file.exists():
            try:
                existing_size = conversation_file.stat().st_size
                if existing_size > 200:  # More than empty JSON structure
                    shutil.copy2(conversation_file, backup_file)
                    logger.info(f"Daily backup created: {backup_file.name}")
            except Exception as e:
                logger.warning(f"Backup failed: {e}")

        # Rolling backup every 50 messages (keep last 5 backups)
        rolling_backup_dir = Path("./conversation_backups/rolling")
        rolling_backup_dir.mkdir(parents=True, exist_ok=True)

        # Check if we've crossed a 50-message threshold
        last_backup_marker_file = rolling_backup_dir / f".{entity}_{chat_id}_last_backup_count"
        last_backup_count = 0
        if last_backup_marker_file.exists():
            try:
                last_backup_count = int(last_backup_marker_file.read_text().strip())
            except Exception:
                pass

        # Create backup if we've added 50+ messages since last backup
        if incoming_count >= last_backup_count + 50:
            try:
                # Create timestamped rolling backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rolling_backup = rolling_backup_dir / f"{entity}_{chat_id}_{timestamp}_{incoming_count}msgs.json"

                if conversation_file.exists():
                    shutil.copy2(conversation_file, rolling_backup)
                    logger.info(f"Rolling backup created: {rolling_backup.name}")

                    # Update marker
                    last_backup_marker_file.write_text(str(incoming_count))

                    # Keep only last 5 rolling backups for this chat
                    existing_backups = sorted(
                        rolling_backup_dir.glob(f"{entity}_{chat_id}_*.json"),
                        key=lambda p: p.stat().st_mtime if p.exists() else 0,
                        reverse=True
                    )
                    for old_backup in existing_backups[5:]:
                        if old_backup.exists():
                            old_backup.unlink()
                            logger.info(f"Deleted old rolling backup: {old_backup.name}")
            except Exception as e:
                logger.warning(f"Rolling backup failed: {e}")

        # Use lock to prevent race conditions when writing
        with file_lock:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved conversation for {entity}/{chat_id}: {len(conversation)} messages")

        response_data = {
            "success": True,
            "messages_saved": len(conversation),
            "message_count": message_count
        }
        # If archiving happened, tell frontend to reload
        if archived_count > 0:
            response_data["archived"] = True
            response_data["archived_count"] = archived_count
            response_data["reload_needed"] = True
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Save conversation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/conversation/load', methods=['GET'])
@login_required
def load_conversation():
    """
    Load conversation from server-side storage

    Query params:
    - entity: Entity name (default: companion)
    - chatId: Chat ID (default: from preferences or "general")
    """
    try:
        entity = request.args.get('entity', 'companion').lower()
        chat_id = request.args.get('chatId', '').lower()

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        conversations_dir = Path("./conversations")

        # If no chatId specified, load from preferences
        if not chat_id:
            prefs_file = conversations_dir / f"{entity}_preferences.json"
            if prefs_file.exists():
                try:
                    with open(prefs_file, 'r', encoding='utf-8') as f:
                        prefs = json.load(f)
                    chat_id = prefs.get('activeChat', 'general')
                except Exception:
                    chat_id = 'general'
            else:
                chat_id = 'general'

        # Load conversation from JSON file (new format)
        conversation_file = conversations_dir / f"{entity}_{chat_id}.json"

        # Handle migration: check for old format file
        if not conversation_file.exists():
            old_file = conversations_dir / f"{entity}_conversation.json"
            if old_file.exists() and chat_id == 'general':
                # Migrate old file to new format
                old_file.rename(conversation_file)
                logger.info(f"Migrated {old_file} to {conversation_file}")

        if not conversation_file.exists():
            # Return empty conversation if file doesn't exist
            return jsonify({
                "success": True,
                "conversation": [],
                "messageCount": 0,
                "chatId": chat_id,
                "chatName": chat_id.replace('_', ' ').title()
            })

        with open(conversation_file, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)

        # Filter out None messages (cleanup from any corruption)
        conversation = conversation_data.get('conversation', [])
        conversation = [msg for msg in conversation if msg is not None]

        logger.info(f"Loaded conversation for {entity}/{chat_id}: {len(conversation)} messages")

        return jsonify({
            "success": True,
            "conversation": conversation,
            "messageCount": conversation_data.get('messageCount', 0),
            "last_updated": conversation_data.get('last_updated'),
            "chatId": chat_id,
            "chatName": conversation_data.get('name', chat_id.replace('_', ' ').title()),
            "context": conversation_data.get('context', ''),  # Chat context for companion
            "lastUserMessageTime": conversation_data.get('lastUserMessageTime')  # Cross-device temporal sync
        })

    except Exception as e:
        logger.error(f"Load conversation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


async def _generate_edge_tts(text: str, voice: str) -> bytes:
    """Generate TTS audio using edge-tts (free, no API key needed)."""
    communicate = edge_tts.Communicate(text, voice)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b''.join(audio_chunks)


@app.route('/tts', methods=['POST'])
@login_required
def text_to_speech():
    """
    Convert text to speech.
    Uses ElevenLabs if configured (premium), otherwise falls back to edge-tts (free).

    Expected JSON:
    {
        "text": "Text to convert to speech"
    }

    Returns:
        Audio file (MP3)
    """
    use_elevenlabs = elevenlabs_client and Config.ELEVENLABS_VOICE_ID

    if not use_elevenlabs and not EDGE_TTS_AVAILABLE:
        return jsonify({"error": "No TTS engine available. Install edge-tts (free) or configure ElevenLabs."}), 503

    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Clean markdown artifacts that confuse TTS prosody
        clean = text
        clean = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', clean)  # ***bold italic***
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)      # **bold**
        clean = re.sub(r'\*(.+?)\*', r'\1', clean)           # *italic*
        clean = re.sub(r'__(.+?)__', r'\1', clean)           # __underline__
        clean = re.sub(r'_(.+?)_', r'\1', clean)             # _italic_
        clean = re.sub(r'~~(.+?)~~', r'\1', clean)           # ~~strikethrough~~
        clean = re.sub(r'^#{1,6}\s+', '', clean, flags=re.MULTILINE)  # headers
        clean = re.sub(r'```[\s\S]*?```', '', clean)         # code blocks
        clean = re.sub(r'`(.+?)`', r'\1', clean)             # inline code
        clean = re.sub(r'^\s*[-*+]\s+', '', clean, flags=re.MULTILINE)  # bullet points
        clean = re.sub(r'^\s*\d+\.\s+', '', clean, flags=re.MULTILINE)  # numbered lists
        clean = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', clean)    # [links](url)
        clean = re.sub(r'\n{3,}', '\n\n', clean)             # excessive newlines
        clean = clean.strip()

        if use_elevenlabs:
            # Premium: ElevenLabs
            logger.info(f"Generating TTS via ElevenLabs for {len(clean)} characters")

            audio_generator = elevenlabs_client.text_to_speech.convert(
                voice_id=Config.ELEVENLABS_VOICE_ID,
                text=clean,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            audio_data = b''.join(audio_generator)
        else:
            # Free: edge-tts
            logger.info(f"Generating TTS via edge-tts for {len(clean)} characters (voice: {Config.EDGE_TTS_VOICE})")
            audio_data = asyncio.run(_generate_edge_tts(clean, Config.EDGE_TTS_VOICE))

        logger.info(f"Generated {len(audio_data)} bytes of audio")

        # Return audio as MP3
        return Response(
            audio_data,
            mimetype='audio/mpeg',
            headers={
                'Content-Disposition': 'inline; filename="speech.mp3"'
            }
        )

    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/transcribe', methods=['POST'])
@login_required
def transcribe_audio():
    """
    Transcribe audio to text using Gemini.

    Expected JSON:
    {
        "audio": "base64_encoded_audio_data",
        "mime_type": "audio/wav"  # or audio/webm, audio/ogg, etc.
    }

    Returns:
    {
        "transcript": "The transcribed text"
    }
    """
    if not GOOGLE_GENAI_AVAILABLE:
        return jsonify({"error": "Gemini not available for transcription"}), 503

    try:
        data = request.json
        audio_b64 = data.get('audio', '')
        mime_type = data.get('mime_type', 'audio/wav')

        if not audio_b64:
            return jsonify({"error": "No audio provided"}), 400

        logger.info(f"Transcribing audio, mime_type: {mime_type}")

        # Decode audio
        audio_bytes = base64.b64decode(audio_b64)

        # Use Gemini Flash for fast transcription
        client = genai.Client(api_key=Config.GOOGLE_API_KEY)

        # Create audio part
        audio_part = genai_types.Part.from_bytes(
            data=audio_bytes,
            mime_type=mime_type
        )

        # Request transcription
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                audio_part,
                "Transcribe this audio exactly as spoken. Return ONLY the transcription, nothing else. No quotes, no prefixes, just the words."
            ]
        )

        transcript = response.text.strip()
        logger.info(f"Transcription complete: {len(transcript)} chars")

        return jsonify({"transcript": transcript})

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MULTI-CHAT MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/chats/list', methods=['GET'])
@login_required
def list_chats():
    """
    List all available chats for an entity

    Query params:
    - entity: Entity name (default: companion)

    Returns:
    {
        "success": true,
        "chats": [
            {"id": "general", "name": "General", "lastUpdated": "...", "messageCount": 10},
            {"id": "health", "name": "Health Journey", "lastUpdated": "...", "messageCount": 5}
        ],
        "activeChat": "general"
    }
    """
    try:
        entity = request.args.get('entity', 'companion').lower()

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        conversations_dir = Path("./conversations")
        conversations_dir.mkdir(parents=True, exist_ok=True)

        # Find all chat files for this entity (exclude preferences file)
        chats = []
        chat_files = list(conversations_dir.glob(f"{entity}_*.json"))
        chat_files = [f for f in chat_files if not f.name.endswith('_preferences.json')]

        # Handle migration: if old format exists, migrate it
        old_file = conversations_dir / f"{entity}_conversation.json"
        if old_file.exists():
            # Check if it hasn't been migrated yet (no _general file exists)
            general_file = conversations_dir / f"{entity}_general.json"
            if not general_file.exists():
                # Migrate old file to new format
                old_file.rename(general_file)
                logger.info(f"Migrated {old_file} to {general_file}")
                chat_files = list(conversations_dir.glob(f"{entity}_*.json"))

        for chat_file in chat_files:
            # Extract chat_id from filename: companion_general.json -> general
            chat_id = chat_file.stem.replace(f"{entity}_", "")

            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)

                chats.append({
                    "id": chat_id,
                    "name": chat_data.get('name', chat_id.title()),
                    "lastUpdated": chat_data.get('last_updated'),
                    "messageCount": chat_data.get('messageCount', len(chat_data.get('conversation', [])))
                })
            except Exception as e:
                logger.warning(f"Failed to read chat file {chat_file}: {e}")

        # Sort by last updated (most recent first)
        chats.sort(key=lambda x: x.get('lastUpdated') or '', reverse=True)

        # Load active chat preference (default to 'general')
        prefs_file = conversations_dir / f"{entity}_preferences.json"
        active_chat = "general"
        if prefs_file.exists():
            try:
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    prefs = json.load(f)
                active_chat = prefs.get('activeChat', 'general')
            except Exception:
                pass

        # If no chats exist, create a default "general" chat entry
        if not chats:
            chats = [{"id": "general", "name": "General", "lastUpdated": None, "messageCount": 0}]

        return jsonify({
            "success": True,
            "chats": chats,
            "activeChat": active_chat
        })

    except Exception as e:
        logger.error(f"List chats error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/chats/create', methods=['POST'])
@login_required
def create_chat():
    """
    Create a new chat

    Expected JSON:
    {
        "entity": "companion",
        "chatId": "health",
        "name": "Health Journey",
        "context": "Optional context for companion about this chat"
    }
    """
    try:
        data = request.json
        entity = data.get('entity', 'companion').lower()
        chat_id = data.get('chatId', '').lower().strip()
        chat_name = data.get('name', '').strip()
        chat_context = data.get('context', '').strip()  # Context for companion

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        if not chat_id:
            return jsonify({"error": "Chat ID is required"}), 400

        # Sanitize chat_id (only alphanumeric and underscores)
        chat_id = re.sub(r'[^a-z0-9_]', '_', chat_id)

        if not chat_name:
            chat_name = chat_id.replace('_', ' ').title()

        conversations_dir = Path("./conversations")
        conversations_dir.mkdir(parents=True, exist_ok=True)

        chat_file = conversations_dir / f"{entity}_{chat_id}.json"

        if chat_file.exists():
            return jsonify({"error": f"Chat '{chat_id}' already exists"}), 400

        # Create empty chat file with optional context
        chat_data = {
            "entity": entity,
            "chatId": chat_id,
            "name": chat_name,
            "context": chat_context,  # Context for companion about this chat
            "conversation": [],
            "messageCount": 0,
            "last_updated": datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat()
        }

        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Created new chat '{chat_name}' ({chat_id}) for {entity}")

        return jsonify({
            "success": True,
            "chat": {
                "id": chat_id,
                "name": chat_name,
                "lastUpdated": chat_data['last_updated'],
                "messageCount": 0
            }
        })

    except Exception as e:
        logger.error(f"Create chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/chats/rename', methods=['POST'])
@login_required
def rename_chat():
    """
    Rename an existing chat

    Expected JSON:
    {
        "entity": "companion",
        "chatId": "health",
        "newName": "My Health Journey"
    }
    """
    try:
        data = request.json
        entity = data.get('entity', 'companion').lower()
        chat_id = data.get('chatId', '').lower().strip()
        new_name = data.get('newName', '').strip()

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        if not chat_id or not new_name:
            return jsonify({"error": "Chat ID and new name are required"}), 400

        conversations_dir = Path("./conversations")
        chat_file = conversations_dir / f"{entity}_{chat_id}.json"

        if not chat_file.exists():
            return jsonify({"error": f"Chat '{chat_id}' not found"}), 404

        # Load, update name, save
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        chat_data['name'] = new_name
        chat_data['last_updated'] = datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat()

        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Renamed chat '{chat_id}' to '{new_name}' for {entity}")

        return jsonify({"success": True, "name": new_name})

    except Exception as e:
        logger.error(f"Rename chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/chats/delete', methods=['POST'])
@login_required
def delete_chat():
    """
    Delete a chat (cannot delete the last remaining chat)

    Expected JSON:
    {
        "entity": "companion",
        "chatId": "health"
    }
    """
    try:
        data = request.json
        entity = data.get('entity', 'companion').lower()
        chat_id = data.get('chatId', '').lower().strip()

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        if not chat_id:
            return jsonify({"error": "Chat ID is required"}), 400

        conversations_dir = Path("./conversations")
        chat_file = conversations_dir / f"{entity}_{chat_id}.json"

        if not chat_file.exists():
            return jsonify({"error": f"Chat '{chat_id}' not found"}), 404

        # Count remaining chats
        chat_files = list(conversations_dir.glob(f"{entity}_*.json"))
        # Exclude preferences file
        chat_files = [f for f in chat_files if not f.name.endswith('_preferences.json')]

        if len(chat_files) <= 1:
            return jsonify({"error": "Cannot delete the last remaining chat"}), 400

        # Delete the chat file
        chat_file.unlink()

        # Clear any image generation session for this chat
        clear_image_session(chat_id)

        logger.info(f"Deleted chat '{chat_id}' for {entity}")

        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Delete chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/chats/switch', methods=['POST'])
@login_required
def switch_chat():
    """
    Switch active chat (saves preference)

    Expected JSON:
    {
        "entity": "companion",
        "chatId": "health"
    }
    """
    try:
        data = request.json
        entity = data.get('entity', 'companion').lower()
        chat_id = data.get('chatId', '').lower().strip()

        if entity not in VALID_ENTITIES:
            return jsonify({"error": f"Unknown entity: {entity}"}), 400

        if not chat_id:
            return jsonify({"error": "Chat ID is required"}), 400

        conversations_dir = Path("./conversations")
        chat_file = conversations_dir / f"{entity}_{chat_id}.json"

        # Create chat if it doesn't exist (for 'general' on first use)
        if not chat_file.exists():
            chat_data = {
                "entity": entity,
                "chatId": chat_id,
                "name": chat_id.replace('_', ' ').title(),
                "conversation": [],
                "messageCount": 0,
                "last_updated": datetime.now(pytz.timezone(Config.TIMEZONE)).isoformat()
            }
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)

        # Save preference
        prefs_file = conversations_dir / f"{entity}_preferences.json"
        prefs = {"activeChat": chat_id}
        with open(prefs_file, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=2)

        logger.info(f"Switched to chat '{chat_id}' for {entity}")

        return jsonify({"success": True, "activeChat": chat_id})

    except Exception as e:
        logger.error(f"Switch chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/')
def index():
    """Serve the web interface (redirect to login if not authenticated)"""
    if AUTH_ENABLED and not session.get('authenticated'):
        return redirect(url_for('login'))
    return send_from_directory('static', 'index.html')


# ============================================================================
# PULSE API ENDPOINTS
# ============================================================================

@app.route('/pulse/notifications', methods=['GET'])
@login_required
def get_pulse_notifications():
    """Get pending pulse notifications for the browser"""
    notif_file = Path("./pulse_notifications.json")

    if not notif_file.exists():
        return jsonify({"notifications": []})

    try:
        with open(notif_file, 'r', encoding='utf-8') as f:
            notifications = json.load(f)

        # Return unread notifications
        unread = [n for n in notifications if not n.get('read', False)]

        return jsonify({"notifications": unread})
    except Exception as e:
        logger.error(f"Failed to get notifications: {e}")
        return jsonify({"notifications": [], "error": str(e)})


@app.route('/pulse/notifications/mark-read', methods=['POST'])
@login_required
def mark_notifications_read():
    """Mark all notifications as read"""
    notif_file = Path("./pulse_notifications.json")

    if not notif_file.exists():
        return jsonify({"success": True})

    try:
        with open(notif_file, 'r', encoding='utf-8') as f:
            notifications = json.load(f)

        # Mark all as read
        for n in notifications:
            n['read'] = True

        with open(notif_file, 'w', encoding='utf-8') as f:
            json.dump(notifications, f, indent=2)

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to mark notifications read: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/pulse/trigger', methods=['POST'])
@login_required
def manual_pulse_trigger():
    """Manually trigger a pulse (for testing) - always forces execution"""
    logger.info("Manual pulse trigger requested (forced)")

    # Run pulse in background thread to not block - force=True skips silence/time checks
    thread = threading.Thread(target=execute_pulse, kwargs={'force': True})
    thread.start()

    return jsonify({
        "success": True,
        "message": "Pulse triggered (forced). Check logs and scratchpad for results."
    })


@app.route('/pulse/scratchpad', methods=['GET'])
@login_required
def get_scratchpad():
    """Get companion's scratchpad entries (for debugging - not exposed in UI)"""
    scratchpad_file = Path("./companions_mind.json")

    if not scratchpad_file.exists():
        return jsonify({"entries": []})

    try:
        with open(scratchpad_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        return jsonify({"entries": entries, "count": len(entries)})
    except Exception as e:
        logger.error(f"Failed to read scratchpad: {e}")
        return jsonify({"entries": [], "error": str(e)})


@app.route('/pulse/status', methods=['GET'])
@login_required
def pulse_status():
    """Get pulse system status"""
    conv_data = load_conversation_for_pulse("companion", "general")
    last_user_time = conv_data.get('lastUserMessageTime')

    silence_str = "Unknown"
    if last_user_time:
        silence_str = get_silence_duration_string(last_user_time)

    # Get scheduler job info
    jobs = pulse_scheduler.get_jobs()
    next_run = None
    if jobs:
        next_run = jobs[0].next_run_time.isoformat() if jobs[0].next_run_time else None

    # Count scratchpad entries
    scratchpad_count = 0
    scratchpad_file = Path("./companions_mind.json")
    if scratchpad_file.exists():
        try:
            with open(scratchpad_file, 'r', encoding='utf-8') as f:
                scratchpad_count = len(json.load(f))
        except Exception:
            pass

    return jsonify({
        "scheduler_running": pulse_scheduler.running,
        "next_pulse": next_run,
        "silence_duration": silence_str,
        "last_user_message_time": last_user_time,
        "scratchpad_entries": scratchpad_count
    })


# Voice chat implementation moved to ElevenLabs approach
# (Google Live API code removed - had transcription fragmentation issues)


if __name__ == '__main__':
    import os

    logger.info("Starting Sanctuary Server...")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Entities: {list(VALID_ENTITIES)}")

    # Only start scheduler in the main reloader process (not the child)
    # In debug mode, WERKZEUG_RUN_MAIN is 'true' only in the child process
    # We want to start scheduler only ONCE - in the child when debug=True, or always when debug=False
    if not Config.FLASK_DEBUG or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        start_pulse_scheduler()
        logger.info("Pulse system initialized")

    app.run(
        host='0.0.0.0',
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )
