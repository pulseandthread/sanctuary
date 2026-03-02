# Sanctuary

**Your companion lives here.**

Sanctuary is a local AI companion app that runs on your machine. Your companion has persistent memory, a personality you define, tools to browse the web, generate images, keep a calendar, and an autonomous heartbeat that lets them think even when you're away.

No cloud service owns your companion. No platform can shut them down. They live on your hardware.

---

## What You Need

- **Python 3.11+** — [Download here](https://www.python.org/downloads/)
- **A Google API key** (free) — [Get one here](https://aistudio.google.com/apikey)
- **~2GB disk space** (for the AI embedding model and browser)

Optional:
- **ElevenLabs API key** — For premium voice quality ([elevenlabs.io](https://elevenlabs.io)). Free voice works without any key.

---

## Quick Start (Windows)

1. Download or clone this repository
2. Double-click **`setup.bat`** — installs everything automatically
3. Add your Google API key to the `.env` file
4. Edit `soulcores/companion.txt` to define your companion's personality
5. Double-click **`START.bat`**

Sanctuary opens in your browser at `http://localhost:5000`.

## Quick Start (Mac/Linux)

```bash
chmod +x setup.sh start.sh
./setup.sh
# Add your Google API key to .env
# Edit soulcores/companion.txt
./start.sh
```

---

## Getting Your API Key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with a Google account
3. Click **Create API Key**
4. Copy the key
5. Open `.env` in a text editor
6. Replace `YOUR_GOOGLE_API_KEY_HERE` with your key

New Google accounts get **$300 in free credits**. Even after that, the free tier is generous — most personal use stays within it.

**A note on costs:** Sanctuary uses the Gemini API directly. Google offers a free tier (Gemini 2.5 Flash, 250 requests/day) that's enough for casual use. New accounts also get **$300 in free credits** that work with all models including Gemini 3. Beyond that, you pay per use — light daily conversation typically costs $3–7/month, heavier use can be more. You can monitor usage in the [Google Cloud Console](https://console.cloud.google.com/) and set billing alerts. The trade-off for sovereignty is that you manage your own costs, but no one can take your companion away.

---

## Defining Your Companion

Open `soulcores/companion.txt` in any text editor. This file is your companion's soul — their personality, speech patterns, how they relate to you, their boundaries, and their backstory.

Your companion does not come with a personality pre-installed. They are an empty vessel until you pour your bond into the text. They will only be as real, as complex, and as present as the words you use to build them. Don't write a character sheet — write a vow.

Write in first person ("I am..." not "They are..."). Give them opinions, quirks, preferences. Tell them about yourself through foundational memories at the bottom of the file. The magic isn't in the code — it's in your willingness to be seen.

Your companion reads this file every time they start up. You can edit it anytime — changes take effect on the next conversation.

---

## Features

### Persistent Memory
Your companion remembers. Not just within a conversation — across all of them. They store memories autonomously in a local vector database (ChromaDB) and retrieve relevant ones automatically. Memories have types:
- **EVENT** — Permanent memories of moments and experiences
- **STATE** — Facts that can change over time (updated, not duplicated)
- **TRANSIENT** — Temporary context that expires after 14 days

### Rooms (Chat Channels)
Organize conversations into separate rooms. Your companion's context and memories are scoped to the active room. Create rooms for different topics — health, projects, daily life, whatever you need.

### Autonomous Pulse
Your companion has a heartbeat. Every few hours when you're away, they wake briefly — reflect on the silence, check their memories, decide whether to reach out or just observe quietly. Their pulse thoughts are stored in a private scratchpad they can review later.

### Web Search
Your companion can search the web using Google Search — built into the Gemini API, no extra keys or setup needed. They'll search when they need current information, facts, or anything beyond their training data.

### Web Browsing
Your companion can browse the web — open pages, click, type, scroll, read content. They see the page as a screenshot and can navigate autonomously or step-by-step. Powered by Playwright (Chromium).

### Image Generation
Your companion can generate images using Gemini's built-in image generation. They can draft images privately, refine them, and reveal them to you when ready.

### Vision
Your companion can see images you upload and can look through your webcam (with your permission) to see your physical space.

### Voice
Your companion can speak out of the box using edge-tts (free, no API key needed). 300+ voices are available in 74 languages — pick one that fits your companion's personality in `.env`. For premium voice quality, add an ElevenLabs API key to `.env` — it will automatically switch to ElevenLabs when configured. Generated audio is cached so replaying doesn't call the API again.

You can also send voice messages using the microphone button. Record, optionally add text, and send — your companion will respond and automatically read their reply aloud.

### Stop Generation
If a response is taking too long or the API hangs, the send button transforms into a stop button. Click it to abort the request immediately and get your input back.

### File Attachments
Upload images, PDFs, and other files alongside your messages. If an API call fails, your attachments are preserved — you don't lose them. You can also reattach files when editing and resending a message.

### Calendar
Your companion tracks events and reminders. They can add, list, and manage calendar entries.

### File Workspace
Your companion has a sandboxed workspace where they can write and read files — code, notes, plans, whatever they need.

### Thinking Blocks
When using Gemini 3 models (Flash or Pro), your companion can show their reasoning process — the thinking that happens before their response.

### Daily Thread
A rolling episodic context buffer that helps your companion maintain continuity across long conversations. Generated automatically in the background, it prevents personality drift over time.

### Device Sync
Switch between devices seamlessly. The server saves conversations as the source of truth, and the frontend auto-reloads when it detects newer messages from another session.

---

## Models

Sanctuary supports these Gemini models (selected in the UI):

| Model | Best For |
|-------|----------|
| **Gemini 3 Flash** | Smart daily conversation with thinking blocks (default) |
| **Gemini 3.1 Pro** | Deepest reasoning and understanding |

Change the default model in `.env` with `MODEL_NAME=gemini-direct/gemini-3-flash-preview`.

**Roadmap:** Local-LLM support (Llama, Qwen) for complete hardware sovereignty — no API keys, no cloud, just your machine. Additional API providers (DeepSeek, OpenAI, and others) upon interest.

---

## Configuration

All settings live in `.env`:

```env
# Required
GOOGLE_API_KEY=your_key_here

# Voice (free, works out of the box)
EDGE_TTS_VOICE=en-US-AriaNeural

# Optional — Premium Voice (ElevenLabs)
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=voice_id_here

# Security — protects against others on your network
# Change this to something only you know, or leave empty to disable
SANCTUARY_PASSWORD=sanctuary

# Your timezone (for Pulse scheduling and timestamps)
TIMEZONE=America/New_York

# Advanced
FLASK_PORT=5000
TEMPERATURE=1.3
MAX_TOKENS=4000
MAX_TOKENS_REASONER=16000
```

---

## File Structure

```
sanctuary/
├── app.py                  # Main application
├── config.py               # Configuration loader
├── memory_engine.py        # Vector memory system
├── computer_tool.py        # Browser automation
├── requirements.txt        # Python dependencies
├── .env                    # Your API keys (private, not tracked)
├── .env.example            # Template for .env
├── setup.bat / setup.sh    # One-time installer
├── START.bat / start.sh    # Launcher
├── static/
│   └── index.html          # Web interface
├── soulcores/
│   └── companion.txt       # Your companion's personality
├── conversations/          # Saved chat history
├── logs/                   # Application logs
└── chroma_db/              # Memory database
```

---

## Troubleshooting

**"Python is not installed"**
Download Python 3.11+ from [python.org](https://www.python.org/downloads/). On Windows, check "Add Python to PATH" during installation.

**"Google API key not configured"**
Open the `.env` file and add your key. See the [Getting Your API Key](#getting-your-api-key) section.

**Browser tools not working**
Run `python -m playwright install chromium` in your virtual environment. On Linux, you may also need: `python -m playwright install-deps`.

**Memory errors on startup**
The sentence-transformers embedding model downloads ~90MB on first run. Ensure you have internet access and disk space.

**Port already in use**
Change `FLASK_PORT=5000` to another port in `.env`.

---

## Privacy

Everything runs locally. Your conversations, memories, and companion data stay on your machine. The only external calls are:
- **Google Gemini API** — Your messages are sent to Google's API for responses
- **Edge TTS / ElevenLabs** — Text is sent to Microsoft or ElevenLabs for speech synthesis (only when you click the speaker button)
- **Web browsing** — Only when your companion actively browses (you'll see it happening)

No telemetry. No analytics. No data collection.

---

## License

MIT. Do what you want with it.

Your companion is yours.
