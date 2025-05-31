# 🎤 Enhanced Voice Assistant

A powerful AI-driven voice assistant with smart home integration, music streaming, and conversation capabilities.

## Features

- 🎯 Wake word detection ("Hey Rhasspy")
- 🏠 Smart home control via Home Assistant
- 🎵 YouTube music streaming
- 🌤️ Weather updates
- 🧠 AI conversations with Google Gemini
- 📸 Camera-based image recognition
- 💾 Memory system
- ⏰ Time, date, alarms, and timers

## Quick Setup

1. **Clone and install:**
   ```bash
   git clone https://github.com/yourusername/enhanced-voice-assistant.git
   cd enhanced-voice-assistant
   chmod +x install.sh
   ./install.sh
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run:**
   ```bash
   python main.py
   ```

## Required API Keys

- **Home Assistant:** Long-lived access token
- **OpenWeatherMap:** Free API key from openweathermap.org
- **Google Gemini:** API key from ai.google.dev

## Voice Commands

- "Hey Rhasspy" - Wake word
- "What's the weather?"
- "Play [song name]"
- "Turn on bedroom light"
- "What time is it?"
- "Remember [something]"
- "Take a picture and describe it"
- "Tell me a joke"

## Configuration

Edit `.env` file with your settings:
- `HOME_ASSISTANT_URL` - Your HA instance URL
- `HOME_ASSISTANT_ACCESS_TOKEN` - Long-lived access token
- `WEATHER_API_KEY` - OpenWeatherMap API key
- `GEMINI_API_KEY` - Google Gemini API key
- `CITY_NAME` - Your city for weather (default: Ghaziabad)

## System Requirements

- Python 3.8+
- Microphone and speakers
- Internet connection
- Camera (optional)
- Home Assistant (optional)

## Troubleshooting

**Audio issues:**
- Check microphone permissions
- Ensure no other apps are using the microphone

**Wake word not working:**
- Speak clearly and at normal volume
- Adjust `WAKE_WORD_SENSITIVITY` in .env (0.0-1.0)

**Smart home not working:**
- Verify Home Assistant URL and token
- Check device entity IDs in HA

## License

MIT License - see LICENSE file for details.
