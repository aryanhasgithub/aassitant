import threading
import time
import os
import uuid
import asyncio
import edge_tts
import pygame
import speech_recognition as sr
import requests
from homeassistant_api import Client
from datetime import datetime, timedelta
from threading import Timer, Event
import yt_dlp
from youtubesearchpython import VideosSearch
import pyjokes
import wikipedia
import subprocess
import webcolors
import pyaudio
import struct
import re
from google import genai
import PIL.Image
import cv2
import sqlite3
import pandas as pd
import numpy as np
from openwakeword.model import Model
from dotenv import load_dotenv
import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Any
import contextlib
from dataclasses import dataclass
from enum import Enum
import functools

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AssistantState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    MUSIC_PLAYING = "music_playing"

@dataclass
class DeviceEntity:
    entity_id: str
    domain: str
    friendly_name: str
    state: str

class EnhancedVoiceAssistant:
    def __init__(self):
        """Initialize the enhanced voice assistant with improved error handling and structure"""
        try:
            self._initialize_components()
            logger.info("Enhanced Voice Assistant initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {e}")
            raise

    def _initialize_components(self):
        """Initialize all components in proper order"""
        self._load_config()
        self._setup_audio()
        self._setup_clients()
        self._setup_database()
        self._setup_wake_word()
        self._setup_state_management()
        self._setup_thread_pool()

    def _load_config(self):
        """Load and validate configuration from environment variables"""
        required_vars = [
            "HOME_ASSISTANT_URL", "HOME_ASSISTANT_ACCESS_TOKEN",
            "WEATHER_API_KEY", "GEMINI_API_KEY"
        ]
        
        self.config = {}
        missing_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            self.config[var.lower()] = value
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Optional configuration with defaults
        self.config.update({
            'city_name': os.getenv("CITY_NAME", "Ghaziabad"),
            'tts_voice': os.getenv("TTS_VOICE", "en-US-AriaNeural"),
            'wake_word_sensitivity': float(os.getenv("WAKE_WORD_SENSITIVITY", "0.7")),
            'audio_quality': os.getenv("AUDIO_QUALITY", "128"),
            'max_command_duration': int(os.getenv("MAX_COMMAND_DURATION", "10")),
            'conversation_timeout': int(os.getenv("CONVERSATION_TIMEOUT", "30"))
        })

    def _setup_audio(self):
        """Initialize audio components with better error handling"""
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2)
            pygame.mixer.init()
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Audio stream parameters
            self.CHUNK_SIZE = 1280
            self.SAMPLE_RATE = 16000
            self.FORMAT = pyaudio.paInt16
            self.CHANNELS = 1
            
            # Adjust microphone for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                
            logger.info("Audio system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
            raise

    def _setup_clients(self):
        """Initialize API clients with retry logic"""
        try:
            # Home Assistant client
            self.ha_client = Client(
                self.config['home_assistant_url'],
                self.config['home_assistant_access_token']
            )
            
            # Gemini client
            self.gemini_client = genai.Client(api_key=self.config['gemini_api_key'])
            
            # Test connections
            self._test_connections()
            logger.info("API clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {e}")
            raise

    def _test_connections(self):
        """Test API connections"""
        try:
            # Test Home Assistant connection
            states = self.ha_client.get_states()
            logger.info(f"Connected to Home Assistant - {len(states)} entities found")
        except Exception as e:
            logger.warning(f"Home Assistant connection test failed: {e}")

    def _setup_database(self):
        """Initialize database with improved schema"""
        self.db_path = "assistant_data.db"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enhanced memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    importance INTEGER DEFAULT 1
                )
            """)
            
            # Command history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    success BOOLEAN DEFAULT TRUE,
                    execution_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")

    def _setup_wake_word(self):
        """Initialize wake word detection with OpenWakeWord"""
        try:
            self.wake_word_model = Model(inference_framework='tflite')
            self.wake_word_cooldown = 2.0
            self.last_wake_detection = 0
            logger.info("Wake word detection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize wake word detection: {e}")
            raise

    def _setup_state_management(self):
        """Initialize state management"""
        self.state = AssistantState.IDLE
        self.state_lock = threading.Lock()
        self.is_running = True
        self.current_task = None
        self.conversation_active = False
        self.conversation_timer = None

    def _setup_thread_pool(self):
        """Initialize thread pool for concurrent operations"""
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AssistantWorker")

    def set_state(self, new_state: AssistantState):
        """Thread-safe state management"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            logger.debug(f"State changed: {old_state.value} -> {new_state.value}")

    def get_state(self) -> AssistantState:
        """Get current state thread-safely"""
        with self.state_lock:
            return self.state

    async def speak_async(self, message: str, interrupt: bool = True):
        """Enhanced TTS with async support and better error handling"""
        if interrupt:
            self.interrupt_speech()
        
        self.set_state(AssistantState.SPEAKING)
        logger.info(f"Speaking: {message[:50]}...")
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Generate TTS audio
            communicate = edge_tts.Communicate(message, voice=self.config['tts_voice'])
            await communicate.save(temp_filename)
            
            # Play audio
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                if not self.is_running:
                    break
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            # Cleanup
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass
            
            if self.get_state() == AssistantState.SPEAKING:
                self.set_state(AssistantState.IDLE)

    def speak(self, message: str, interrupt: bool = True):
        """Synchronous wrapper for speak_async"""
        try:
            asyncio.run(self.speak_async(message, interrupt))
        except Exception as e:
            logger.error(f"Speak error: {e}")

    def interrupt_speech(self):
        """Stop any ongoing speech"""
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            logger.error(f"Error interrupting speech: {e}")

    def stream_youtube_audio(self, search_term: str) -> bool:
        """Stream YouTube audio without downloading to disk"""
        try:
            self.set_state(AssistantState.MUSIC_PLAYING)
            logger.info(f"Streaming music: {search_term}")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'mp3',
                'logtostderr': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(
                    f"ytsearch:{search_term}",
                    download=False
                )
                
                if not search_results.get('entries'):
                    logger.warning(f"No results found for: {search_term}")
                    return False
                
                url = search_results['entries'][0]['url']
                title = search_results['entries'][0].get('title', 'Unknown')
                
                # Stream directly
                pygame.mixer.music.load(url)
                pygame.mixer.music.play()
                
                self.speak(f"Now playing: {title}")
                
                # Start music control thread
                self.executor.submit(self._music_control_loop)
                return True
                
        except Exception as e:
            logger.error(f"Music streaming error: {e}")
            self.speak("Sorry, I couldn't play that music.")
            return False

    def _music_control_loop(self):
        """Handle music control commands while music is playing"""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            while pygame.mixer.music.get_busy() and self.get_state() == AssistantState.MUSIC_PLAYING:
                try:
                    with sr.Microphone() as source:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        command = self.recognizer.recognize_google(audio).lower()
                        logger.info(f"Music control command: {command}")
                        
                        if any(word in command for word in ["stop", "pause", "quiet"]):
                            pygame.mixer.music.stop()
                            self.speak("Music stopped")
                            break
                        elif "volume up" in command:
                            current_vol = pygame.mixer.music.get_volume()
                            pygame.mixer.music.set_volume(min(1.0, current_vol + 0.1))
                        elif "volume down" in command:
                            current_vol = pygame.mixer.music.get_volume()
                            pygame.mixer.music.set_volume(max(0.0, current_vol - 0.1))
                        elif "next" in command or "skip" in command:
                            pygame.mixer.music.stop()
                            self.speak("Skipping to next song")
                            break
                        else:
                            # Process other commands
                            self.executor.submit(self.process_command, command)
                            
                except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
                    continue
                    
        except Exception as e:
            logger.error(f"Music control error: {e}")
        finally:
            if self.get_state() == AssistantState.MUSIC_PLAYING:
                self.set_state(AssistantState.IDLE)

    def enhanced_command_processor(self, command: str):
        """Enhanced command processing with better pattern matching"""
        command = command.lower().strip()
        logger.info(f"Processing command: {command}")
        
        start_time = time.time()
        success = True
        
        try:
            # Smart home device control
            if self._handle_smart_home_commands(command):
                pass
            # Time and alarms
            elif self._handle_time_commands(command):
                pass
            # Entertainment
            elif self._handle_entertainment_commands(command):
                pass
            # Information queries
            elif self._handle_information_commands(command):
                pass
            # Memory management
            elif self._handle_memory_commands(command):
                pass
            # System control
            elif self._handle_system_commands(command):
                pass
            # Image recognition
            elif "recognise" in command or "recognize" in command:
                self._handle_image_recognition()
            # General AI conversation
            else:
                self._handle_general_conversation(command)
                
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            self.speak("Sorry, I encountered an error processing that command.")
            success = False
        finally:
            # Log command execution
            execution_time = time.time() - start_time
            self._log_command(command, success, execution_time)

    def _handle_smart_home_commands(self, command: str) -> bool:
        """Handle smart home device commands"""
        patterns = {
            r'turn (on|off) (light|fan|switch) (.+)': self._control_device,
            r'set (brightness|color|colour) of (.+) to (.+)': self._set_device_property,
            r'dim (.+) to (\d+)': self._set_brightness,
        }
        
        for pattern, handler in patterns.items():
            match = re.search(pattern, command)
            if match:
                return handler(match.groups())
        return False

    def _control_device(self, groups: tuple) -> bool:
        """Control smart home devices"""
        action, device_type, entity_name = groups
        entity_id = f"{device_type}.{entity_name.replace(' ', '_')}"
        
        try:
            domain = self.ha_client.get_domain(device_type)
            if action == "on":
                domain.turn_on(entity_id=entity_id)
                self.speak(f"Turning on {entity_name}")
            else:
                domain.turn_off(entity_id=entity_id)
                self.speak(f"Turning off {entity_name}")
            return True
        except Exception as e:
            logger.error(f"Device control error: {e}")
            self.speak(f"Sorry, I couldn't control {entity_name}")
            return False

    def _handle_time_commands(self, command: str) -> bool:
        """Handle time-related commands"""
        if "time" in command:
            current_time = datetime.now().strftime('%I:%M %p')
            self.speak(f"The current time is {current_time}")
            return True
        elif "date" in command:
            current_date = datetime.now().strftime('%A, %B %d, %Y')
            self.speak(f"Today is {current_date}")
            return True
        elif "set alarm" in command:
            return self._set_alarm(command)
        elif "set timer" in command:
            return self._set_timer(command)
        return False

    def _handle_entertainment_commands(self, command: str) -> bool:
        """Handle entertainment commands"""
        if "play" in command:
            music_query = command.replace("play", "").strip()
            return self.stream_youtube_audio(music_query)
        elif "stop" in command and self.get_state() == AssistantState.MUSIC_PLAYING:
            pygame.mixer.music.stop()
            self.speak("Music stopped")
            self.set_state(AssistantState.IDLE)
            return True
        elif "joke" in command:
            try:
                joke = pyjokes.get_joke()
                self.speak(joke)
                return True
            except Exception as e:
                logger.error(f"Joke error: {e}")
                self.speak("Sorry, I couldn't think of a joke right now.")
                return True
        return False

    def _handle_information_commands(self, command: str) -> bool:
        """Handle information queries"""
        if "weather" in command:
            return self._get_weather()
        elif "wikipedia" in command:
            return self._search_wikipedia(command)
        return False

    def _handle_memory_commands(self, command: str) -> bool:
        """Handle memory-related commands"""
        if "remember" in command:
            content = command.replace("remember", "").strip()
            return self._remember(content)
        elif "recall" in command or "what did I tell you" in command:
            return self._recall_memories()
        elif "forget" in command:
            content = command.replace("forget", "").strip()
            return self._forget(content)
        elif "clear memory" in command:
            return self._clear_all_memories()
        return False

    def _handle_system_commands(self, command: str) -> bool:
        """Handle system control commands"""
        if "shutdown" in command or "goodbye" in command:
            self.speak("Goodbye! Shutting down.")
            self.shutdown()
            return True
        elif "restart" in command:
            self.speak("Restarting...")
            self.restart()
            return True
        return False

    def _handle_image_recognition(self):
        """Handle image recognition with improved camera handling"""
        try:
            self.speak("Taking a photo for recognition")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                self.speak("Sorry, I couldn't access the camera")
                return
            
            # Take multiple frames to ensure good capture
            for _ in range(5):
                ret, frame = cap.read()
                if ret:
                    break
                time.sleep(0.1)
            
            if not ret:
                self.speak("Sorry, I couldn't capture an image")
                cap.release()
                return
                
            # Save and process image
            image_path = "recognition_image.jpg"
            cv2.imwrite(image_path, frame)
            cap.release()
            
            # AI recognition
            image = PIL.Image.open(image_path)
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=["Describe what you see in this image in detail. If you can identify specific objects, brands, or models, please mention them.", image]
            )
            
            description = response.text.replace("*", "")
            self.speak(f"I can see: {description}")
            
            # Cleanup
            if os.path.exists(image_path):
                os.remove(image_path)
                
        except Exception as e:
            logger.error(f"Image recognition error: {e}")
            self.speak("Sorry, I had trouble with image recognition.")

    def _handle_general_conversation(self, command: str):
        """Handle general conversation using AI"""
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Respond as a helpful voice assistant. Keep responses concise and conversational. User said: {command}"
            )
            response_text = response.text.replace("*", "")
            self.speak(response_text)
        except Exception as e:
            logger.error(f"AI conversation error: {e}")
            self.speak("Sorry, I'm having trouble understanding that right now.")

    def _get_weather(self) -> bool:
        """Get weather information"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={self.config['city_name']}&appid={self.config['weather_api_key']}&units=metric"
            response = requests.get(url, timeout=10)
            weather_data = response.json()
            
            if weather_data["cod"] == 200:
                temp = weather_data["main"]["temp"]
                desc = weather_data["weather"][0]["description"]
                humidity = weather_data["main"]["humidity"]
                feels_like = weather_data["main"]["feels_like"]
                
                weather_report = f"The temperature in {self.config['city_name']} is {temp}°C, feels like {feels_like}°C, with {desc}. Humidity is {humidity}%."
                self.speak(weather_report)
                return True
            else:
                self.speak("Sorry, I couldn't get the weather information.")
                return False
        except Exception as e:
            logger.error(f"Weather error: {e}")
            self.speak("Sorry, I'm having trouble getting weather information.")
            return False

    def _remember(self, content: str) -> bool:
        """Store information in memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO memories (content, category) VALUES (?, ?)",
                    (content, "user_request")
                )
                conn.commit()
            self.speak(f"I'll remember: {content}")
            return True
        except Exception as e:
            logger.error(f"Memory storage error: {e}")
            self.speak("Sorry, I had trouble remembering that.")
            return False

    def _recall_memories(self) -> bool:
        """Recall stored memories"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(
                    "SELECT content, created_at FROM memories ORDER BY created_at DESC LIMIT 10",
                    conn
                )
            
            if df.empty:
                self.speak("I don't have any memories stored.")
                return True
            
            memories_text = "Here's what I remember: " + "; ".join(df['content'].tolist())
            self.speak(memories_text)
            return True
        except Exception as e:
            logger.error(f"Memory recall error: {e}")
            self.speak("Sorry, I had trouble accessing my memories.")
            return False

    def _log_command(self, command: str, success: bool, execution_time: float):
        """Log command execution for analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO command_history (command, success, execution_time) VALUES (?, ?, ?)",
                    (command, success, execution_time)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Command logging error: {e}")

    def listen_continuously(self):
        """Main listening loop with improved wake word detection"""
        logger.info("Starting continuous listening...")
        
        try:
            # Initialize audio stream
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            logger.info("Listening for wake word 'Hey Rhasspy'...")
            
            while self.is_running:
                try:
                    # Read audio data
                    audio_data = np.frombuffer(
                        stream.read(self.CHUNK_SIZE, exception_on_overflow=False),
                        dtype=np.int16
                    )
                    
                    # Wake word detection
                    prediction = self.wake_word_model.predict(audio_data)
                    
                    if prediction.get("hey_rhasspy", 0) > self.config['wake_word_sensitivity']:
                        current_time = time.time()
                        
                        if current_time - self.last_wake_detection > self.wake_word_cooldown:
                            logger.info("Wake word detected!")
                            self.last_wake_detection = current_time
                            
                            # Play activation sound
                            self._play_activation_sound()
                            
                            # Process command
                            self.executor.submit(self._handle_voice_command)
                            
                            # Clear audio buffer
                            time.sleep(0.5)
                            while stream.get_read_available() > 0:
                                stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                                
                except Exception as e:
                    logger.error(f"Listening loop error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
        finally:
            try:
                stream.close()
                audio.terminate()
            except:
                pass

    def _play_activation_sound(self):
        """Play activation sound"""
        try:
            if os.path.exists('plop.mp3'):
                pygame.mixer.music.load('plop.mp3')
                pygame.mixer.music.play()
                time.sleep(0.5)  # Wait for sound to finish
        except Exception as e:
            logger.error(f"Activation sound error: {e}")

    def _handle_voice_command(self):
        """Handle voice command with timeout"""
        try:
            self.set_state(AssistantState.LISTENING)
            
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Listening for command...")
                
                audio = self.recognizer.listen(
                    source,
                    timeout=self.config['max_command_duration'],
                    phrase_time_limit=self.config['max_command_duration']
                )
            
            self.set_state(AssistantState.PROCESSING)
            command = self.recognizer.recognize_google(audio).lower()
            logger.info(f"Recognized command: {command}")
            
            self.enhanced_command_processor(command)
            
        except sr.UnknownValueError:
            logger.warning("Could not understand the audio")
            self.speak("Sorry, I didn't understand that.")
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            self.speak("Sorry, I'm having trouble with speech recognition.")
        except sr.WaitTimeoutError:
            logger.info("No command detected within timeout")
        except Exception as e:
            logger.error(f"Voice command error: {e}")
            self.speak("Sorry, I encountered an error.")
        finally:
            self.set_state(AssistantState.IDLE)

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down assistant...")
        self.is_running = False
        
        try:
            pygame.mixer.quit()
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    def run(self):
        """Main run method"""
        try:
            self.speak("Voice assistant is ready. Say 'Hey Rhasspy' to activate me.")
            self.listen_continuously()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.shutdown()

if __name__ == "__main__":
    try:
        assistant = EnhancedVoiceAssistant()
        assistant.run()
    except Exception as e:
        logger.error(f"Failed to start assistant: {e}")
        print(f"Error: {e}")
        print("Please check your configuration and try again.")
