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
            # Ensure genai is configured if you intend to use it.
            # For example, by setting GOOGLE_API_KEY environment variable
            # or genai.configure(api_key="YOUR_API_KEY")
            # self.gemini_client = genai.Client(api_key=self.config['gemini_api_key']) # This line might need adjustment based on GenAI library updates
            genai.configure(api_key=self.config['gemini_api_key']) # More common way to configure
            
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
            if self.ha_client.is_api_running():
                 states = self.ha_client.get_states()
                 logger.info(f"Connected to Home Assistant - {len(states)} entities found")
            else:
                 logger.warning("Home Assistant API is not running.")
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
        
        temp_filename = None # Initialize to avoid UnboundLocalError in finally
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
            if temp_filename and os.path.exists(temp_filename): # Check if temp_filename was assigned
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass # Ignore if removal fails (e.g., file still in use)
            
            if self.get_state() == AssistantState.SPEAKING:
                self.set_state(AssistantState.IDLE)

    def speak(self, message: str, interrupt: bool = True):
        """Synchronous wrapper for speak_async"""
        try:
            # Get or create an event loop for the current thread if not running in main thread
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(self.speak_async(message, interrupt))
        except RuntimeError as e: # Handle cases where loop might be running in a different thread
            if "cannot be called from a running event loop" in str(e):
                 # If already in an event loop, schedule it
                 asyncio.ensure_future(self.speak_async(message, interrupt))
            else:
                 logger.error(f"Speak error (RuntimeError): {e}")
                 # Fallback or re-raise if it's a different RuntimeError
                 # For simplicity, just logging here.
                 # Depending on context, you might want a synchronous fallback for TTS.
                 print(f"Fallback TTS (sync error): {message}") # Basic console fallback
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
                 # 'outtmpl': '%(title)s.%(ext)s', # Not needed for streaming URLs
                'noplaylist': True, # Process only the first video if a playlist is found by search
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Search and extract info for the first video
                info = ydl.extract_info(f"ytsearch:{search_term}", download=False)
                
                if not info.get('entries'):
                    logger.warning(f"No results found for: {search_term}")
                    self.speak(f"Sorry, I couldn't find any music for {search_term}.")
                    return False
                
                video_info = info['entries'][0]
                title = video_info.get('title', 'Unknown Title')
                
                # Find a suitable audio stream URL
                audio_url = None
                if video_info.get('url') and video_info.get('acodec') != 'none': # Direct URL if it's audio
                    audio_url = video_info.get('url')
                else:
                    for f in video_info.get('formats', []):
                        # Prefer opus or aac if available, otherwise any audio-only
                        if f.get('acodec') != 'none' and f.get('vcodec') == 'none':
                            if f.get('ext') in ['opus', 'm4a', 'mp3', 'ogg', 'webm']: # common audio formats
                                audio_url = f.get('url')
                                break # Take the first good one
                    if not audio_url and video_info.get('url'): # Fallback to main URL if no specific audio stream found
                         # This might be a video URL, pygame might handle it or not.
                         logger.warning("No specific audio-only stream found, trying main URL.")
                         audio_url = video_info.get('url')


                if not audio_url:
                    logger.warning(f"No streamable audio URL found for: {title}")
                    self.speak("Sorry, I couldn't find a streamable audio for that.")
                    return False
                
                # Stream directly
                pygame.mixer.music.load(audio_url)
                pygame.mixer.music.play()
                
                self.speak(f"Now playing: {title}")
                
                # Start music control thread
                self.executor.submit(self._music_control_loop)
                return True
                
        except Exception as e:
            logger.error(f"Music streaming error: {e}")
            self.speak("Sorry, I couldn't play that music.")
            return False
        finally:
            # Check if music stopped or finished, then reset state
            # This check should ideally be more robust, e.g., in the music_control_loop
            if self.get_state() == AssistantState.MUSIC_PLAYING and not pygame.mixer.music.get_busy():
                 self.set_state(AssistantState.IDLE)


    def _music_control_loop(self):
        """Handle music control commands while music is playing"""
        try:
            # No need to re-adjust for ambient noise here if already done globally
            # or if the microphone instance is shared and adjusted.
            # If using a new Microphone instance, then adjust:
            # with sr.Microphone() as source:
            #     self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            while pygame.mixer.music.get_busy() and self.get_state() == AssistantState.MUSIC_PLAYING:
                try:
                    with self.microphone as source: # Use the class's microphone instance
                        logger.debug("Music control loop: Listening for command...")
                        audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=4) # Increased timeout slightly
                        command = self.recognizer.recognize_google(audio).lower()
                        logger.info(f"Music control command received: {command}")
                        
                        if any(word in command for word in ["stop music", "pause music", "quiet", "silence"]):
                            pygame.mixer.music.stop()
                            self.speak("Music stopped")
                            self.set_state(AssistantState.IDLE) # Explicitly set IDLE
                            break # Exit loop
                        elif "volume up" in command:
                            current_vol = pygame.mixer.music.get_volume()
                            new_vol = min(1.0, current_vol + 0.2) # Increase by 20%
                            pygame.mixer.music.set_volume(new_vol)
                            self.speak(f"Volume set to {int(new_vol*100)}%")
                        elif "volume down" in command:
                            current_vol = pygame.mixer.music.get_volume()
                            new_vol = max(0.0, current_vol - 0.2) # Decrease by 20%
                            pygame.mixer.music.set_volume(new_vol)
                            self.speak(f"Volume set to {int(new_vol*100)}%")
                        elif "next song" in command or "skip song" in command:
                            pygame.mixer.music.stop()
                            self.speak("Skipping to the next song.") # This implies a playlist feature not fully implemented here.
                                                                    # For now, it just stops the current song.
                            # To implement "next", you'd need to re-trigger stream_youtube_audio with a new search or next item in a queue.
                            self.set_state(AssistantState.IDLE)
                            break
                        # Add more controls like "resume music" if you implement pause separately
                        # else:
                            # If it's not a music control command, it might be a general command.
                            # Decide if you want to process general commands while music is playing.
                            # For now, we ignore non-music commands in this loop.
                            # logger.debug(f"Non-music command during playback: {command}")
                            
                except sr.WaitTimeoutError:
                    logger.debug("Music control: No command heard (timeout).")
                    continue # No speech detected, continue loop
                except (sr.UnknownValueError, sr.RequestError) as e:
                    logger.warning(f"Music control: Speech recognition error: {e}")
                    continue # Ignore if speech is unintelligible or API error
                except Exception as e: # Catch any other unexpected errors in listen/recognize
                    logger.error(f"Unexpected error in music control listen/recognize: {e}")
                    time.sleep(1) # Brief pause before retrying
                    
            logger.info("Music control loop finished or music stopped.")
        except Exception as e: # Catch errors in the outer setup of the loop (e.g., microphone access)
            logger.error(f"Music control loop setup error: {e}")
        finally:
            # Ensure state is IDLE if music is no longer playing
            if self.get_state() == AssistantState.MUSIC_PLAYING and not pygame.mixer.music.get_busy():
                self.set_state(AssistantState.IDLE)
            elif self.get_state() == AssistantState.MUSIC_PLAYING and not self.is_running: # If assistant shutting down
                pygame.mixer.music.stop()
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
            elif "recognise image" in command or "recognize image" in command or "what is this image" in command:
                self._handle_image_recognition() 
            # General AI conversation
            else:
                self._handle_general_conversation(command) 
                
        except Exception as e:
            logger.error(f"Command processing error: {e}", exc_info=True) # Add exc_info for traceback
            self.speak("Sorry, I encountered an error processing that command.")
            success = False
        finally:
            # Log command execution
            execution_time = time.time() - start_time
            self._log_command(command, success, execution_time) 
            if self.get_state() == AssistantState.PROCESSING: # Reset state if still processing
                self.set_state(AssistantState.IDLE)


    def _handle_smart_home_commands(self, command: str) -> bool:
        """Handle smart home device commands"""
        patterns = {
            # Order matters: more specific patterns first
            r'set (brightness|dim|dimmer) of (.+?) to (\d+)%?': self._set_device_property, # e.g., "set brightness of living room light to 50%"
            r'set (color|colour) of (.+?) to (.+)': self._set_device_property, # e.g., "set color of bedroom lamp to blue"
            r'turn (on|off) (the\s*)?(.+)': self._control_device, # e.g., "turn on the kitchen fan" or "turn off living room light"
        }
        
        for pattern, handler in patterns.items():
            match = re.search(pattern, command, re.IGNORECASE) # Add IGNORECASE
            if match:
                groups = match.groups()
                # Dispatch to handler. The handler needs to be aware of the group structure for its pattern.
                if handler == self._control_device:
                    # Expected: (action, "the " or None, entity_name_phrase)
                    action = groups[0]
                    entity_name_phrase = groups[2]
                    # Infer device type or make it more generic in _control_device
                    # For now, _control_device tries to infer from common names or requires HA entity_id structure
                    return self._control_device((action, entity_name_phrase)) # Pass (action, entity_name)
                elif handler == self._set_device_property:
                    # Pattern 1: (property_type, device_name, value) -> e.g. ("brightness", "living room light", "50")
                    # Pattern 2: (property_type, device_name, value) -> e.g. ("color", "bedroom lamp", "blue")
                    return self._set_device_property(groups)
                # Note: _set_brightness is now part of _set_device_property logic
                # If you had a separate _set_brightness, you'd call it here.
                # return handler(groups) # Generic call if groups match handler's expectation
        return False

    def _control_device(self, groups: tuple) -> bool:
        """Control smart home devices using HA API"""
        if not self.ha_client or not self.ha_client.is_api_running(): # Check if API is running
            self.speak("Home Assistant is not connected or not running.")
            return False
        
        try:
            # Groups for 'turn (on|off) (the\s*)?(.+)' will be (action, optional_the, entity_name_phrase)
            action = groups[0]
            entity_name_phrase = groups[1] # This was 'groups[2]' before, adjusted for the new pattern structure.

            # Attempt to find a matching entity_id. This is tricky with natural language.
            # We might need a more sophisticated lookup or rely on HA's fuzzy matching if its API supports it.
            # For now, let's try to guess common domains (light, switch, fan)
            
            potential_domains = ["light", "switch", "fan", "media_player"]
            entity_id_to_control = None
            
            # First, check if entity_name_phrase itself is a valid entity_id (e.g., "light.living_room")
            if '.' in entity_name_phrase and self.ha_client.get_entity(entity_id=entity_name_phrase.replace(' ', '_')):
                entity_id_to_control = entity_name_phrase.replace(' ', '_')
            else:
                # Try to find by friendly name across common domains
                cleaned_name = entity_name_phrase.lower().replace("the ", "").strip()
                all_states = self.ha_client.get_states()
                matched_entity = None
                for state in all_states:
                    friendly_name = state.attributes.get('friendly_name', '').lower()
                    if cleaned_name in friendly_name or cleaned_name == friendly_name:
                         # Prioritize light, switch, fan if multiple matches
                        if state.domain in potential_domains:
                            matched_entity = state
                            if state.domain == "light": break # Prefer light if "light" is in name
                
                if matched_entity:
                    entity_id_to_control = matched_entity.entity_id
                else: # Fallback: construct a plausible ID and try it
                    # This is a guess and might not work well.
                    # A better approach would be to query HA for entities matching the name.
                    for domain_guess in potential_domains:
                        guessed_entity_id = f"{domain_guess}.{entity_name_phrase.replace(' ', '_').lower()}"
                        if self.ha_client.get_entity(entity_id=guessed_entity_id):
                            entity_id_to_control = guessed_entity_id
                            break
            
            if not entity_id_to_control:
                self.speak(f"Could not find a controllable device named {entity_name_phrase}.")
                return False

            # Get the actual domain from the resolved entity_id
            domain = entity_id_to_control.split('.')[0]
            service_to_call = f"turn_{action}" # action is "on" or "off"

            logger.info(f"Attempting to call service {domain}.{service_to_call} for {entity_id_to_control}")
            
            # Call the appropriate service
            # The homeassistant_api Client's trigger_service might be what we need,
            # or it might have direct turn_on/turn_off methods on domain objects.
            # Let's assume trigger_service is generic.
            # result = self.ha_client.trigger_service( # Old direct call
            #     domain=domain,
            #     service=service_to_call,
            #     entity_id=entity_id_to_control # Pass as kwarg if expected
            # )
            
            # Using domain objects if available and preferred by homeassistant_api lib
            domain_obj = self.ha_client.get_domain(domain)
            if action == "on":
                domain_obj.turn_on(entity_id=entity_id_to_control)
            else: # off
                domain_obj.turn_off(entity_id=entity_id_to_control)
            
            # Assuming success if no exception. Some APIs might return a status.
            self.speak(f"Turning {action} {entity_name_phrase} (which is {entity_id_to_control}).")
            return True
                        
        except Exception as e:
            logger.error(f"Device control error for '{groups[1]}': {e}", exc_info=True)
            self.speak(f"Sorry, I couldn't control {groups[1]}. There might be an issue with the device name or Home Assistant connection.")
            return False

    def _set_device_property(self, groups: tuple) -> bool:
        """Set device properties like brightness, color, etc."""
        if not self.ha_client or not self.ha_client.is_api_running():
            self.speak("Home Assistant is not connected or not running.")
            return False
        
        try:
            # Groups: (property_type, device_name_phrase, value)
            # e.g., ("brightness", "living room light", "50") OR ("color", "bedroom lamp", "blue")
            property_type = groups[0].lower()
            device_name_phrase = groups[1].lower().replace("the ", "").strip()
            value = groups[2].lower()

            # Find the light entity
            # This is a simplified lookup. A real system might need more robust entity resolution.
            entity_id = None
            all_lights = self.ha_client.get_states(domain_filter="light")
            for light_entity in all_lights:
                if device_name_phrase in light_entity.attributes.get('friendly_name', '').lower() or \
                   device_name_phrase == light_entity.attributes.get('friendly_name', '').lower() or \
                   device_name_phrase.replace(" ", "_") == light_entity.entity_id.split('.')[1]:
                    entity_id = light_entity.entity_id
                    break
            
            if not entity_id:
                 # Fallback: try constructing it, assuming it's a light
                entity_id = f"light.{device_name_phrase.replace(' ', '_')}"
                if not self.ha_client.get_entity(entity_id=entity_id): # Check if this constructed ID exists
                    self.speak(f"Could not find light named {device_name_phrase}.")
                    return False


            service_data = {"entity_id": entity_id}
            speak_property = property_type

            if property_type in ['brightness', 'dim', 'dimmer']:
                speak_property = "brightness"
                try:
                    brightness_pct = int(value.replace('%',''))
                    if not (0 <= brightness_pct <= 100):
                        self.speak("Brightness should be between 0 and 100 percent.")
                        return False
                    # HA uses 0-255 for brightness usually, but brightness_pct for service calls
                    service_data["brightness_pct"] = brightness_pct
                except ValueError:
                    self.speak("Invalid brightness value. Please use a number between 0 and 100.")
                    return False
            
            elif property_type in ['color', 'colour']:
                # HA service `light.turn_on` can take `color_name` or `hs_color`, `rgb_color` etc.
                # `color_name` is the easiest for voice.
                service_data["color_name"] = value 
                # You might want to validate color name if possible, or let HA handle it.
            
            else:
                self.speak(f"I don't know how to set {property_type}.")
                return False

            logger.info(f"Calling light.turn_on for {entity_id} with data: {service_data}")
            # Call the light.turn_on service with parameters
            # Most property changes for lights are done via the turn_on service.
            light_domain = self.ha_client.get_domain("light")
            light_domain.turn_on(**service_data) # Pass service_data as keyword arguments

            self.speak(f"Set {speak_property} of {device_name_phrase} to {value}.")
            return True
                        
        except Exception as e:
            logger.error(f"Error setting device property for '{groups[1]}': {e}", exc_info=True)
            self.speak(f"Sorry, I couldn't set the {groups[0]} for {groups[1]}.")
            return False

    def _set_brightness(self, groups: tuple) -> bool:
        """Set device brightness (Helper, now integrated into _set_device_property but kept for direct pattern if needed)"""
        # This method is largely redundant if _set_device_property handles brightness.
        # However, if you have a very specific regex pattern that only extracts (device_name, brightness_pct)
        # and you want to call this directly, it can stay.
        # For the current setup, _set_device_property is more general.
        
        # Simplified call to the more general function:
        property_type = "brightness"
        device_name, brightness_val_str = groups # Expects (device_name, brightness_percentage_string)
        return self._set_device_property((property_type, device_name, brightness_val_str))


    def _handle_time_commands(self, command: str) -> bool:
        """Handle time-related commands"""
        if "what time is it" in command or "current time" in command or ("time" in command and not "timer" in command and not "alarm" in command):
            current_time = datetime.now().strftime('%I:%M %p')
            self.speak(f"The current time is {current_time}")
            return True
        elif "what is today's date" in command or "current date" in command or "date today" in command:
            current_date = datetime.now().strftime('%A, %B %d, %Y')
            self.speak(f"Today is {current_date}")
            return True
        elif "set alarm" in command or "wake me up" in command :
            return self._set_alarm(command)
        elif "set timer" in command or "timer for" in command: # "timer for 5 minutes"
            return self._set_timer(command)
        elif re.match(r"(\d+)\s*(minute|second|hour) timer", command): # "5 minute timer"
            return self._set_timer(command)
        return False

    def _set_alarm(self, command: str) -> bool:
        """Set an alarm based on voice command"""
        try:
            # Enhanced pattern matching for various alarm formats
            patterns = [
                # More specific first
                r'(?:set alarm|wake me up) for (\d{1,2}):(\d{2})\s*(am|pm)',      # "set alarm for 7:30 am/pm" (explicit am/pm)
                r'(?:set alarm|wake me up) for (\d{1,2}):(\d{2})',                # "set alarm for 7:30" (assume context or next occurrence)
                r'(?:set alarm|wake me up) at (\d{1,2}):(\d{2})\s*(am|pm)',       # "set alarm at 7:30 am/pm"
                r'(?:set alarm|wake me up) at (\d{1,2}):(\d{2})',                 # "set alarm at 7:30"
                r'(?:set alarm|wake me up) for (\d{1,2})\s*(am|pm)',              # "set alarm for 7 am/pm"
                r'(?:set alarm|wake me up) at (\d{1,2})\s*(am|pm)',               # "set alarm at 7 am/pm"
                r'(?:set alarm|wake me up) in (\d+)\s*(minute|minutes|hour|hours)',# "set alarm in 30 minutes/1 hour"
            ]

            alarm_time = None
            alarm_label = "Alarm"
            now = datetime.now()

            for pattern in patterns:
                match = re.search(pattern, command.lower())
                if match:
                    groups = match.groups()
                    logger.debug(f"Alarm pattern matched: {pattern} with groups: {groups}")

                    if "in" in pattern:  # Relative time (in X minutes/hours)
                        duration = int(groups[0])
                        unit = groups[1]
                        if "hour" in unit:
                            alarm_time = now + timedelta(hours=duration)
                            alarm_label = f"Alarm in {duration} {'hour' if duration == 1 else 'hours'}"
                        else: # minutes
                            alarm_time = now + timedelta(minutes=duration)
                            alarm_label = f"Alarm in {duration} {'minute' if duration == 1 else 'minutes'}"
                    else:  # Absolute time (at specific time)
                        hour = int(groups[0])
                        minute = 0
                        am_pm = None

                        if len(groups) >= 2 and groups[1] and groups[1].isdigit(): # e.g. (\d{1,2}):(\d{2})
                            minute = int(groups[1])
                            if len(groups) >= 3 and groups[2]: am_pm = groups[2].lower()
                        elif len(groups) >= 2 and groups[1]: # e.g. (\d{1,2})\s*(am|pm)
                            am_pm = groups[1].lower()
                        
                        # Adjust hour for AM/PM
                        if am_pm == 'pm' and 1 <= hour <= 11:
                            hour += 12
                        elif am_pm == 'am' and hour == 12: # 12 AM is midnight
                            hour = 0
                        
                        # If no AM/PM, guess based on current time (e.g., if it's 8 PM and user says "alarm for 7", assume 7 AM)
                        # This can be complex. A simpler approach: if time is in past, set for next day.
                        alarm_time_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

                        if alarm_time_today <= now:
                             alarm_time = alarm_time_today + timedelta(days=1)
                        else:
                             alarm_time = alarm_time_today
                        
                        time_str = alarm_time.strftime('%I:%M %p')
                        alarm_label = f"Alarm for {time_str}"
                    break # Found a match

            if not alarm_time:
                self.speak(
                    "Sorry, I couldn't understand the alarm time. "
                    "Please try saying something like 'set alarm for 7:30 AM' or 'set alarm in 30 minutes'."
                )
                return False

            # Store alarm in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alarms (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alarm_time TIMESTAMP NOT NULL,
                        label TEXT NOT NULL,
                        active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute(
                    "INSERT INTO alarms (alarm_time, label) VALUES (?, ?)",
                    (alarm_time.isoformat(), alarm_label)
                )
                alarm_id = cursor.lastrowid
                conn.commit()

            # Start alarm thread
            alarm_thread = threading.Thread(
                target=self._alarm_worker,
                args=(alarm_id, alarm_time, alarm_label),
                daemon=True
            )
            alarm_thread.start()

            # Confirmation message
            time_until = alarm_time - datetime.now()
            total_seconds_until = int(time_until.total_seconds())
            
            if total_seconds_until < 0 : total_seconds_until = 0 # Should not happen if logic above is correct

            hours, remainder = divmod(total_seconds_until, 3600)
            minutes, seconds = divmod(remainder, 60)

            time_desc_parts = []
            if hours > 0:
                time_desc_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0 : 
                time_desc_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            
            if not time_desc_parts and seconds > 0: # if less than a minute
                 time_desc_parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
            elif not time_desc_parts and total_seconds_until == 0 : # if exactly on time
                 time_desc_parts.append("just now")


            time_desc = " and ".join(time_desc_parts) if time_desc_parts else "less than a second"


            self.speak(
                f"{alarm_label} set for {alarm_time.strftime('%I:%M %p')}. "
                f"That's in {time_desc}."
            )
            logger.info(f"Alarm set: {alarm_label} at {alarm_time.isoformat()}")

            return True

        except Exception as e:
            logger.error(f"Alarm setting error: {e}", exc_info=True)
            self.speak("Sorry, I had trouble setting the alarm.")
            return False

    def _set_timer(self, command: str) -> bool:
        """Set a timer based on voice command"""
        try:
            # Pattern matching for timer formats
            patterns = [
                r'(?:set timer for|timer for|timer)\s*(\d+)\s*(second|seconds|minute|minutes|hour|hours)', 
                r'(\d+)\s*(second|seconds|minute|minutes|hour|hours)\s*timer'
            ]
            
            duration_seconds = 0
            timer_label = "Timer"
            
            for pattern in patterns:
                match = re.search(pattern, command.lower())
                if match:
                    groups = match.groups()
                    amount = int(groups[0])
                    unit = groups[1] 
                    
                    if "hour" in unit:
                        duration_seconds = amount * 3600
                        timer_label = f"{amount} {'hour' if amount == 1 else 'hours'} timer"
                    elif "minute" in unit:
                        duration_seconds = amount * 60
                        timer_label = f"{amount} {'minute' if amount == 1 else 'minutes'} timer"
                    else:  # seconds
                        duration_seconds = amount
                        timer_label = f"{amount} {'second' if amount == 1 else 'seconds'} timer"
                    
                    break 
            
            if duration_seconds == 0:
                self.speak("Sorry, I couldn't understand the timer duration. Please try 'timer for 5 minutes' or '30 second timer'.")
                return False
            
            # Calculate end time
            end_time = datetime.now() + timedelta(seconds=duration_seconds)
            
            # Store timer in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS timers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        duration_seconds INTEGER NOT NULL,
                        end_time TIMESTAMP NOT NULL,
                        label TEXT NOT NULL,
                        active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute(
                    "INSERT INTO timers (duration_seconds, end_time, label) VALUES (?, ?, ?)",
                    (duration_seconds, end_time.isoformat(), timer_label)
                )
                timer_id = cursor.lastrowid
                conn.commit()
            
            timer_thread = threading.Thread(
                target=self._timer_worker,
                args=(timer_id, duration_seconds, timer_label),
                daemon=True
            )
            timer_thread.start()
            
            self.speak(f"{timer_label} started.")
            logger.info(f"Timer set: {timer_label} for {duration_seconds} seconds, ending at {end_time.isoformat()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Timer setting error: {e}", exc_info=True)
            self.speak("Sorry, I had trouble setting the timer.")
            return False

    def _alarm_worker(self, alarm_id: int, alarm_time: datetime, label: str):
        """Background worker for alarm"""
        try:
            time_to_wait = (alarm_time - datetime.now()).total_seconds()
            if time_to_wait > 0:
                logger.info(f"Alarm worker for '{label}' (ID: {alarm_id}) waiting for {time_to_wait:.2f} seconds.")
                # Sleep in chunks to allow is_running check
                sleep_interval = 1 
                while time_to_wait > 0 and self.is_running:
                    actual_sleep = min(time_to_wait, sleep_interval)
                    time.sleep(actual_sleep)
                    time_to_wait -= actual_sleep
            
            if not self.is_running:
                logger.info(f"Alarm worker for '{label}' (ID: {alarm_id}) cancelled due to assistant shutdown.")
                return 
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT active FROM alarms WHERE id = ?", (alarm_id,))
                result = cursor.fetchone()
                
                if not result or not result[0]: 
                    logger.info(f"Alarm '{label}' (ID: {alarm_id}) was cancelled or removed before triggering.")
                    return  
            
            logger.info(f"Alarm TRIGGERED: {label} (ID: {alarm_id})")
            
            alarm_messages = [
                f"Alarm! {label}",
                "Wake up! Your alarm is going off!",
            ]
            
            for i, message in enumerate(alarm_messages):
                if not self.is_running: break
                self.speak(message, interrupt=False) 
                
                try:
                    if not pygame.mixer.get_init(): pygame.mixer.init()
                    if os.path.exists('alarm_sound.mp3'): # You should have an alarm_sound.mp3
                        pygame.mixer.music.load('alarm_sound.mp3')
                        pygame.mixer.music.play()
                        play_start_time = time.time()
                        while pygame.mixer.music.get_busy() and (time.time() - play_start_time < 5) and self.is_running : 
                            time.sleep(0.1)
                        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
                    else: logger.warning("alarm_sound.mp3 not found.")
                except Exception as e: logger.error(f"Alarm sound play error: {e}")
                
                if i < len(alarm_messages) - 1 and self.is_running: time.sleep(5) # Longer pause between alarm messages
            
            if self.is_running : 
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE alarms SET active = FALSE WHERE id = ?", (alarm_id,))
                    conn.commit()
                    logger.info(f"Alarm '{label}' (ID: {alarm_id}) marked as inactive.")
                    
        except Exception as e:
            logger.error(f"Alarm worker error for '{label}' (ID: {alarm_id}): {e}", exc_info=True)

    def _timer_worker(self, timer_id: int, duration_seconds: int, label: str):
        """Background worker for timer"""
        try:
            remaining_seconds = duration_seconds
            logger.info(f"Timer worker for '{label}' (ID: {timer_id}) started for {duration_seconds}s.")
            
            milestones_announced = set() 
            milestones = []
            if duration_seconds >= 300: milestones.extend([s for s in [60, 30] if s < duration_seconds]) # 1min, 30s for 5min+ timers
            elif duration_seconds >= 60: milestones.extend([s for s in [30, 10] if s < duration_seconds]) # 30s, 10s for 1min+ timers
            elif duration_seconds > 10: milestones.append(10) # 10s for >10s timers
            milestones = sorted(list(set(milestones)), reverse=True)

            sleep_interval = 1
            while remaining_seconds > 0 and self.is_running:
                for ms in milestones:
                    if remaining_seconds <= ms and ms not in milestones_announced:
                        if ms >= 60:
                            minutes_left = ms // 60
                            self.speak(f"{minutes_left} {'minute' if minutes_left == 1 else 'minutes'} remaining on {label}", interrupt=False)
                        else:
                            self.speak(f"{ms} seconds remaining on {label}", interrupt=False)
                        milestones_announced.add(ms)
                
                actual_sleep = min(remaining_seconds, sleep_interval)
                time.sleep(actual_sleep)
                remaining_seconds -= actual_sleep
            
            if not self.is_running:
                logger.info(f"Timer worker for '{label}' (ID: {timer_id}) cancelled due to assistant shutdown.")
                return
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT active FROM timers WHERE id = ?", (timer_id,))
                result = cursor.fetchone()
                if not result or not result[0]:
                    logger.info(f"Timer '{label}' (ID: {timer_id}) was cancelled or removed before finishing.")
                    return
            
            logger.info(f"Timer FINISHED: {label} (ID: {timer_id})")
            timer_messages = [ f"Timer finished! Your {label} is complete.", "Time's up!" ]
            
            for i, message in enumerate(timer_messages):
                if not self.is_running: break
                self.speak(message, interrupt=False)
                try: # Play sound
                    if not pygame.mixer.get_init(): pygame.mixer.init()
                    if os.path.exists('timer_sound.mp3'): # You should have a timer_sound.mp3
                        pygame.mixer.music.load('timer_sound.mp3')
                        pygame.mixer.music.play()
                        play_start_time = time.time()
                        while pygame.mixer.music.get_busy() and (time.time() - play_start_time < 3) and self.is_running:
                            time.sleep(0.1)
                        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
                    else: logger.warning("timer_sound.mp3 not found.")
                except Exception as e: logger.error(f"Timer sound play error: {e}")
                if i < len(timer_messages) - 1 and self.is_running: time.sleep(3)
            
            if self.is_running:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE timers SET active = FALSE WHERE id = ?", (timer_id,))
                    conn.commit()
                    logger.info(f"Timer '{label}' (ID: {timer_id}) marked as inactive.")
        except Exception as e:
            logger.error(f"Timer worker error for '{label}' (ID: {timer_id}): {e}", exc_info=True)

    def _handle_entertainment_commands(self, command: str) -> bool:
        """Handle entertainment commands"""
        if "play music" in command or ("play" in command and "song" in command):
            music_query_match = re.search(r'play (?:music|song)?\s*(?:called|named)?\s*(.+)', command, re.IGNORECASE)
            if music_query_match:
                music_query = music_query_match.group(1).strip()
                if music_query: 
                    return self.stream_youtube_audio(music_query)
                else:
                    self.speak("What music would you like to play?")
                    return True 
            else: 
                self.speak("What song or artist would you like me to play?")
                return True

        elif "stop music" in command and self.get_state() == AssistantState.MUSIC_PLAYING:
            pygame.mixer.music.stop()
            self.speak("Music stopped")
            self.set_state(AssistantState.IDLE)
            return True
        elif "tell me a joke" in command or "joke" in command:
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
            city_match = re.search(r'weather in (.+)', command, re.IGNORECASE)
            city = self.config['city_name'] # Default city
            if city_match:
                city = city_match.group(1).strip()
            return self._get_weather(city) 
        elif "wikipedia" in command or "what is" in command or "who is" in command or "tell me about" in command:
            query_match = re.search(r'(?:wikipedia|what is|who is|tell me about)\s+(.+)', command, re.IGNORECASE)
            if query_match:
                query = query_match.group(1).strip()
                if query:
                    return self._search_wikipedia(query) 
                else:
                    self.speak("What topic would you like to know about?")
                    return True
            else: 
                self.speak("What would you like me to look up on Wikipedia?")
                return True
        return False

    def _handle_memory_commands(self, command: str) -> bool:
        """Handle memory-related commands"""
        if "remember that" in command or "remember this" in command:
            content_match = re.search(r'remember (?:that|this)\s+(.+)', command, re.IGNORECASE)
            if content_match:
                content = content_match.group(1).strip()
                if content:
                    return self._remember(content) 
                else:
                    self.speak("What should I remember?")
                    return True
            else:
                self.speak("What would you like me to remember?")
                return True
        elif "what did i tell you" in command or "recall memories" in command or "do you remember" in command:
            keyword_match = re.search(r'do you remember (.+)', command, re.IGNORECASE)
            keyword = None
            if keyword_match:
                keyword = keyword_match.group(1).strip()
            return self._recall_memories(search_term=keyword)
        elif "forget all memories" in command: # More specific command for forgetting
            # Add confirmation step here
            self.speak("Are you sure you want me to forget all memories? This cannot be undone.")
            # Listen for "yes" or "no" - this requires a follow-up listening mechanism
            # For now, just log and don't delete.
            logger.warning("Received 'forget all memories' but confirmation not implemented.")
            self.speak("For safety, I need a confirmation to forget all memories. This feature is not fully active yet.")
            return True
        elif "forget that" in command or "forget about" in command:
             self.speak("I can only forget all memories right now, not specific ones. Would you like to forget everything?")
             # Again, needs confirmation logic.
             return True
        return False

    # --- Placeholder methods for assumed functionality ---
    def _get_weather(self, city_name: Optional[str] = None) -> bool:
        target_city = city_name if city_name else self.config['city_name']
        logger.info(f"Fetching weather for {target_city}...")
        api_key = self.config.get('weather_api_key')
        if not api_key:
            self.speak("Weather API key is not configured.")
            return False
        
        # Using OpenWeatherMap API as an example
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": target_city,
            "appid": api_key,
            "units": "metric" # For Celsius
        }
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
            data = response.json()
            
            if data.get("cod") == 200: # Check for success code from OpenWeatherMap
                main = data.get("main", {})
                weather_list = data.get("weather", [])
                
                temp = main.get("temp")
                feels_like = main.get("feels_like")
                humidity = main.get("humidity")
                description = weather_list[0].get("description") if weather_list else "not available"
                
                if temp is not None:
                    weather_report = f"The weather in {data.get('name', target_city)} is {description}. " \
                                     f"The temperature is {temp}°C"
                    if feels_like is not None and abs(temp - feels_like) > 1: # Only mention if significantly different
                        weather_report += f", but it feels like {feels_like}°C"
                    if humidity is not None:
                         weather_report += f". Humidity is at {humidity}%."
                    self.speak(weather_report)
                else:
                    self.speak(f"Sorry, I couldn't get detailed weather for {target_city}.")
                return True
            else:
                error_message = data.get("message", "Unknown error")
                self.speak(f"Sorry, I couldn't fetch the weather for {target_city}. Error: {error_message}")
                logger.error(f"OpenWeatherMap API error for {target_city}: {error_message} (Code: {data.get('cod')})")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request error for {target_city}: {e}")
            self.speak(f"Sorry, I'm having trouble connecting to the weather service for {target_city}.")
            return False
        except Exception as e: # Catch any other unexpected error
            logger.error(f"Unexpected error fetching weather for {target_city}: {e}", exc_info=True)
            self.speak(f"An unexpected error occurred while fetching weather for {target_city}.")
            return False


    def _search_wikipedia(self, query: str) -> bool:
        logger.info(f"Searching Wikipedia for: {query}")
        try:
            # Set a user-agent for Wikipedia API requests
            wikipedia.set_user_agent("EnhancedVoiceAssistant/1.0 (https://example.com/assistant; assistant@example.com)")
            
            # Try to get a page first to handle disambiguation better
            try:
                page = wikipedia.page(query, auto_suggest=False, redirect=True) # auto_suggest=False to avoid suggesting too broadly initially
                summary = wikipedia.summary(page.title, sentences=2) # Get summary of the specific page title
            except wikipedia.exceptions.DisambiguationError as e:
                options = e.options[:3] 
                if options:
                    self.speak(f"'{query}' could mean a few things, like {', '.join(options)}. Which one did you mean?")
                else:
                    self.speak(f"'{query}' is ambiguous and I couldn't find specific options. Please try a more specific query.")
                return True 
            except wikipedia.exceptions.PageError:
                # If page not found directly, try with auto_suggest in summary
                summary = wikipedia.summary(query, sentences=2, auto_suggest=True, redirect=True)

            self.speak(f"According to Wikipedia: {summary}")
            return True
        except wikipedia.exceptions.PageError: # Catch PageError from the second summary attempt
            self.speak(f"Sorry, I couldn't find a Wikipedia page for '{query}'.")
            return False
        except Exception as e: # Catch other potential errors (network, etc.)
            logger.error(f"Wikipedia search error for '{query}': {e}", exc_info=True)
            self.speak("Sorry, I had trouble searching Wikipedia right now.")
            return False

    def _remember(self, content: str, category: str = 'general', importance: int = 1) -> bool:
        logger.info(f"Remembering: '{content}' under category '{category}' with importance {importance}")
        if not content:
            self.speak("There's nothing to remember.")
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO memories (content, category, importance) VALUES (?, ?, ?)",
                    (content, category, importance)
                )
                conn.commit()
            self.speak("Okay, I'll remember that.")
            return True
        except Exception as e:
            logger.error(f"Error remembering content '{content}': {e}", exc_info=True)
            self.speak("Sorry, I had trouble remembering that.")
            return False

    def _recall_memories(self, category: Optional[str] = None, search_term: Optional[str] = None, limit: int = 3) -> bool:
        logger.info(f"Recalling memories (Category: {category}, Search: {search_term}, Limit: {limit})")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query_sql = "SELECT content FROM memories"
                params = []
                conditions = []

                if category:
                    conditions.append("category = ?")
                    params.append(category)
                if search_term:
                    conditions.append("content LIKE ?")
                    params.append(f"%{search_term}%") # Simple wildcard search
                
                if conditions:
                    query_sql += " WHERE " + " AND ".join(conditions)
                
                query_sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query_sql, tuple(params))
                memories = cursor.fetchall()
                
                if memories:
                    if search_term:
                        response_intro = f"Here's what I remember related to '{search_term}': "
                    elif category:
                        response_intro = f"Here are some memories from the '{category}' category: "
                    else:
                        response_intro = "Here's what I remember: "
                    
                    response_items = [mem[0] for mem in memories]
                    # Join with a slight pause for better speech flow
                    self.speak(response_intro + ". ".join(response_items))
                else:
                    if search_term:
                        self.speak(f"I don't have any memories matching '{search_term}'.")
                    elif category:
                         self.speak(f"I don't have any memories in the '{category}' category.")
                    else:
                        self.speak("My memory is currently empty, or I couldn't find anything relevant.")
            return True
        except Exception as e:
            logger.error(f"Error recalling memories: {e}", exc_info=True)
            self.speak("Sorry, I had trouble recalling my memories.")
            return False

    def _log_command(self, command: str, success: bool, execution_time: float):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO command_history (command, success, execution_time) VALUES (?, ?, ?)",
                    (command, success, execution_time)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log command '{command}': {e}", exc_info=True)

    def _handle_system_commands(self, command: str) -> bool:
        if "shutdown assistant" in command or "turn off yourself" in command: # Make it more specific
            self.speak("Are you sure you want me to shut down? This will stop the assistant program.")
            # This requires a follow-up listening mechanism for "yes"/"no"
            # For now, just log and state the need for confirmation.
            logger.warning("Shutdown command received. Confirmation mechanism not yet implemented.")
            self.speak("To shut down, please confirm. For now, I'll stay active.")
            # In a real scenario, you'd set a flag or state to listen for confirmation.
            # If confirmed:
            # self.is_running = False
            # self.speak("Shutting down. Goodbye.")
            # self.executor.shutdown(wait=True) # Cleanly shutdown thread pool
            # pygame.quit()
            # sys.exit() # Or raise a specific exception to be caught by the main loop for clean exit
            return True
        return False

    def _handle_image_recognition(self):
        # This is a placeholder. Real image recognition would involve:
        # 1. Capturing an image (e.g., from a webcam connected to the system running this script).
        # 2. Sending the image to a service like Google Cloud Vision API or a local model.
        # 3. Processing the response.
        self.speak("Image recognition is a planned feature but not fully implemented yet. I would need access to a camera and an image processing model.")
        logger.info("Image recognition called (not implemented).")
        # Example of how you might use Gemini for image understanding if you had an image:
        # try:
        #     # Assume `image_bytes` is the image data
        #     # img = PIL.Image.open(io.BytesIO(image_bytes))
        #     # response = self.gemini_client.generate_content(["Describe this image", img]) # Fictional Gemini client usage
        #     # self.speak(response.text)
        #     pass
        # except Exception as e:
        #     logger.error(f"Image recognition error: {e}")
        #     self.speak("I had trouble analyzing the image.")
        return True


    def _handle_general_conversation(self, command: str):
        logger.info(f"Handling general conversation: '{command}'")
        try:
            # Ensure genai is configured (usually done in _setup_clients)
            # model = genai.GenerativeModel('gemini-pro') # Or your preferred model
            # response = model.generate_content(command)
            # self.speak(response.text)
            
            # Using the configured client approach if that's how genai is structured for you
            # This part depends heavily on the exact `google-generativeai` library usage.
            # The library typically uses `genai.GenerativeModel('model-name')` then `model.generate_content()`
            # The `genai.Client` might be for other purposes or an older API style.
            # Assuming the common usage:
            if not hasattr(self, '_gemini_model_instance'): # Cache model instance
                 self._gemini_model_instance = genai.GenerativeModel('gemini-1.0-pro-latest') # Use a valid model name

            response = self._gemini_model_instance.generate_content(command)
            
            # Check for empty or blocked response
            if response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                if generated_text:
                    self.speak(generated_text)
                else:
                    logger.warning(f"Gemini response for '{command}' was empty or had no text parts.")
                    self.speak("I'm not sure how to respond to that right now.")
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                 logger.warning(f"Gemini request for '{command}' blocked. Reason: {response.prompt_feedback.block_reason_message}")
                 self.speak("I'm sorry, I can't respond to that request due to safety guidelines.")
            else:
                logger.warning(f"Gemini returned no parts and no block reason for '{command}'.")
                self.speak("I'm having a little trouble formulating a response right now.")

        except Exception as e:
            logger.error(f"Gemini conversation error for '{command}': {e}", exc_info=True)
            self.speak("I'm having a bit of trouble thinking right now. Please try again later.")
        return True

    def main_loop(self):
        """Main operational loop for wake word listening and command processing."""
        logger.info("Starting main assistant loop...")
        self.speak("Enhanced Voice Assistant is now active.")

        try:
            # Initialize PyAudio stream for wake word detection
            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                format=self.FORMAT,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
                # input_device_index=None # Or specify your microphone's index
            )
            logger.info("Audio stream for wake word opened.")

            while self.is_running:
                if self.get_state() not in [AssistantState.SPEAKING, AssistantState.MUSIC_PLAYING, AssistantState.PROCESSING]:
                    try:
                        # Read audio chunk for wake word
                        audio_chunk = audio_stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                        
                        # Convert to numpy array for OpenWakeWord
                        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
                        
                        # Predict wake word
                        prediction = self.wake_word_model.predict(audio_np)
                        
                        # Check for 'hey_jarvis' or other wake words configured in OpenWakeWord model
                        # The key in `prediction` dict depends on the model used.
                        # Common models might have 'hey_jarvis', 'alexa', etc.
                        # You might need to inspect `self.wake_word_model.model_definition` or `prediction.keys()`
                        
                        # Example: assuming 'hey_jarvis' is a key if that model is loaded.
                        # Or, iterate through predictions if multiple wake words are possible.
                        primary_wake_word = "hey_jarvis" # Replace with your actual wake word key if different
                        if primary_wake_word in prediction and prediction[primary_wake_word] > self.config['wake_word_sensitivity']:
                            current_time = time.time()
                            if current_time - self.last_wake_detection > self.wake_word_cooldown:
                                self.last_wake_detection = current_time
                                logger.info(f"Wake word '{primary_wake_word}' detected!")
                                self.interrupt_speech() # Stop any ongoing TTS
                                if pygame.mixer.music.get_busy(): # If music is playing, pause it or lower volume
                                    # pygame.mixer.music.pause() # Or implement volume ducking
                                    logger.info("Wake word detected during music playback.")
                                    # For now, we'll let music continue but listen over it.
                                    # A better UX would be to pause or lower music volume.

                                self.set_state(AssistantState.LISTENING)
                                self.speak("Yes?", interrupt=True) # Acknowledge wake word

                                # Listen for command
                                with self.microphone as source:
                                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5) # Quick re-adjust
                                    logger.info("Listening for command...")
                                    try:
                                        audio_command = self.recognizer.listen(
                                            source, 
                                            timeout=5, # Give user 5 seconds to start speaking
                                            phrase_time_limit=self.config['max_command_duration']
                                        )
                                        command_text = self.recognizer.recognize_google(audio_command).lower()
                                        logger.info(f"Command received: {command_text}")
                                        
                                        self.set_state(AssistantState.PROCESSING)
                                        self.executor.submit(self.enhanced_command_processor, command_text)
                                        # enhanced_command_processor will set state to IDLE or SPEAKING

                                    except sr.WaitTimeoutError:
                                        logger.warning("No command heard after wake word.")
                                        self.speak("I didn't hear anything.")
                                        self.set_state(AssistantState.IDLE)
                                    except sr.UnknownValueError:
                                        logger.warning("Could not understand audio after wake word.")
                                        self.speak("Sorry, I didn't catch that.")
                                        self.set_state(AssistantState.IDLE)
                                    except sr.RequestError as e:
                                        logger.error(f"Speech recognition request error: {e}")
                                        self.speak("Sorry, I'm having trouble with speech recognition right now.")
                                        self.set_state(AssistantState.IDLE)
                                    finally:
                                        if pygame.mixer.music.get_busy() and not pygame.mixer.music.get_pos() > 0: # If music was paused
                                            # pygame.mixer.music.unpause() # Resume music
                                            pass

                        # Check other wake words if your model supports multiple
                        # for wake_word_key, confidence in prediction.items():
                        #    if confidence > self.config['wake_word_sensitivity'] and wake_word_key != primary_wake_word:
                        #        # Handle other wake words if necessary
                        #        logger.info(f"Alternate wake word '{wake_word_key}' detected.")
                        #        pass


                    except IOError as e:
                        if e.errno == pyaudio.paInputOverflowed:
                            logger.warning("PyAudio input overflowed. Skipping frame.")
                        else:
                            logger.error(f"PyAudio IO error in main loop: {e}", exc_info=True)
                            self.speak("There's an issue with the audio input. Please check the microphone.")
                            self.is_running = False # Stop if audio is critically failing
                    except Exception as e:
                        logger.error(f"Error in wake word detection loop: {e}", exc_info=True)
                        time.sleep(1) # Avoid rapid error looping
                else:
                    time.sleep(0.1) # Brief sleep if assistant is busy (speaking, music, processing)

        except Exception as e:
            logger.critical(f"Critical error in main_loop: {e}", exc_info=True)
            self.speak("A critical error occurred, and I need to shut down.")
        finally:
            logger.info("Exiting main assistant loop.")
            if 'audio_stream' in locals() and audio_stream.is_active():
                audio_stream.stop_stream()
                audio_stream.close()
            if 'pa' in locals():
                pa.terminate()
            
            self.executor.shutdown(wait=False, cancel_futures=True) # Aggressively shutdown threads
            pygame.mixer.quit()
            # pygame.quit() # If you initialized more of pygame globally

            logger.info("Enhanced Voice Assistant has shut down.")
            # Potentially speak a final "shutting down" message if possible,
            # but TTS might not work if pygame.mixer is already quit.

if __name__ == '__main__':
    try:
        assistant = EnhancedVoiceAssistant()
        assistant.main_loop() # Start the main operational loop
    except ValueError as ve: # Catch config errors from __init__
        logger.critical(f"Configuration Error: {ve}. Assistant cannot start.")
    except Exception as e:
        logger.critical(f"Unhandled exception at assistant startup: {e}", exc_info=True)

