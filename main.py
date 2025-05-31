#!/usr/bin/env python3
"""
Enhanced Voice Assistant - Main Entry Point
Author: Your Name
Version: 1.0.0
"""

import sys
import os
import logging
from assistant import EnhancedVoiceAssistant

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('assistant.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'pygame', 'speech_recognition', 'edge_tts', 'openwakeword',
        'google.generativeai', 'homeassistant_api', 'requests',
        'yt_dlp', 'youtubesearchpython', 'cv2', 'numpy', 'pandas'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {', '.join(missing_modules)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("1. Copy .env.example to .env")
        print("2. Fill in your API keys")
        return False
    return True

def main():
    """Main function"""
    print("🎤 Enhanced Voice Assistant v1.0.0")
    print("=" * 40)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        sys.exit(1)
    
    try:
        # Initialize and run assistant
        logger.info("Starting Enhanced Voice Assistant...")
        assistant = EnhancedVoiceAssistant()
        assistant.run()
        
    except KeyboardInterrupt:
        logger.info("Assistant stopped by user")
        print("\n👋 Goodbye!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Error: {e}")
        print("Check the logs for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
