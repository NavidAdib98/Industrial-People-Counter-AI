"""
Simplified settings loader
Loads configuration from .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Settings:
    """
    Application settings loaded from environment variables
    """
    
    def __init__(self):
        """Load settings from .env file"""
        # Get project root
        self.project_root = Path(__file__).parent.parent
        
        # Load .env file
        env_path = self.project_root / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded .env file")
        else:
            print(f"⚠️  No .env file found, using defaults")
        
        # Load settings with defaults
        self.VIDEO_PATH = self._get('VIDEO_PATH', 'videos/test_video.mp4')
        self.MODEL_NAME = self._get('MODEL_NAME', 'yolo11n.pt')
        self.CONF_THRESHOLD = self._get_float('CONF_THRESHOLD', 0.4)
        self.DEVICE = self._get('DEVICE', 'cpu')
        self.TRACKER_TYPE = self._get('TRACKER_TYPE', 'bytetrack.yaml')
        self.SAVE_OUTPUT = self._get_bool('SAVE_OUTPUT', True)
        self.SHOW_PREVIEW = self._get_bool('SHOW_PREVIEW', True)
        
        # Resolve paths
        self.VIDEO_PATH = self._resolve_path(self.VIDEO_PATH)
        
        # Create output directory
        self.OUTPUT_DIR = self.project_root / 'outputs'
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.OUTPUT_VIDEO = self.OUTPUT_DIR / 'tracked_video.mp4'
    
    def _get(self, key, default):
        """Get string value from environment"""
        return os.getenv(key, default)
    
    def _get_float(self, key, default):
        """Get float value from environment"""
        try:
            return float(os.getenv(key, default))
        except ValueError:
            return default
    
    def _get_bool(self, key, default):
        """Get boolean value from environment"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _resolve_path(self, path):
        """Resolve path relative to project root"""
        if os.path.isabs(path):
            return Path(path)
        return self.project_root / path
    
    def get_model_path(self):
        """Get full model path"""
        return str(self.project_root / 'models' / self.MODEL_NAME)
    
    def print_settings(self):
        """Print current settings"""
        print("=" * 50)
        print("📋 CURRENT SETTINGS")
        print("=" * 50)
        print(f"VIDEO_PATH: {self.VIDEO_PATH}")
        print(f"MODEL_NAME: {self.MODEL_NAME}")
        print(f"CONF_THRESHOLD: {self.CONF_THRESHOLD}")
        print(f"DEVICE: {self.DEVICE}")
        print(f"TRACKER_TYPE: {self.TRACKER_TYPE}")
        print(f"SAVE_OUTPUT: {self.SAVE_OUTPUT}")
        print(f"SHOW_PREVIEW: {self.SHOW_PREVIEW}")
        print(f"OUTPUT_VIDEO: {self.OUTPUT_VIDEO}")
        print("=" * 50)
        print()


# Create global settings instance
settings = Settings()


if __name__ == "__main__":
    settings.print_settings()