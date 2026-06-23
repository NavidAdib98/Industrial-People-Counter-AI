"""
Simple settings loader - only loads .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Settings:
    """
    Simple settings loader from .env file
    """
    
    def __init__(self):
        """Load settings from .env file"""
        # Get project root
        self.project_root = Path(__file__).parent.parent.parent  # Changed: now in utils/
        
        # Load .env file
        env_path = self.project_root / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded .env file")
        else:
            print(f"⚠️  No .env file found, using defaults")
        
        # Load only essential settings
        self.VIDEO_PATH = self._get('VIDEO_PATH', 'videos/test_video.mp4')
        self.MODEL_PATH = self._get('MODEL_PATH', 'models/yolo11n.pt')
        self.CONF_THRESHOLD = self._get_float('CONF_THRESHOLD', 0.4)
        self.DEVICE = self._get('DEVICE', 'cpu')
        self.TRACKER_TYPE = self._get('TRACKER_TYPE', 'bytetrack.yaml')
        self.SAVE_OUTPUT = self._get_bool('SAVE_OUTPUT', True)
        self.SHOW_PREVIEW = self._get_bool('SHOW_PREVIEW', True)
        
        # Video resize
        self.RESIZE_VIDEO = self._get_bool('RESIZE_VIDEO', False)
        self.RESIZE_WIDTH = self._get_int('RESIZE_WIDTH', 640)
        self.RESIZE_HEIGHT = self._get_int('RESIZE_HEIGHT', 480)
        
        # Polygon file path
        self.POLYGON_FILE = self._get('POLYGON_FILE', None)
        
        # Resolve paths
        self.VIDEO_PATH = self._resolve_path(self.VIDEO_PATH)
        self.MODEL_PATH = self._resolve_path(self.MODEL_PATH)
        
        # Create output directory
        self.OUTPUT_DIR = self.project_root / 'outputs'
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_VIDEO = self.OUTPUT_DIR / 'tracked_video.mp4'
        
        # Create other directories
        (self.project_root / 'models').mkdir(parents=True, exist_ok=True)
        (self.project_root / 'polygons').mkdir(parents=True, exist_ok=True)
    
    def _get(self, key, default):
        """Get string value from environment"""
        return os.getenv(key, default)
    
    def _get_int(self, key, default):
        """Get integer value from environment"""
        try:
            return int(os.getenv(key, default))
        except ValueError:
            return default
    
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
    
    def print_settings(self):
        """Print current settings"""
        print("=" * 50)
        print("📋 CURRENT SETTINGS")
        print("=" * 50)
        print(f"VIDEO_PATH: {self.VIDEO_PATH}")
        print(f"MODEL_PATH: {self.MODEL_PATH}")
        print(f"CONF_THRESHOLD: {self.CONF_THRESHOLD}")
        print(f"DEVICE: {self.DEVICE}")
        print(f"TRACKER_TYPE: {self.TRACKER_TYPE}")
        print(f"RESIZE_VIDEO: {self.RESIZE_VIDEO}")
        if self.RESIZE_VIDEO:
            print(f"RESIZE: {self.RESIZE_WIDTH}x{self.RESIZE_HEIGHT}")
        print(f"SAVE_OUTPUT: {self.SAVE_OUTPUT}")
        print(f"SHOW_PREVIEW: {self.SHOW_PREVIEW}")
        print(f"POLYGON_FILE: {self.POLYGON_FILE or 'None'}")
        print(f"OUTPUT_VIDEO: {self.OUTPUT_VIDEO}")
        print("=" * 50)
        print()


# Create global settings instance
settings = Settings()


if __name__ == "__main__":
    settings.print_settings()