"""
Simple settings loader - only loads .env file
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv


class Settings:
    """
    Simple settings loader from .env file
    """
    
    def __init__(self):
        """Load settings from .env file"""
        # Get project root
        self.project_root = Path(__file__).parent.parent.parent
        
        # Load .env file
        env_path = self.project_root / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded .env file")
        else:
            print(f"⚠️  No .env file found, using defaults")
        
        # ==================== LOGGING ====================
        self.LOG_LEVEL = self._get('LOG_LEVEL', 'INFO')
        self._setup_logging()
        
        # ==================== VIDEO ====================
        self.VIDEO_PATH = self._get('VIDEO_PATH', 'videos/test_video.mp4')
        
        # Video resize
        self.RESIZE_VIDEO = self._get_bool('RESIZE_VIDEO', False)
        self.RESIZE_WIDTH = self._get_int('RESIZE_WIDTH', 640)
        self.RESIZE_HEIGHT = self._get_int('RESIZE_HEIGHT', 480)
        
        # ==================== MODEL ====================
        self.MODEL_PATH = self._get('MODEL_PATH', 'models/yolo11n.pt')
        
        # ==================== DETECTION ====================
        self.CONF_THRESHOLD = self._get_float('CONF_THRESHOLD', 0.4)
        self.IOU_THRESHOLD = self._get_float('IOU_THRESHOLD', 0.5)
        self.MAX_DETECTIONS = self._get_int('MAX_DETECTIONS', 300)
        self.HALF_PRECISION = self._get_bool('HALF_PRECISION', False)
        self.DEVICE = self._get('DEVICE', 'cpu')
        
        # ==================== TRACKER ====================
        self.TRACK_ACTIVATION_THRESHOLD = self._get_float('TRACK_ACTIVATION_THRESHOLD', 0.25)
        self.MINIMUM_MATCHING_THRESHOLD = self._get_float('MINIMUM_MATCHING_THRESHOLD', 0.8)
        self.LOST_TRACK_BUFFER = self._get_int('LOST_TRACK_BUFFER', 30)
        self.MINIMUM_CONSECUTIVE_FRAMES = self._get_int('MINIMUM_CONSECUTIVE_FRAMES', 1)
        self.FRAME_RATE = self._get_int('FRAME_RATE', 30)
        
        # ==================== POLYGON ====================
        self.POLYGON_FILE = self._get('POLYGON_FILE', None)
        self.COLOR_INSIDE = self._get_color('COLOR_INSIDE', (0, 255, 0))
        self.COLOR_OUTSIDE = self._get_color('COLOR_OUTSIDE', (255, 0, 0))
        self.POLYGON_ALPHA = self._get_float('POLYGON_ALPHA', 0.2)
        
        # ==================== OUTPUT ====================
        self.SAVE_OUTPUT = self._get_bool('SAVE_OUTPUT', True)
        self.SHOW_PREVIEW = self._get_bool('SHOW_PREVIEW', True)
        
        # ==================== RESOLVE PATHS ====================
        self.VIDEO_PATH = self._resolve_path(self.VIDEO_PATH)
        self.MODEL_PATH = self._resolve_path(self.MODEL_PATH)
        
        # Create output directory
        self.OUTPUT_DIR = self.project_root / 'outputs'
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_VIDEO = self.OUTPUT_DIR / 'tracked_video.mp4'
        
        # Create other directories
        (self.project_root / 'models').mkdir(parents=True, exist_ok=True)
        (self.project_root / 'polygons').mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """
        Setup logging configuration based on LOG_LEVEL
        """
        log_level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        level = log_level_map.get(self.LOG_LEVEL.upper(), logging.INFO)
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Silence noisy third-party loggers
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        logging.getLogger('supervision').setLevel(logging.WARNING)
        
        print(f"✅ Logging configured: {self.LOG_LEVEL}")
    
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
    
    def _get_color(self, key, default):
        """
        Get color from environment in RGB format
        Format: R,G,B (e.g., 0,255,0) or (R,G,B)
        Returns: Tuple (R, G, B)
        """
        value = os.getenv(key)
        if not value:
            return default
        
        try:
            # Remove parentheses if present
            value = value.strip('()')
            # Split by comma and convert to int
            colors = [int(c.strip()) for c in value.split(',')]
            if len(colors) == 3:
                return (colors[0], colors[1], colors[2])
            return default
        except (ValueError, IndexError):
            print(f"⚠️  Invalid color format for {key}: {value}")
            return default
    
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
        print(f"LOG_LEVEL: {self.LOG_LEVEL}")
        print(f"VIDEO_PATH: {self.VIDEO_PATH}")
        print(f"MODEL_PATH: {self.MODEL_PATH}")
        print(f"CONF_THRESHOLD: {self.CONF_THRESHOLD}")
        print(f"IOU_THRESHOLD: {self.IOU_THRESHOLD}")
        print(f"MAX_DETECTIONS: {self.MAX_DETECTIONS}")
        print(f"HALF_PRECISION: {self.HALF_PRECISION}")
        print(f"DEVICE: {self.DEVICE}")
        print(f"RESIZE_VIDEO: {self.RESIZE_VIDEO}")
        if self.RESIZE_VIDEO:
            print(f"RESIZE: {self.RESIZE_WIDTH}x{self.RESIZE_HEIGHT}")
        print(f"SAVE_OUTPUT: {self.SAVE_OUTPUT}")
        print(f"SHOW_PREVIEW: {self.SHOW_PREVIEW}")
        print(f"POLYGON_FILE: {self.POLYGON_FILE or 'None'}")
        print(f"POLYGON_ALPHA: {self.POLYGON_ALPHA}")
        print(f"OUTPUT_VIDEO: {self.OUTPUT_VIDEO}")
        print("=" * 50)
        print()


# Create global settings instance
settings = Settings()


if __name__ == "__main__":
    settings.print_settings()