"""
Simplified settings loader
Loads configuration from .env file
"""

import os
import json
import numpy as np
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
        self.MODEL_PATH = self._get('MODEL_PATH', 'models/yolo11n.pt')
        self.CONF_THRESHOLD = self._get_float('CONF_THRESHOLD', 0.4)
        self.DEVICE = self._get('DEVICE', 'cpu')
        self.TRACKER_TYPE = self._get('TRACKER_TYPE', 'bytetrack.yaml')
        self.SAVE_OUTPUT = self._get_bool('SAVE_OUTPUT', True)
        self.SHOW_PREVIEW = self._get_bool('SHOW_PREVIEW', True)
        
        # Video resize settings
        self.RESIZE_VIDEO = self._get_bool('RESIZE_VIDEO', False)
        self.RESIZE_WIDTH = self._get_int('RESIZE_WIDTH', 640)
        self.RESIZE_HEIGHT = self._get_int('RESIZE_HEIGHT', 480)
        
        # Polygon settings
        self.POLYGON_FILE = self._get('POLYGON_FILE', None)
        
        # Default colors (fallback if not in GeoJSON)
        # Store as RGB, convert to BGR when needed
        self.COLOR_INSIDE_RGB = self._get_color_rgb('COLOR_INSIDE', (0, 255, 0))
        self.COLOR_OUTSIDE_RGB = self._get_color_rgb('COLOR_OUTSIDE', (255, 0, 0))
        self.POLYGON_ALPHA = self._get_float('POLYGON_ALPHA', 0.3)
        
        # Load polygon from GeoJSON
        self.polygon_data = self._load_geojson()
        
        # Resolve paths
        self.VIDEO_PATH = self._resolve_path(self.VIDEO_PATH)
        self.MODEL_PATH = self._resolve_path(self.MODEL_PATH)
        
        # Create output directory
        self.OUTPUT_DIR = self.project_root / 'outputs'
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.OUTPUT_VIDEO = self.OUTPUT_DIR / 'tracked_video.mp4'
        
        # Create models directory if it doesn't exist
        models_dir = self.project_root / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create polygons directory if it doesn't exist
        polygons_dir = self.project_root / 'polygons'
        polygons_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def _get_color_rgb(self, key, default):
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
                return (colors[0], colors[1], colors[2])  # RGB
            return default
        except (ValueError, IndexError):
            print(f"⚠️  Invalid color format for {key}: {value}")
            return default
    
    def _rgb_to_bgr(self, rgb_color):
        """Convert RGB to BGR for OpenCV"""
        return (rgb_color[2], rgb_color[1], rgb_color[0])
    
    def _load_geojson(self):
        """
        Load polygon from GeoJSON file
        """
        if not self.POLYGON_FILE:
            return None
        
        polygon_path = self._resolve_path(self.POLYGON_FILE)
        
        if not polygon_path.exists():
            print(f"⚠️  Polygon file not found: {polygon_path}")
            return None
        
        print(f"📐 Loading polygon from: {polygon_path}")
        
        try:
            with open(polygon_path, 'r') as f:
                data = json.load(f)
            
            # Validate GeoJSON structure
            if data.get('type') != 'FeatureCollection':
                print(f"⚠️  Invalid GeoJSON: expected FeatureCollection")
                return None
            
            if not data.get('features') or len(data['features']) == 0:
                print(f"⚠️  Invalid GeoJSON: no features found")
                return None
            
            # Get first feature
            feature = data['features'][0]
            
            # Validate geometry
            if feature.get('geometry', {}).get('type') != 'Polygon':
                print(f"⚠️  Invalid GeoJSON: expected Polygon geometry")
                return None
            
            # Extract coordinates
            coordinates = feature['geometry']['coordinates']
            if not coordinates or len(coordinates) == 0:
                print(f"⚠️  Invalid GeoJSON: no coordinates found")
                return None
            
            # Get the first ring (exterior polygon)
            points = coordinates[0]
            
            # GeoJSON uses [longitude, latitude] -> we use [x, y]
            # Convert to list of tuples (x, y)
            polygon_points = [(p[0], p[1]) for p in points]
            
            # Validate minimum points (at least 3 for a triangle)
            if len(polygon_points) < 3:
                print(f"⚠️  Invalid polygon: need at least 3 points, got {len(polygon_points)}")
                return None
            
            # Get properties
            properties = feature.get('properties', {})
            
            # Get colors from properties or fallback to settings
            # GeoJSON stores as RGB
            color_inside_rgb = properties.get('color_inside')
            if color_inside_rgb and len(color_inside_rgb) == 3:
                color_inside_rgb = tuple(color_inside_rgb)
            else:
                color_inside_rgb = self.COLOR_INSIDE_RGB
            
            color_outside_rgb = properties.get('color_outside')
            if color_outside_rgb and len(color_outside_rgb) == 3:
                color_outside_rgb = tuple(color_outside_rgb)
            else:
                color_outside_rgb = self.COLOR_OUTSIDE_RGB
            
            # Convert to BGR for OpenCV
            color_inside_bgr = self._rgb_to_bgr(color_inside_rgb)
            color_outside_bgr = self._rgb_to_bgr(color_outside_rgb)
            
            # Get alpha from properties or fallback to settings
            alpha = properties.get('alpha', self.POLYGON_ALPHA)
            
            polygon_data = {
                'name': properties.get('name', polygon_path.stem),
                'description': properties.get('description', ''),
                'points': polygon_points,
                'color_inside_rgb': color_inside_rgb,
                'color_outside_rgb': color_outside_rgb,
                'color_inside_bgr': color_inside_bgr,
                'color_outside_bgr': color_outside_bgr,
                'alpha': alpha
            }
            
            print(f"✅ Loaded polygon: {polygon_data['name']} ({len(polygon_points)} points)")
            print(f"   Inside Color (RGB): {color_inside_rgb}")
            print(f"   Outside Color (RGB): {color_outside_rgb}")
            return polygon_data
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing GeoJSON: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading polygon: {e}")
            return None
    
    def _resolve_path(self, path):
        """Resolve path relative to project root"""
        if os.path.isabs(path):
            return Path(path)
        return self.project_root / path
    
    def get_polygon_points(self, frame_width, frame_height):
        """
        Convert normalized polygon points to pixel coordinates
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            List of (x,y) pixel coordinates or None if no polygon
        """
        if not self.polygon_data:
            return None
        
        points = self.polygon_data['points']
        pixel_points = []
        for x_norm, y_norm in points:
            x = int(x_norm * frame_width)
            y = int(y_norm * frame_height)
            pixel_points.append((x, y))
        
        return np.array(pixel_points, dtype=np.int32)
    
    def get_polygon_colors(self):
        """
        Get polygon colors as BGR for OpenCV
        
        Returns:
            tuple: (color_inside_bgr, color_outside_bgr)
        """
        if not self.polygon_data:
            return (
                self._rgb_to_bgr(self.COLOR_INSIDE_RGB),
                self._rgb_to_bgr(self.COLOR_OUTSIDE_RGB)
            )
        
        return (
            self.polygon_data.get('color_inside_bgr', self._rgb_to_bgr(self.COLOR_INSIDE_RGB)),
            self.polygon_data.get('color_outside_bgr', self._rgb_to_bgr(self.COLOR_OUTSIDE_RGB))
        )
    
    def get_polygon_colors_rgb(self):
        """
        Get polygon colors as RGB
        
        Returns:
            tuple: (color_inside_rgb, color_outside_rgb)
        """
        if not self.polygon_data:
            return self.COLOR_INSIDE_RGB, self.COLOR_OUTSIDE_RGB
        
        return (
            self.polygon_data.get('color_inside_rgb', self.COLOR_INSIDE_RGB),
            self.polygon_data.get('color_outside_rgb', self.COLOR_OUTSIDE_RGB)
        )
    
    def get_polygon_alpha(self):
        """Get polygon alpha"""
        if not self.polygon_data:
            return self.POLYGON_ALPHA
        
        return self.polygon_data.get('alpha', self.POLYGON_ALPHA)
    
    def get_polygon_name(self):
        """Get polygon name"""
        if not self.polygon_data:
            return None
        
        return self.polygon_data.get('name', 'Unnamed')
    
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
        
        # Print resize settings
        print(f"RESIZE_VIDEO: {self.RESIZE_VIDEO}")
        if self.RESIZE_VIDEO:
            print(f"RESIZE_WIDTH: {self.RESIZE_WIDTH}")
            print(f"RESIZE_HEIGHT: {self.RESIZE_HEIGHT}")
        
        print(f"SAVE_OUTPUT: {self.SAVE_OUTPUT}")
        print(f"SHOW_PREVIEW: {self.SHOW_PREVIEW}")
        
        if self.polygon_data:
            print(f"POLYGON_FILE: {self.POLYGON_FILE}")
            print(f"  Name: {self.polygon_data.get('name', 'Unknown')}")
            print(f"  Points: {len(self.polygon_data['points'])} points")
            inside_rgb, outside_rgb = self.get_polygon_colors_rgb()
            print(f"  Inside Color (RGB): {inside_rgb}")
            print(f"  Outside Color (RGB): {outside_rgb}")
            print(f"  Alpha: {self.get_polygon_alpha()}")
        else:
            print(f"POLYGON_FILE: None (no ROI)")
        
        print(f"OUTPUT_VIDEO: {self.OUTPUT_VIDEO}")
        print("=" * 50)
        print()


# Create global settings instance
settings = Settings()


if __name__ == "__main__":
    settings.print_settings()