"""
Polygon loader for GeoJSON files
Handles loading, parsing, and color conversion
"""

import json
import numpy as np
from pathlib import Path


class PolygonLoader:
    """
    Load and parse polygon from GeoJSON file
    """
    
    def __init__(self, polygon_file=None):
        """
        Initialize polygon loader
        
        Args:
            polygon_file: Path to GeoJSON file (optional)
        """
        self.polygon_file = polygon_file
        self.data = None
        self.name = None
        self.description = None
        self.points = None  # Normalized points (0-1)
        self.color_inside_rgb = (0, 255, 0)  # Default green
        self.color_outside_rgb = (255, 0, 0)  # Default red
        self.color_inside_bgr = (0, 255, 0)   # Default green (BGR)
        self.color_outside_bgr = (0, 0, 255)  # Default red (BGR)
        self.alpha = 0.3
        
        if polygon_file:
            self.load(polygon_file)
    
    def load(self, polygon_file):
        """
        Load polygon from GeoJSON file
        
        Args:
            polygon_file: Path to GeoJSON file
            
        Returns:
            bool: True if loaded successfully
        """
        self.polygon_file = Path(polygon_file)
        
        if not self.polygon_file.exists():
            print(f"⚠️  Polygon file not found: {self.polygon_file}")
            return False
        
        print(f"📐 Loading polygon from: {self.polygon_file}")
        
        try:
            with open(self.polygon_file, 'r') as f:
                data = json.load(f)
            
            # Validate GeoJSON structure
            if data.get('type') != 'FeatureCollection':
                print(f"⚠️  Invalid GeoJSON: expected FeatureCollection")
                return False
            
            if not data.get('features') or len(data['features']) == 0:
                print(f"⚠️  Invalid GeoJSON: no features found")
                return False
            
            # Get first feature
            feature = data['features'][0]
            
            # Validate geometry
            if feature.get('geometry', {}).get('type') != 'Polygon':
                print(f"⚠️  Invalid GeoJSON: expected Polygon geometry")
                return False
            
            # Extract coordinates
            coordinates = feature['geometry']['coordinates']
            if not coordinates or len(coordinates) == 0:
                print(f"⚠️  Invalid GeoJSON: no coordinates found")
                return False
            
            # Get the first ring (exterior polygon)
            points = coordinates[0]
            
            # Convert to list of tuples (x, y)
            self.points = [(p[0], p[1]) for p in points]
            
            # Validate minimum points
            if len(self.points) < 3:
                print(f"⚠️  Invalid polygon: need at least 3 points, got {len(self.points)}")
                return False
            
            # Get properties
            properties = feature.get('properties', {})
            self.name = properties.get('name', self.polygon_file.stem)
            self.description = properties.get('description', '')
            
            # Get colors
            color_inside = properties.get('color_inside')
            if color_inside and len(color_inside) == 3:
                self.color_inside_rgb = tuple(color_inside)
                self.color_inside_bgr = (color_inside[2], color_inside[1], color_inside[0])
            
            color_outside = properties.get('color_outside')
            if color_outside and len(color_outside) == 3:
                self.color_outside_rgb = tuple(color_outside)
                self.color_outside_bgr = (color_outside[2], color_outside[1], color_outside[0])
            
            # Get alpha
            self.alpha = properties.get('alpha', 0.3)
            
            print(f"✅ Loaded polygon: {self.name} ({len(self.points)} points)")
            print(f"   Inside Color (RGB): {self.color_inside_rgb}")
            print(f"   Outside Color (RGB): {self.color_outside_rgb}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing GeoJSON: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading polygon: {e}")
            return False
    
    def get_pixel_points(self, frame_width, frame_height):
        """
        Convert normalized points to pixel coordinates
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            np.array: Array of pixel coordinates or None
        """
        if not self.points:
            return None
        
        pixel_points = []
        for x_norm, y_norm in self.points:
            x = int(x_norm * frame_width)
            y = int(y_norm * frame_height)
            pixel_points.append((x, y))
        
        return np.array(pixel_points, dtype=np.int32)
    
    def get_colors_bgr(self):
        """
        Get polygon colors in BGR format for OpenCV
        
        Returns:
            tuple: (color_inside_bgr, color_outside_bgr)
        """
        return self.color_inside_bgr, self.color_outside_bgr
    
    def get_colors_rgb(self):
        """
        Get polygon colors in RGB format
        
        Returns:
            tuple: (color_inside_rgb, color_outside_rgb)
        """
        return self.color_inside_rgb, self.color_outside_rgb
    
    def is_loaded(self):
        """Check if polygon is loaded"""
        return self.points is not None and len(self.points) >= 3
    
    def print_info(self):
        """Print polygon info"""
        if not self.is_loaded():
            print("No polygon loaded")
            return
        
        print("=" * 50)
        print("📐 POLYGON INFO")
        print("=" * 50)
        print(f"Name: {self.name}")
        print(f"Description: {self.description}")
        print(f"Points: {len(self.points)}")
        print(f"File: {self.polygon_file}")
        print(f"Inside Color (RGB): {self.color_inside_rgb}")
        print(f"Outside Color (RGB): {self.color_outside_rgb}")
        print(f"Alpha: {self.alpha}")
        print("=" * 50)


# Example usage
if __name__ == "__main__":
    # Test loading
    loader = PolygonLoader("polygons/default_polygon.geojson")
    loader.print_info()
    
    # Test pixel conversion
    if loader.is_loaded():
        pixels = loader.get_pixel_points(1920, 1080)
        print(f"Pixel points: {pixels[:5]}...")  # Show first 5 points