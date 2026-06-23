"""
Core components for the people tracker
"""

from .tracker import PersonTracker
from .realtime_processor import RealtimeProcessor

__all__ = ['PersonTracker', 'RealtimeProcessor']