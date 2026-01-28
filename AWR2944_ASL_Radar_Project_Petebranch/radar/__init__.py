"""
Radar Data Processing and Visualization Package
"""
__version__ = "1.0.0"
__author__ = "Radar Project Team"

from .transport import DCA1000Client, UdpReceiver
from .dsp import RangeDopplerProcessor, CFARDetector, AoAEstimator
from .visualizer import RadarVisualizer, PlotWidget3D
from .controller import RadarController

__all__ = [
    'DCA1000Client',
    'UdpReceiver',
    'RangeDopplerProcessor',
    'CFARDetector',
    'AoAEstimator',
    'RadarVisualizer',
    'PlotWidget3D',
    'RadarController'
]