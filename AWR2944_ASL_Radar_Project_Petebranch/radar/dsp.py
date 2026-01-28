"""
Digital Signal Processing functions for radar data
"""
import numpy as np
from scipy import signal
from typing import Tuple, List

class RangeDopplerProcessor:
    """Process ADC data to Range-Doppler maps"""
    
    def __init__(self, num_range_bins: int = 256, num_doppler_bins: int = 64):
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        
    def process_chirp(self, adc_data: np.ndarray) -> np.ndarray:
        """Process single chirp: Windowing + FFT"""
        # Apply window function
        window = np.hanning(len(adc_data))
        windowed = adc_data * window
        
        # Range FFT
        range_fft = np.fft.fft(windowed, n=self.num_range_bins)
        return range_fft
    
    def process_frame(self, chirp_data: List[np.ndarray]) -> np.ndarray:
        """Process full frame: Range + Doppler processing"""
        num_chirps = len(chirp_data)
        
        # Range processing for each chirp
        range_profiles = np.zeros((num_chirps, self.num_range_bins), dtype=np.complex64)
        for i, chirp in enumerate(chirp_data):
            range_profiles[i] = self.process_chirp(chirp)
        
        # Doppler processing (FFT across chirps)
        doppler_fft = np.fft.fft(range_profiles, axis=0, n=self.num_doppler_bins)
        
        # Magnitude for visualization
        rd_map = np.abs(doppler_fft)
        return rd_map

class CFARDetector:
    """Constant False Alarm Rate detection"""
    
    def __init__(self, guard_cells: int = 2, training_cells: int = 10, pfa: float = 1e-6):
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.pfa = pfa
        
    def detect(self, rd_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply 2D CFAR detection"""
        rows, cols = rd_map.shape
        threshold_map = np.zeros_like(rd_map)
        detection_map = np.zeros_like(rd_map, dtype=bool)
        
        for i in range(rows):
            for j in range(cols):
                # Define training region
                row_start = max(0, i - self.guard_cells - self.training_cells)
                row_end = min(rows, i + self.guard_cells + self.training_cells + 1)
                col_start = max(0, j - self.guard_cells - self.training_cells)
                col_end = min(cols, j + self.guard_cells + self.training_cells + 1)
                
                # Extract training cells (exclude guard cells)
                training_region = rd_map[row_start:row_end, col_start:col_end]
                guard_mask = np.ones_like(training_region, dtype=bool)
                
                guard_row_start = max(0, i - self.guard_cells - row_start)
                guard_row_end = min(training_region.shape[0], 
                                  i + self.guard_cells + 1 - row_start)
                guard_col_start = max(0, j - self.guard_cells - col_start)
                guard_col_end = min(training_region.shape[1], 
                                  j + self.guard_cells + 1 - col_start)
                
                guard_mask[guard_row_start:guard_row_end, 
                          guard_col_start:guard_col_end] = False
                
                training_cells = training_region[guard_mask]
                
                if len(training_cells) > 0:
                    # Calculate threshold
                    noise_floor = np.mean(training_cells)
                    threshold = noise_floor * (-np.log(self.pfa))
                    threshold_map[i, j] = threshold
                    
                    # Compare with CUT
                    if rd_map[i, j] > threshold:
                        detection_map[i, j] = True
        
        return detection_map, threshold_map

class AoAEstimator:
    """Angle of Arrival estimation using FFT"""
    
    def __init__(self, num_antennas: int = 8, angle_resolution: float = 1.0):
        self.num_antennas = num_antennas
        self.angle_resolution = angle_resolution
        
    def estimate(self, antenna_data: np.ndarray) -> np.ndarray:
        """Estimate angles using FFT beamforming"""
        # Apply window to antenna data
        window = np.hanning(self.num_antennas)
        windowed = antenna_data * window[:, np.newaxis]
        
        # Angular FFT
        angle_fft = np.fft.fft(windowed, axis=0, n=360)
        angle_spectrum = np.abs(angle_fft)
        
        return angle_spectrum