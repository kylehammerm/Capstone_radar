"""
Unit tests for DSP functions
"""
import numpy as np
import pytest
from radar.dsp import RangeDopplerProcessor, CFARDetector

def test_range_doppler_processor():
    """Test Range-Doppler processing"""
    processor = RangeDopplerProcessor(num_range_bins=128, num_doppler_bins=32)
    
    # Create test chirp data
    num_chirps = 64
    chirp_length = 256
    chirp_data = []
    
    for i in range(num_chirps):
        # Create linear chirp
        t = np.linspace(0, 1, chirp_length)
        chirp = np.exp(2j * np.pi * (10 * t + 5 * t**2))
        chirp_data.append(chirp + 0.1 * np.random.randn(chirp_length))
    
    # Process frame
    rd_map = processor.process_frame(chirp_data)
    
    # Check output shape
    assert rd_map.shape == (32, 128)  # Doppler bins x Range bins
    assert np.all(np.isfinite(rd_map))
    assert rd_map.min() >= 0  # Magnitude should be non-negative

def test_cfar_detector():
    """Test CFAR detection"""
    detector = CFARDetector(guard_cells=2, training_cells=4, pfa=1e-4)
    
    # Create test data with a single target
    rd_map = np.random.randn(64, 128)
    rd_map = np.abs(rd_map)
    
    # Add a strong target
    target_row, target_col = 32, 64
    rd_map[target_row, target_col] = 100.0
    
    # Run detection
    detection_map, threshold_map = detector.detect(rd_map)
    
    # Check detection
    assert detection_map[target_row, target_col] == True
    assert detection_map.sum() >= 1  # Should have at least the target
    
    # Check threshold map
    assert threshold_map.shape == rd_map.shape
    assert np.all(threshold_map >= 0)

def test_aoa_estimator():
    """Test Angle of Arrival estimation"""
    from radar.dsp import AoAEstimator
    
    estimator = AoAEstimator(num_antennas=8, angle_resolution=1.0)
    
    # Create test antenna data with a single angle
    num_samples = 256
    antenna_data = np.zeros((8, num_samples), dtype=np.complex64)
    
    # Simulate signal coming from 30 degrees
    for ant in range(8):
        phase_shift = 2 * np.pi * ant * np.sin(np.radians(30)) / 2
        antenna_data[ant] = np.exp(1j * (phase_shift + 
                                        np.linspace(0, 4*np.pi, num_samples)))
    
    # Add noise
    antenna_data += 0.1 * (np.random.randn(8, num_samples) + 
                          1j * np.random.randn(8, num_samples))
    
    # Estimate angles
    angle_spectrum = estimator.estimate(antenna_data)
    
    # Check output
    assert angle_spectrum.shape[0] == 360  # 1 degree resolution
    assert np.all(np.isfinite(angle_spectrum))
    
    # Peak should be around 30 degrees
    peak_idx = np.argmax(np.mean(angle_spectrum, axis=1))
    assert 25 <= peak_idx <= 35  # Allow some tolerance

if __name__ == "__main__":
    pytest.main([__file__, "-v"])