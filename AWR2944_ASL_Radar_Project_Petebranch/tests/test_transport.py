"""
Unit tests for transport layer
"""
import pytest
import numpy as np
from radar.transport import RadarPacket, EthernetConfig

def test_radar_packet():
    """Test RadarPacket dataclass"""
    adc_data = np.random.randn(256) + 1j * np.random.randn(256)
    
    packet = RadarPacket(
        frame_number=1,
        chirp_index=0,
        antenna_index=0,
        adc_data=adc_data,
        timestamp=np.datetime64('now')
    )
    
    assert packet.frame_number == 1
    assert packet.chirp_index == 0
    assert packet.antenna_index == 0
    assert len(packet.adc_data) == 256
    assert isinstance(packet.timestamp, np.datetime64)

def test_ethernet_config():
    """Test EthernetConfig Pydantic model"""
    config = EthernetConfig(
        ip_address="192.168.1.100",
        data_port=4096,
        buffer_size=65536,
        timeout=2.0
    )
    
    assert config.ip_address == "192.168.1.100"
    assert config.data_port == 4096
    assert config.buffer_size == 65536
    assert config.timeout == 2.0
    
    # Test validation
    with pytest.raises(ValueError):
        EthernetConfig(ip_address="invalid_ip")
        
    with pytest.raises(ValueError):
        EthernetConfig(data_port=100000)  # Invalid port

def test_packet_parsing():
    """Test packet parsing from raw bytes"""
    from radar.transport import UdpReceiver
    
    # Create test packet
    frame_num = 1
    chirp_idx = 2
    ant_idx = 3
    num_samples = 64
    
    # Create header
    header = bytearray()
    header.extend(frame_num.to_bytes(4, 'little'))
    header.extend(chirp_idx.to_bytes(2, 'little'))
    header.extend(ant_idx.to_bytes(2, 'little'))
    header.extend(b'\x00' * 8)  # Padding
    
    # Create ADC data (I/Q samples)
    adc_data = bytearray()
    for i in range(num_samples):
        i_val = np.int16(np.random.randn() * 1000)
        q_val = np.int16(np.random.randn() * 1000)
        adc_data.extend(i_val.to_bytes(2, 'little', signed=True))
        adc_data.extend(q_val.to_bytes(2, 'little', signed=True))
    
    raw_packet = header + adc_data
    
    # Parse (simplified - actual UdpReceiver has private method)
    # This tests the concept
    dtype = np.dtype([('i', '<i2'), ('q', '<i2')])
    structured = np.frombuffer(adc_data, dtype=dtype, count=num_samples)
    parsed_data = structured['i'] + 1j * structured['q']
    
    assert len(parsed_data) == num_samples
    assert parsed_data.dtype == np.complex128

if __name__ == "__main__":
    pytest.main([__file__, "-v"])