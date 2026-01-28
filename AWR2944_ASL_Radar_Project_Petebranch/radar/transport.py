"""
Transport layer for radar data acquisition
Handles Ethernet/UART communication
"""
import socket
import threading
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class RadarPacket:
    """Structure for parsed radar data packets"""
    frame_number: int
    chirp_index: int
    antenna_index: int
    adc_data: np.ndarray
    timestamp: float

class EthernetConfig(BaseModel):
    """Ethernet connection configuration"""
    ip_address: str = "192.168.1.100"
    data_port: int = 4096
    buffer_size: int = 65536
    timeout: float = 2.0

class UdpReceiver(threading.Thread):
    """Thread for receiving UDP radar data"""
    
    def __init__(self, config: EthernetConfig, callback: Callable[[RadarPacket], None]):
        super().__init__(daemon=True)
        self.config = config
        self.callback = callback
        self.running = False
        self.socket = None
        
    def run(self):
        """Main receiver thread loop"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(self.config.timeout)
        self.socket.bind(('', self.config.data_port))
        
        self.running = True
        print(f"UDP receiver started on port {self.config.data_port}")
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(self.config.buffer_size)
                packet = self._parse_packet(data)
                if packet and self.callback:
                    self.callback(packet)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error receiving data: {e}")
                
    def _parse_packet(self, raw_data: bytes) -> Optional[RadarPacket]:
        """Parse binary UDP packet into RadarPacket"""
        try:
            # Parse header (example format)
            header = raw_data[:16]
            frame_num = int.from_bytes(header[0:4], 'little')
            chirp_idx = int.from_bytes(header[4:6], 'little')
            ant_idx = int.from_bytes(header[6:8], 'little')
            
            # Parse ADC data (16-bit complex samples)
            adc_bytes = raw_data[16:]
            num_samples = len(adc_bytes) // 4  # 2 bytes I + 2 bytes Q
            
            if num_samples > 0:
                # Convert to numpy array of complex numbers
                dtype = np.dtype([('i', '<i2'), ('q', '<i2')])
                structured = np.frombuffer(adc_bytes, dtype=dtype, count=num_samples)
                adc_data = structured['i'] + 1j * structured['q']
                
                return RadarPacket(
                    frame_number=frame_num,
                    chirp_index=chirp_idx,
                    antenna_index=ant_idx,
                    adc_data=adc_data,
                    timestamp=np.datetime64('now')
                )
        except Exception as e:
            print(f"Packet parsing error: {e}")
        return None
    
    def stop(self):
        """Stop the receiver thread"""
        self.running = False
        if self.socket:
            self.socket.close()

class DCA1000Client:
    """Client for DCA1000EVM data capture"""
    
    def __init__(self, config: EthernetConfig):
        self.config = config
        self.receiver = None
        
    def start_capture(self, callback: Callable[[RadarPacket], None]):
        """Start data capture from DCA1000"""
        self.receiver = UdpReceiver(self.config, callback)
        self.receiver.start()
        
    def stop_capture(self):
        """Stop data capture"""
        if self.receiver:
            self.receiver.stop()
            self.receiver.join(timeout=2.0)
            
    def send_config(self, config_data: bytes):
        """Send configuration to DCA1000 (TCP)"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.config.ip_address, self.config.control_port))
                s.sendall(config_data)
                return True
        except Exception as e:
            print(f"Failed to send config: {e}")
            return False