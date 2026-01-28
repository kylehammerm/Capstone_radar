"""
Radar controller for configuration and orchestration
"""
import serial
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class RadarConfig:
    """Radar configuration parameters"""
    frequency: float  # GHz
    bandwidth: float  # MHz
    chirp_time: float  # μs
    num_samples: int
    num_chirps: int
    num_antennas: int

class SerialConfig(BaseModel):
    """Serial port configuration"""
    port: str = "COM9"
    baudrate: int = 115200
    timeout: float = 2.0

class RadarController:
    """Main controller for radar operations"""
    
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else None
        self.radar_config: Optional[RadarConfig] = None
        self.serial_config = SerialConfig()
        self.serial_port: Optional[serial.Serial] = None
        
        if config_path:
            self.load_config(config_path)
            
    def load_config(self, config_path: str) -> bool:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                    # Parse YAML config
                    self.radar_config = RadarConfig(
                        frequency=config_data.get('frequency', 77.0),
                        bandwidth=config_data.get('bandwidth', 4000.0),
                        chirp_time=config_data.get('chirp_time', 50.0),
                        num_samples=config_data.get('num_samples', 256),
                        num_chirps=config_data.get('num_chirps', 64),
                        num_antennas=config_data.get('num_antennas', 8)
                    )
                elif config_path.endswith('.cfg'):
                    # Parse TI radar config file
                    self._parse_ti_config(f.read())
                    
            return True
        except Exception as e:
            print(f"Failed to load config: {e}")
            return False
            
    def _parse_ti_config(self, config_text: str):
        """Parse TI radar configuration file"""
        # Simple parser for demo - expand as needed
        lines = config_text.strip().split('\n')
        params = {}
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    params[parts[0]] = parts[1:]
        
        # Extract key parameters
        # This is a simplified example - real parser would be more complex
        if 'profileCfg' in params:
            profile = params['profileCfg']
            self.radar_config = RadarConfig(
                frequency=float(profile[1]),
                bandwidth=4000.0,  # Would calculate from slope
                chirp_time=float(profile[3]),
                num_samples=int(profile[10]),
                num_chirps=64,  # Default
                num_antennas=8   # Default
            )
            
    def connect_serial(self) -> bool:
        """Connect to radar via serial port"""
        try:
            self.serial_port = serial.Serial(
                port=self.serial_config.port,
                baudrate=self.serial_config.baudrate,
                timeout=self.serial_config.timeout
            )
            print(f"Connected to {self.serial_config.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to serial port: {e}")
            return False
            
    def disconnect_serial(self):
        """Disconnect from serial port"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            
    def send_command(self, command: str) -> str:
        """Send command to radar and get response"""
        if not self.serial_port or not self.serial_port.is_open:
            return "Not connected"
            
        try:
            self.serial_port.write(f"{command}\r\n".encode())
            response = self.serial_port.read_until(b'\r\n').decode().strip()
            return response
        except Exception as e:
            return f"Error: {e}"
            
    def configure(self) -> bool:
        """Configure radar with loaded configuration"""
        if not self.radar_config:
            print("No configuration loaded")
            return False
            
        if not self.connect_serial():
            return False
            
        try:
            # Send configuration commands
            # This is simplified - real implementation would send actual CLI commands
            commands = [
                "sensorStop",
                "flushCfg",
                "dfeDataOutputMode 1",
                f"profileCfg 0 {self.radar_config.frequency} 7 6 0 0 0 0 68 1 {self.radar_config.num_samples} 5500 0 0 48",
                "frameCfg 0 7 1 0 50 1 0",
                "sensorStart"
            ]
            
            for cmd in commands:
                print(f"Sending: {cmd}")
                response = self.send_command(cmd)
                print(f"Response: {response}")
                
            return True
        except Exception as e:
            print(f"Configuration failed: {e}")
            return False
        finally:
            self.disconnect_serial()
            
    def start_capture(self, ip_address: str, port: int):
        """Start data capture from radar"""
        print(f"Starting capture from {ip_address}:{port}")
        # Implementation would connect to Ethernet receiver
        
    def get_status(self) -> dict:
        """Get radar status"""
        return {
            "connected": self.serial_port is not None and self.serial_port.is_open,
            "configured": self.radar_config is not None,
            "frequency": self.radar_config.frequency if self.radar_config else 0,
            "samples": self.radar_config.num_samples if self.radar_config else 0
        }