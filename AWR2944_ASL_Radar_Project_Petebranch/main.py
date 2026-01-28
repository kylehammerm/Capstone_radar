#!/usr/bin/env python3
"""
Main entry point for Radar Visualization System
Supports both GUI and CLI modes
"""
import sys
import argparse
from pathlib import Path
from radar.controller import RadarController
from radar.visualizer import RadarVisualizer
from PyQt6.QtWidgets import QApplication

def cli_mode(config_path: str, ip_address: str, port: int):
    """Command-line interface mode for data capture"""
    print(f"Starting CLI mode with config: {config_path}")
    controller = RadarController(config_path)
    
    # Configure radar
    if controller.configure():
        print("Radar configured successfully")
        
        # Start data capture
        if ip_address and port:
            print(f"Capturing data from {ip_address}:{port}")
            controller.start_capture(ip_address, port)
    else:
        print("Failed to configure radar")
        sys.exit(1)

def gui_mode(config_path: str):
    """GUI mode for interactive visualization"""
    app = QApplication(sys.argv)
    visualizer = RadarVisualizer(config_path)
    visualizer.show()
    sys.exit(app.exec())

def main():
    parser = argparse.ArgumentParser(description='Radar Data Visualization System')
    parser.add_argument('--mode', choices=['gui', 'cli'], default='gui',
                       help='Operation mode: gui (default) or cli')
    parser.add_argument('--config', type=str, default='config/radar_profile.cfg',
                       help='Path to radar configuration file')
    parser.add_argument('--ip', type=str, default='192.168.1.100',
                       help='IP address for data capture (CLI mode)')
    parser.add_argument('--port', type=int, default=4096,
                       help='Port for data capture (CLI mode)')
    
    args = parser.parse_args()
    
    # Ensure config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    if args.mode == 'cli':
        cli_mode(str(config_path), args.ip, args.port)
    else:
        gui_mode(str(config_path))

if __name__ == "__main__":
    main()