"""
Live Micro-Doppler ASL Recognition Demo
========================================
- Connects to radar stream
- Runs CNN inference in real-time
- Shows predictions with confidence
"""

import os
import socket
import time
import threading
import numpy as np
from queue import Queue
import torch
import torch.nn as nn
from collections import deque, Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

# Add these imports
import serial
from realAWR2944v4 import (
    awr_config_and_start, dca_startup_commands, dca_config_only, dca_start,
    HOST_IP, HOST_CONFIG_PORT, DCA_IP, DCA_CONFIG_PORT,
    USE_UART_TO_START_RADAR, RADAR_CLI_PORT, RADAR_CLI_BAUD, RADAR_CFG_FILE,
    DO_NOT_START_DCA, DSPState, bytes_to_cube_real, compute_rd_and_profile,
    RANGE_CROP, MICRO_RANGE_CROP, parse_awr_cfg, UdpFrameAssembler
)

class LiveASLDemo:
    def __init__(self, model_path='checkpoints/best_model.pth'):
        # Load CNN model
        self.model, self.class_names = self.load_model(model_path)
        
        # Radar config
        self.cfg = self.init_radar()
        self.dsp = DSPState(self.cfg.chirps_per_frame, self.cfg.adc_samples)
        
        # Buffers
        self.frame_buffer = deque(maxlen=200)
        self.md_buffer = deque(maxlen=200)
        self.pred_history = deque(maxlen=10)
        self.conf_history = deque(maxlen=50)
        self.prob_history = deque(maxlen=50)
        
        # UDP streaming
        self.frame_q = Queue()
        self.assembler = None
        self.running = True
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        self.last_inference_time = 0
        
    def load_model(self, model_path):
        """Load trained PyTorch model"""
        # Define CNN architecture (same as training)
        class MicroDopplerCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Conv2d(1, 32, (5,7), padding=(2,3)),
                    nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 32, (5,7), padding=(2,3)),
                    nn.BatchNorm2d(32), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Dropout2d(0.1)
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(32, 64, (3,7), padding=(1,3)),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 64, (3,7), padding=(1,3)),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Dropout2d(0.2)
                )
                self.block3 = nn.Sequential(
                    nn.Conv2d(64, 128, (3,5), padding=(1,2)),
                    nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 128, (3,5), padding=(1,2)),
                    nn.BatchNorm2d(128), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4,8)), nn.Dropout2d(0.2)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128*4*8, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                return self.classifier(x)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        
        model = MicroDopplerCNN(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded model with classes: {class_names}")
        print(f"Model validation accuracy: {checkpoint.get('val_acc', 0):.1f}%")
        
        return model, class_names
    
    def send_radar_config(self):
        """Send configuration to radar and DCA1000"""
        print("Sending radar configuration...")
        
        # DCA1000 commands
        cmds = dca_startup_commands()
        cfg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cfg_sock.settimeout(1.0)
        cfg_sock.bind((HOST_IP, HOST_CONFIG_PORT))
        
        ok = True
        if not DO_NOT_START_DCA:
            ok &= dca_config_only(cfg_sock, cmds)
            print("  DCA1000 configured")
        
        if USE_UART_TO_START_RADAR:
            ok &= awr_config_and_start(RADAR_CLI_PORT, RADAR_CLI_BAUD, RADAR_CFG_FILE)
            print("  AWR2944 configured and started")
        
        if not DO_NOT_START_DCA:
            ok &= dca_start(cfg_sock, cmds)
            print("  DCA1000 started")
        
        cfg_sock.close()
        
        if ok:
            print("✓ Radar configuration successful")
        else:
            print("⚠ Radar configuration had warnings")
        
        return ok

    def init_radar(self):
        """Initialize radar configuration"""
        from pathlib import Path
        RADAR_CFG_FILE = r".\config\awr2944_cfg_updated.cfg"
        cfg_text = Path(RADAR_CFG_FILE).read_text(encoding="utf-8", errors="ignore")
        return parse_awr_cfg(cfg_text)
    
    def start_stream(self):
        """Start UDP stream"""
        self.assembler = UdpFrameAssembler(
            frame_bytes=self.cfg.frame_bytes,
            out_queue=self.frame_q,
            bind_ip="0.0.0.0",
            bind_port=4098,
            header_bytes=10
        )
        self.assembler.start()
        print("Radar stream started")
    
    def stop_stream(self):
        """Stop UDP stream"""
        if self.assembler:
            self.assembler.stop()
        self.running = False
    
    def infer(self, md_tensor):
        """Run inference on MD tensor"""
        # Preprocess
        tensor = (md_tensor + 10) / 50
        tensor = np.clip(tensor, 0, 1)
        tensor = tensor[np.newaxis, np.newaxis, :, :]
        
        # Inference
        with torch.no_grad():
            input_tensor = torch.from_numpy(tensor).float()
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, 1)
        
        return prediction.item(), confidence.item(), probs[0].numpy()
    
    def run(self):
        """Main loop"""
        # Send radar config FIRST
        if not self.send_radar_config():
            print("Failed to configure radar. Exiting.")
            return
        self.start_stream()
        
        # Wait for initial frames
        print("Waiting for radar data...")
        timeout = time.time() + 5  # 5 second timeout
        while len(self.md_buffer) < 30 and time.time() < timeout:
            try:
                frame = self.frame_q.get(timeout=0.1)
                cube = bytes_to_cube_real(frame, self.cfg)
                r0, r1 = RANGE_CROP
                m0, m1 = (r0, r1) if MICRO_RANGE_CROP is None else MICRO_RANGE_CROP
                rd_db, md_prof_db = compute_rd_and_profile(cube, self.dsp, m0, m1)
                self.md_buffer.append(md_prof_db)
                self.frame_count += 1
            except:
                continue
        
        if len(self.md_buffer) < 30:
            print("Failed to get radar data. Check connection.")
            return
        
        # Setup live plot
        plt.ion()
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 0.8, 0.8], height_ratios=[1, 1])
        
        # Micro-Doppler spectrogram
        md_ax = plt.subplot(gs[:, 0])
        md_tensor = np.stack(list(self.md_buffer)[-min(200, len(self.md_buffer)):], axis=1)
        md_img = md_ax.imshow(md_tensor, aspect='auto', cmap='viridis', 
                             vmin=-10, vmax=40, extent=[0, 200, 64, -64])
        md_ax.set_title('Live Micro-Doppler', fontsize=12, fontweight='bold')
        md_ax.set_xlabel('Time (frames)')
        md_ax.set_ylabel('Doppler (bin)')
        plt.colorbar(md_img, ax=md_ax, label='dB')
        
        # Current prediction
        pred_ax = plt.subplot(gs[0, 1])
        pred_ax.set_xlim(0, 1)
        pred_ax.set_xlabel('Confidence')
        pred_ax.set_title('Current Prediction', fontsize=12, fontweight='bold')
        pred_bars = pred_ax.barh(self.class_names, [0]*len(self.class_names), 
                                color='skyblue', edgecolor='navy')
        pred_ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Confidence history
        conf_ax = plt.subplot(gs[1, 1])
        conf_ax.set_title('Confidence History', fontsize=12, fontweight='bold')
        conf_ax.set_xlabel('Frame')
        conf_ax.set_ylabel('Confidence')
        conf_ax.set_ylim(0, 1)
        conf_ax.grid(True, alpha=0.3)
        conf_line, = conf_ax.plot([], [], 'g-', linewidth=2)
        
        # MD Spectrogram (small)
        spec_ax = plt.subplot(gs[0, 2])
        spec_img = spec_ax.imshow(md_tensor, aspect='auto', cmap='viridis',
                                 vmin=-10, vmax=40)
        spec_ax.set_title('MD Spectrogram', fontsize=12, fontweight='bold')
        spec_ax.set_xlabel('Time')
        spec_ax.set_ylabel('Doppler')
        
        # Stats
        stats_ax = plt.subplot(gs[1, 2])
        stats_ax.axis('off')
        stats_text = stats_ax.text(0.1, 0.5, '', transform=stats_ax.transAxes,
                                  fontsize=11, verticalalignment='center',
                                  family='monospace')
        
        plt.tight_layout()
        
        # Main loop
        print("\n=== Live ASL Recognition Demo ===")
        print("Make gestures in front of radar...")
        print("Press Ctrl+C to stop\n")
        
        frame_times = deque(maxlen=30)
        inference_counter = 0
        
        try:
            while self.running:
                # Get frame
                try:
                    frame = self.frame_q.get_nowait()
                except:
                    plt.pause(0.01)
                    continue
                
                # Process frame
                try:
                    cube = bytes_to_cube_real(frame, self.cfg)
                    r0, r1 = RANGE_CROP
                    m0, m1 = (r0, r1) if MICRO_RANGE_CROP is None else MICRO_RANGE_CROP
                    rd_db, md_prof_db = compute_rd_and_profile(cube, self.dsp, m0, m1)
                    
                    # Store
                    self.md_buffer.append(md_prof_db)
                    self.frame_count += 1
                    
                    # Update FPS
                    frame_times.append(time.time())
                    if len(frame_times) > 1:
                        self.fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                    
                    # Run inference every 5 frames (after buffer is full)
                    current_time = time.time()
                    if (len(self.md_buffer) >= 200 and 
                        self.frame_count % 5 == 0 and 
                        current_time - self.last_inference_time > 0.2):
                        
                        # Build tensor from last 200 frames
                        md_tensor = np.stack(list(self.md_buffer)[-200:], axis=1)
                        
                        # Infer
                        pred_class, confidence, probs = self.infer(md_tensor)
                        self.pred_history.append(pred_class)
                        self.conf_history.append(confidence)
                        self.prob_history.append(probs)
                        self.last_inference_time = current_time
                        inference_counter += 1
                        
                        # Smooth prediction
                        if len(self.pred_history) >= 5:
                            smoothed_pred = Counter(self.pred_history).most_common(1)[0][0]
                            smoothed_conf = np.mean([c for p, c in 
                                                   zip(self.pred_history, self.conf_history)
                                                   if p == smoothed_pred])
                        else:
                            smoothed_pred = pred_class
                            smoothed_conf = confidence
                        
                        # Clear console and print status
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print("="*60)
                        print("LIVE ASL RECOGNITION DEMO")
                        print("="*60)
                        print(f"\nCurrent Gesture: {self.class_names[smoothed_pred].upper()}")
                        print(f"Confidence: {smoothed_conf*100:.1f}%")
                        print(f"FPS: {self.fps:.1f}")
                        print(f"Inferences: {inference_counter}")
                        print("\nClass Probabilities:")
                        for i, name in enumerate(self.class_names):
                            bar = "█" * int(probs[i] * 50)
                            print(f"  {name:8s}: {probs[i]*100:5.1f}% {bar}")
                        print("\n" + "="*60)
                        
                        # Update plots
                        # Update MD image
                        md_img.set_data(md_tensor)
                        md_img.autoscale()
                        
                        # Update spectrogram
                        spec_img.set_data(md_tensor)
                        
                        # Update prediction bars
                        for bar, prob in zip(pred_bars, probs):
                            bar.set_width(prob)
                        
                        # Update confidence history
                        conf_line.set_data(range(len(self.conf_history)), 
                                         list(self.conf_history))
                        conf_ax.relim()
                        conf_ax.autoscale_view()
                        
                        # Update stats text
                        stats_text.set_text(
                            f"Model: {os.path.basename(self.model.__class__.__name__)}\n"
                            f"Classes: {len(self.class_names)}\n"
                            f"Buffer: {len(self.md_buffer)}/200\n"
                            f"Frames: {self.frame_count}\n"
                            f"FPS: {self.fps:.1f}\n\n"
                            f"Current:\n"
                            f"  {self.class_names[smoothed_pred]}\n"
                            f"  Conf: {smoothed_conf*100:.1f}%"
                        )
                        
                        plt.pause(0.01)
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\n\nDemo stopped by user")
        finally:
            self.stop_stream()
            plt.ioff()
            plt.close('all')
            print("Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description='Live ASL Recognition Demo')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    args = parser.parse_args()
    
    demo = LiveASLDemo(args.model)
    demo.run()

if __name__ == "__main__":
    main()