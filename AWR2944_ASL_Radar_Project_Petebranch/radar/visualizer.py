"""
GUI Visualization components using PyQt6 and pyqtgraph
"""
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, 
                            QComboBox, QSpinBox, QGroupBox)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl

class PlotWidget2D(pg.GraphicsLayoutWidget):
    """2D plotting widget for Range-Doppler maps"""
    
    def __init__(self):
        super().__init__()
        self.plot = self.addPlot(title="Range-Doppler Map")
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.plot.setLabel('left', 'Doppler', units='bins')
        self.plot.setLabel('bottom', 'Range', units='bins')
        
        # Add color bar
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.addItem(self.hist)
        
    def update_data(self, rd_map: np.ndarray):
        """Update with new Range-Doppler data"""
        self.img.setImage(rd_map.T)
        self.img.setLevels([rd_map.min(), rd_map.max()])

class PlotWidget3D(gl.GLViewWidget):
    """3D point cloud visualization"""
    
    def __init__(self):
        super().__init__()
        self.point_cloud = gl.GLScatterPlotItem()
        self.addItem(self.point_cloud)
        
        # Set up 3D view
        self.setCameraPosition(distance=10)
        
    def update_points(self, points: np.ndarray, colors: np.ndarray = None):
        """Update 3D point cloud"""
        if colors is None:
            colors = np.ones((len(points), 4))  # Default white
            colors[:, 3] = 0.8  # Alpha
        
        self.point_cloud.setData(
            pos=points,
            color=colors,
            size=5,
            pxMode=True
        )

class RadarVisualizer(QMainWindow):
    """Main visualization window"""
    
    def __init__(self, config_path: str = None):
        super().__init__()
        self.config_path = config_path
        self.init_ui()
        self.setup_timer()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Radar Data Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Left panel - controls
        control_panel = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Capture")
        self.stop_btn = QPushButton("Stop Capture")
        self.stop_btn.setEnabled(False)
        
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Range-Doppler", "3D Point Cloud", "Spectrum"])
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setPrefix("FPS: ")
        
        control_layout.addWidget(QLabel("View Mode:"))
        control_layout.addWidget(self.view_combo)
        control_layout.addWidget(QLabel("Frame Rate:"))
        control_layout.addWidget(self.fps_spin)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        
        # Right panel - visualization
        vis_panel = QWidget()
        vis_layout = QVBoxLayout()
        
        self.plot_2d = PlotWidget2D()
        self.plot_3d = PlotWidget3D()
        
        self.stacked_widget = pg.GraphicsLayoutWidget()
        self.stacked_widget.addItem(self.plot_2d)
        
        vis_layout.addWidget(self.stacked_widget)
        
        # Status bar
        self.status_label = QLabel("Ready")
        vis_layout.addWidget(self.status_label)
        
        vis_panel.setLayout(vis_layout)
        
        # Add to main layout
        layout.addWidget(control_panel, 1)
        layout.addWidget(vis_panel, 4)
        
        # Connect signals
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)
        self.view_combo.currentTextChanged.connect(self.change_view)
        
    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        
    def start_capture(self):
        """Start data capture"""
        self.status_label.setText("Capturing...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(1000 // self.fps_spin.value())
        
    def stop_capture(self):
        """Stop data capture"""
        self.status_label.setText("Stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.timer.stop()
        
    def change_view(self, view_name: str):
        """Change visualization view"""
        if view_name == "3D Point Cloud":
            # For now, show placeholder
            points = np.random.randn(100, 3) * 2
            self.plot_3d.update_points(points)
            self.status_label.setText("3D Point Cloud View")
        else:
            self.status_label.setText(f"{view_name} View")
            
    def update_display(self):
        """Update visualization with new data"""
        # Generate dummy data for demo
        if self.view_combo.currentText() == "Range-Doppler":
            rd_map = np.random.randn(64, 256)
            rd_map = np.abs(np.fft.fft2(rd_map))
            self.plot_2d.update_data(rd_map)
        elif self.view_combo.currentText() == "3D Point Cloud":
            points = np.random.randn(50, 3) * 2
            colors = np.random.rand(50, 4)
            colors[:, 3] = 0.8
            self.plot_3d.update_points(points, colors)
            
    def closeEvent(self, event):
        """Handle window close"""
        self.stop_capture()
        super().closeEvent(event)