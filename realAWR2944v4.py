"""
realtime_rdi_ra_md_awr2944_v4.py
=====================================

Goal:
- Start and record signs using labeled folders/files
- Make Micro-Doppler *synchronous* with RD (no visible lag)

Key changes vs v3:
1) The real fix for MD sharpness
2) configured for GPT ASL signs
"""

from __future__ import annotations

import os
from datetime import datetime
import socket
import time
import threading as th
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
import re

import numpy as np

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import serial

# ==============================
# NETWORK (DCA1000)
# ==============================
HOST_IP = "192.168.33.30"
HOST_CONFIG_PORT = 4096
HOST_DATA_BIND_IP = "0.0.0.0"
HOST_DATA_PORT = 4098

DCA_IP = "192.168.33.180"
DCA_CONFIG_PORT = 4096

DCA_DATA_HEADER_BYTES = 10
UDP_RCVBUF = 8 * 1024 * 1024
UDP_READ_SIZE = 2 * 1024 * 1024
SOCKET_TIMEOUT_S = 1.0

# ==============================
# UART (AWR2944 CLI)
# ==============================
USE_UART_TO_START_RADAR = True
RADAR_CLI_PORT = "COM12" #Note: change to your COM port
RADAR_CLI_BAUD = 115200
RADAR_CFG_FILE = r".\config\awr2944_ASL.cfg"  #r".\awr2944_cfg_updated.cfg"

PROMPT = b"mmwDemo:/>"
CHAR_DELAY_S = 0.0015
PROMPT_TIMEOUT_S = 3.0
DO_NOT_START_DCA = False

# ==============================
# PERFORMANCE / QUALITY KNOBS
# ==============================
ENABLE_MTI = True
MTI_MODE = "mean"  # "mean" or "diff"

# FFT sizes (main speed lever)
DOPPLER_FFT = 128
RANGE_FFT = 256
ANGLE_FFT = 64

# Range crop for RD/RA (bins)
RANGE_CROP = (5, 110)

# Micro-Doppler range window (bins). None -> use RANGE_CROP
MICRO_RANGE_CROP = None  # e.g. (30, 70)

USE_DB = True
DB_FLOOR_RD = -10.0
DB_FLOOR_RA = -10.0
DB_FLOOR_MD = -10.0

RX_COMBINE = "sum"

ENABLE_DOPPLER_NOTCH = True
DOPPLER_NOTCH_BINS = 1
ENABLE_RX_PHASE_NORM = False

# Temporal smoothing (separate for MD to reduce lag)
EMA_ALPHA_RD = 0.65
EMA_ALPHA_RA = 0.70
EMA_ALPHA_MD = 0.55   # LOWER = less lag

# Spatial smoothing (lighter overall)
SMOOTH_KERNEL_RD = 3
SMOOTH_KERNEL_RA = 3
SMOOTH_KERNEL_MD = 1  # <-- key for speed + less lag

AUTO_LEVELS = True
LEVEL_PCT_LOW_RD = 20.0
LEVEL_PCT_HIGH_RD = 99.7
LEVEL_PCT_LOW_RA = 15.0
LEVEL_PCT_HIGH_RA = 99.7
LEVEL_PCT_LOW_MD = 15.0
LEVEL_PCT_HIGH_MD = 99.7
LEVEL_EMA = 0.80

# RA throttling (big FPS lever)
ANGLE_UPDATE_EVERY = 2      # compute RA every N frames
RA_RANGE_DECIM = 2          # decimate range bins for RA

# Micro-Doppler history (time axis)
MICRO_HISTORY = 200

# GUI update rate (can be faster now)
PLOT_UPDATE_MS = 25   # ~40 Hz target

# ==============================
# PyQtGraph look
# ==============================
pg.setConfigOption("background", "k")
pg.setConfigOption("foreground", "w")

def jet_like_lut(n: int = 256) -> np.ndarray:
    pos = np.array([0.0, 0.35, 0.66, 0.89, 1.0], dtype=np.float32)
    col = np.array(
        [
            [0, 0, 128],
            [0, 0, 255],
            [0, 255, 255],
            [255, 255, 0],
            [128, 0, 0],
        ],
        dtype=np.float32,
    )
    x = np.linspace(0, 1, n, dtype=np.float32)
    out = np.zeros((n, 3), dtype=np.float32)
    for ch in range(3):
        out[:, ch] = np.interp(x, pos, col[:, ch])
    return np.clip(out, 0, 255).astype(np.uint8)

LUT = jet_like_lut(256)

def box_blur_2d(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    if k % 2 == 0:
        k += 1
    r = k // 2
    ar = np.pad(a, ((0, 0), (r, r)), mode="edge")
    c = np.cumsum(ar, axis=1)
    a1 = (c[:, k:] - c[:, :-k]) / float(k)
    ac = np.pad(a1, ((r, r), (0, 0)), mode="edge")
    c2 = np.cumsum(ac, axis=0)
    a2 = (c2[k:, :] - c2[:-k, :]) / float(k)
    return a2.astype(a.dtype, copy=False)

def levels_from_percentiles(arr: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[float, float]:
    lo, hi = np.percentile(arr, (lo_pct, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)

# ==============================
# DCA1000 packets
# ==============================
def make_cmd(code_hex: str, data: bytes = b"") -> bytes:
    header = (0xA55A).to_bytes(2, "little")
    footer = (0xEEAA).to_bytes(2, "little")
    code = int(code_hex, 16).to_bytes(2, "little")
    length = len(data).to_bytes(2, "little")
    return header + code + length + data + footer

def dca_startup_commands() -> dict[str, bytes]:
    return {
        "CONNECT": make_cmd("09"),
        "READ_FPGA": make_cmd("0E"),
        "CONFIG_FPGA": make_cmd("03", (0x01020102031E).to_bytes(6, "big")),
        "CONFIG_PACKET": make_cmd("0B", (0xC005350C0000).to_bytes(6, "big")),
        "START": make_cmd("05"),
        "STOP": make_cmd("06"),
    }

def dca_send_cmd(cfg_sock: socket.socket, cmds: dict[str, bytes], name: str) -> bool:
    cfg_sock.sendto(cmds[name], (DCA_IP, DCA_CONFIG_PORT))
    try:
        cfg_sock.recvfrom(2048)
        return True
    except socket.timeout:
        print(f"[DCA] No ACK for {name}")
        return False

def dca_config_only(cfg_sock: socket.socket, cmds: dict[str, bytes]) -> bool:
    ok = True
    ok &= dca_send_cmd(cfg_sock, cmds, "CONNECT")
    ok &= dca_send_cmd(cfg_sock, cmds, "READ_FPGA")
    ok &= dca_send_cmd(cfg_sock, cmds, "CONFIG_FPGA")
    ok &= dca_send_cmd(cfg_sock, cmds, "CONFIG_PACKET")
    return ok

def dca_start(cfg_sock: socket.socket, cmds: dict[str, bytes]) -> bool:
    return dca_send_cmd(cfg_sock, cmds, "START")


def dca_stop(cfg_sock: socket.socket, cmds: dict[str, bytes]) -> bool:
    """Best-effort stop capture on DCA1000."""
    # Many DCA1000 command sets use 0x06 as STOP.
    return dca_send_cmd(cfg_sock, cmds, "STOP")

# ==============================
# UART helpers
# ==============================
def _drain_read(s: serial.Serial, seconds: float = 0.15) -> bytes:
    end = time.time() + seconds
    out = b""
    while time.time() < end:
        chunk = s.read(4096)
        if chunk:
            out += chunk
        else:
            time.sleep(0.01)
    return out

def _read_until_prompt(s: serial.Serial, timeout_s: float) -> bytes:
    end = time.time() + timeout_s
    buf = b""
    while time.time() < end:
        chunk = s.read(4096)
        if chunk:
            buf += chunk
            if PROMPT in buf:
                break
        else:
            time.sleep(0.01)
    return buf

def uart_send_line_prompted(s: serial.Serial, line: str) -> bytes:
    _drain_read(s, 0.05)
    payload = (line + "\r\n").encode("ascii", errors="ignore")
    for b in payload:
        s.write(bytes([b]))
        s.flush()
        time.sleep(CHAR_DELAY_S)
    return _read_until_prompt(s, timeout_s=PROMPT_TIMEOUT_S)

def awr_stop(cli_port: str, baud: int) -> bool:
    """Best-effort: send sensorStop so the radar firmware stops streaming."""
    try:
        with serial.Serial(cli_port, baudrate=baud, timeout=0.3) as s:
            time.sleep(0.2)
            s.reset_input_buffer()
            s.reset_output_buffer()
            resp = uart_send_line_prompted(s, "sensorStop")
            txt = resp.decode(errors="ignore")
            if "Error" in txt:
                print("[AWR] sensorStop returned error:", txt)
                return False
            print("[AWR] sensorStop sent.")
            return True
    except Exception as e:
        print(f"[AWR] stop failed: {e}")
        return False


def awr_config_and_start(cli_port: str, baud: int, cfg_path: str) -> bool:
    cfg_abs = str(Path(cfg_path).resolve())
    if not Path(cfg_abs).exists():
        print(f"[FATAL] AWR cfg file not found: {cfg_abs}")
        return False
    ok = True
    with serial.Serial(cli_port, baudrate=baud, timeout=0.2) as s:
        time.sleep(0.5)
        s.reset_input_buffer()
        s.reset_output_buffer()
        uart_send_line_prompted(s, "")
        uart_send_line_prompted(s, "sensorStop")
        with open(cfg_abs, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if (not line) or line.startswith("%") or line.startswith("#"):
                    continue
                if line.lower().startswith("sensorstart"):
                    continue
                resp = uart_send_line_prompted(s, line)
                t = resp.decode(errors="ignore")
                if ("Error" in t) or ("not recognized" in t) or ("Invalid usage" in t) or ("Unknown command" in t):
                    ok = False
        resp = uart_send_line_prompted(s, "sensorStart")
        if "Error" in resp.decode(errors="ignore"):
            ok = False
    return ok

# ==============================
# Parse cfg
# ==============================
@dataclass(frozen=True)
class RadarFrameConfig:
    adc_samples: int
    chirps_per_frame: int
    rx_channels: int

    @property
    def frame_bytes(self) -> int:
        return self.chirps_per_frame * self.adc_samples * self.rx_channels * 2

def _popcount(x: int) -> int:
    return bin(x & 0xFFFFFFFF).count("1")

def parse_awr_cfg(cfg_text: str) -> RadarFrameConfig:
    adc_samples = None
    chirps_per_frame = None
    rx_channels = None
    for raw in cfg_text.splitlines():
        line = raw.strip()
        if (not line) or line.startswith("%") or line.startswith("#"):
            continue
        if line.startswith("profileCfg"):
            parts = re.split(r"\s+", line)
            if len(parts) >= 11:
                adc_samples = int(float(parts[10]))
        if line.startswith("frameCfg"):
            parts = re.split(r"\s+", line)
            if len(parts) >= 4:
                chirps_per_frame = int(float(parts[3]))
        if line.startswith("channelCfg"):
            parts = re.split(r"\s+", line)
            if len(parts) >= 3:
                rx_en = int(parts[1])
                rx_channels = _popcount(rx_en & 0xF)
    if adc_samples is None or chirps_per_frame is None or rx_channels is None:
        raise ValueError("Could not parse adc_samples / chirps_per_frame / rx_channels from cfg.")
    return RadarFrameConfig(adc_samples=adc_samples, chirps_per_frame=chirps_per_frame, rx_channels=rx_channels)

# ==============================
# UDP frame receiver
# ==============================
class UdpFrameAssembler(th.Thread):
    def __init__(self, frame_bytes: int, out_queue: Queue[bytes], bind_ip: str, bind_port: int, header_bytes: int = 10):
        super().__init__(daemon=True)
        self.frame_bytes = frame_bytes
        self.out_queue = out_queue
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.header_bytes = header_bytes
        self._stop = th.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_ip, self.bind_port))
        sock.settimeout(SOCKET_TIMEOUT_S)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UDP_RCVBUF)
        buf = bytearray()
        while not self._stop.is_set():
            try:
                packet, _ = sock.recvfrom(UDP_READ_SIZE)
            except socket.timeout:
                continue
            payload = packet[self.header_bytes:] if len(packet) > self.header_bytes else b""
            if payload:
                buf.extend(payload)
            while len(buf) >= self.frame_bytes:
                frame = bytes(buf[: self.frame_bytes])
                del buf[: self.frame_bytes]
                if self.out_queue.qsize() > 1:
                    try:
                        self.out_queue.get_nowait()
                    except Empty:
                        pass
                self.out_queue.put(frame)
        sock.close()

# ==============================
# DSP (precompute windows)
# ==============================
class DSPState:
    def __init__(self, chirps: int, samples: int):
        self.w_r = np.hanning(samples).astype(np.float32)
        self.w_d = np.hanning(chirps).astype(np.float32)

    def apply_windows(self, cube: np.ndarray) -> np.ndarray:
        return cube * self.w_d[:, None, None] * self.w_r[None, :, None]

def bytes_to_cube_real(frame_bytes: bytes, cfg: RadarFrameConfig) -> np.ndarray:
    raw = np.frombuffer(frame_bytes, dtype=np.dtype(np.int16).newbyteorder("<"))
    expected = cfg.chirps_per_frame * cfg.adc_samples * cfg.rx_channels
    if raw.size != expected:
        raise ValueError(f"Frame int16 count mismatch: got {raw.size}, expected {expected}")
    return raw.reshape(cfg.chirps_per_frame, cfg.adc_samples, cfg.rx_channels).astype(np.float32, copy=False)

def _mti(x: np.ndarray) -> np.ndarray:
    if not ENABLE_MTI:
        return x
    if MTI_MODE == "diff":
        return np.diff(x, axis=0)
    return x - x.mean(axis=0, keepdims=True)

def compute_rd_and_profile(cube: np.ndarray, dsp: DSPState, r0: int, r1: int) -> tuple[np.ndarray, np.ndarray]:
    x = dsp.apply_windows(cube)
    x = _mti(x)

    Xr = np.fft.fft(x, n=RANGE_FFT, axis=1).astype(np.complex64, copy=False)
    Xrd = np.fft.fft(Xr, n=DOPPLER_FFT, axis=0)
    Xrd = np.fft.fftshift(Xrd, axes=0)

    mag = np.abs(Xrd).astype(np.float32, copy=False)

    # ===============================
    # RANGE–DOPPLER (for display)
    # ===============================
    rd = mag.max(axis=2) if RX_COMBINE == "max" else mag.sum(axis=2)

    # ===============================
    # ANGLE FFT FOR GATING (from Xr, not rd)
    # ===============================
    Xr_ra = np.fft.fftshift(
        np.fft.fft(Xr[:, r0:r1, :], n=ANGLE_FFT, axis=2),
        axes=2
    )

    ra_mag = np.abs(Xr_ra).mean(axis=0)  # [range, angle]

    angle_energy = ra_mag.sum(axis=0)
    theta_center = int(np.argmax(angle_energy))

    ANGLE_WIN = 3
    theta_start = max(0, theta_center - ANGLE_WIN)
    theta_end   = min(ANGLE_FFT, theta_center + ANGLE_WIN + 1)

    # ===============================
    # ANGLE-GATED MICRO-DOPPLER
    # ===============================
    md_cube = mag[:, r0:r1, :]  # Doppler × Range × RX

    md_angle = np.fft.fftshift(
        np.fft.fft(md_cube, n=ANGLE_FFT, axis=2),
        axes=2
    )

    md_gated = md_angle[:, :, theta_start:theta_end]

    # ===============================
    # APPLY DOPPLER NOTCH (HERE)
    # ===============================
    if ENABLE_DOPPLER_NOTCH and DOPPLER_NOTCH_BINS > 0:
        c = md_gated.shape[0] // 2
        b = int(DOPPLER_NOTCH_BINS)
        md_gated[max(0, c - b): min(md_gated.shape[0], c + b + 1), :, :] = 0.0

    # ===============================
    # FINAL MD PROFILE
    # ===============================
    prof = np.abs(md_gated).sum(axis=(1, 2))


    if USE_DB:
        rd_db = 20.0 * np.log10(rd + 1e-6)
        rd_db = np.maximum(rd_db, DB_FLOOR_RD)
        prof_db = 20.0 * np.log10(prof + 1e-6)
        prof_db = np.maximum(prof_db, DB_FLOOR_MD)
    else:
        rd_db = rd
        prof_db = prof

    return rd_db.astype(np.float32, copy=False), prof_db.astype(np.float32, copy=False)

def compute_ra(cube: np.ndarray, dsp: DSPState) -> np.ndarray:
    x = dsp.apply_windows(cube)
    x = _mti(x)

    Xr = np.fft.fft(x, n=RANGE_FFT, axis=1).astype(np.complex64, copy=False)

    if ENABLE_RX_PHASE_NORM and Xr.shape[2] > 1:
        ref = Xr[:, :, 0:1]
        ph = ref / (np.abs(ref) + 1e-9)
        Xr = Xr * np.conj(ph)

    Xr_ra = Xr[:, ::RA_RANGE_DECIM, :]
    RA_ch = np.fft.fftshift(np.fft.fft(Xr_ra, n=ANGLE_FFT, axis=2), axes=2)
    ra = np.mean(np.abs(RA_ch), axis=0).T

    if USE_DB:
        ra = 20.0 * np.log10(ra + 1e-6)
        ra = np.maximum(ra, DB_FLOOR_RA)
    return ra.astype(np.float32, copy=False)

# ==============================
# GUI
# ==============================
class RadarUI(QtWidgets.QMainWindow):
    def __init__(self, cfg: RadarFrameConfig):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Real-Time Radar")

        self.frame_q: Queue[bytes] = Queue()
        self.assembler: UdpFrameAssembler | None = None
        self.dsp = DSPState(cfg.chirps_per_frame, cfg.adc_samples)

        # EMA states
        self._rd_ema: np.ndarray | None = None
        self._ra_ema: np.ndarray | None = None
        self._md_ema: np.ndarray | None = None

        # Level EMA states
        self._rd_levels: tuple[float, float] | None = None
        self._ra_levels: tuple[float, float] | None = None
        self._md_levels: tuple[float, float] | None = None

        self._frame_count = 0
        self._shutting_down = False
        self._last_ra: np.ndarray | None = None

        # Micro-doppler buffer (doppler x time)
        self._md_hist = np.full((DOPPLER_FFT, MICRO_HISTORY), DB_FLOOR_MD, dtype=np.float32)

        # ==============================
        # Recording state
        # ==============================
        self.recording_enabled = False
        self.recording_fh = None
        self.recording_dir = Path("recordings")
        self.recording_dir.mkdir(exist_ok=True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # ====== TOP ROW: RD + RA ======
        titles_top = QtWidgets.QHBoxLayout()
        self.lbl_rd = QtWidgets.QLabel("Range-Doppler Image")
        self.lbl_ra = QtWidgets.QLabel("Range-Angle Image")
        for lbl in (self.lbl_rd, self.lbl_ra):
            f = lbl.font()
            f.setPointSize(14)
            f.setBold(True)
            lbl.setFont(f)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        titles_top.addWidget(self.lbl_rd)
        titles_top.addWidget(self.lbl_ra)
        outer.addLayout(titles_top)

        plots_top = QtWidgets.QHBoxLayout()
        outer.addLayout(plots_top, stretch=2)

        self.rd_glw = pg.GraphicsLayoutWidget()
        self.ra_glw = pg.GraphicsLayoutWidget()
        plots_top.addWidget(self.rd_glw)
        plots_top.addWidget(self.ra_glw)

        self.rd_plot = self.rd_glw.addPlot()
        self.ra_plot = self.ra_glw.addPlot()
        for p in (self.rd_plot, self.ra_plot):
            p.setMenuEnabled(False)
            p.hideButtons()
            p.getViewBox().setBackgroundColor("k")

        self.rd_img = pg.ImageItem()
        self.ra_img = pg.ImageItem()
        self.rd_img.setAutoDownsample(False)
        self.ra_img.setAutoDownsample(False)
        self.rd_plot.addItem(self.rd_img)
        self.ra_plot.addItem(self.ra_img)
        self.rd_img.setLookupTable(LUT)
        self.ra_img.setLookupTable(LUT)

        # ====== BOTTOM ROW: MICRO-DOPPLER ======
        self.lbl_md = QtWidgets.QLabel("Micro-Doppler (Doppler vs Time)")
        f = self.lbl_md.font()
        f.setPointSize(13)
        f.setBold(True)
        self.lbl_md.setFont(f)
        self.lbl_md.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.lbl_md)

        self.md_glw = pg.GraphicsLayoutWidget()
        outer.addWidget(self.md_glw, stretch=1)

        self.md_plot = self.md_glw.addPlot()
        self.md_plot.setMenuEnabled(False)
        self.md_plot.hideButtons()
        self.md_plot.getViewBox().setBackgroundColor("k")
        self.md_img = pg.ImageItem()
        self.md_img.setAutoDownsample(False)
        self.md_plot.addItem(self.md_img)
        self.md_img.setLookupTable(LUT)
        self.md_plot.setAspectLocked(False)
        self.md_plot.setLabel('bottom', 'Time →')
        self.md_plot.setLabel('left', 'Doppler bin')



        # ====== Buttons ======
        btn_row = QtWidgets.QHBoxLayout()
        outer.addLayout(btn_row)
        self.btn_send = QtWidgets.QPushButton("Send Radar Config")
        
        # SINGLE toggle recording button
        self.btn_record_toggle = QtWidgets.QPushButton("Start ▶")
        self.btn_record_toggle.clicked.connect(self.toggle_recording)
        
        self.btn_exit = QtWidgets.QPushButton("Exit")
        self.btn_send.clicked.connect(self.on_send_config)
        self.btn_exit.clicked.connect(self.shutdown)

        btn_row.addWidget(self.btn_send, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        btn_row.addWidget(self.btn_record_toggle, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_exit, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        # ====== Status + timer ======
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        self._fps_frames = 0
        self._fps_t0 = time.time()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(PLOT_UPDATE_MS)

    def start_udp(self):
        if self.assembler is None:
            self.assembler = UdpFrameAssembler(
                frame_bytes=self.cfg.frame_bytes,
                out_queue=self.frame_q,
                bind_ip=HOST_DATA_BIND_IP,
                bind_port=HOST_DATA_PORT,
                header_bytes=DCA_DATA_HEADER_BYTES,
            )
            self.assembler.start()

    def on_send_config(self):
        cmds = dca_startup_commands()
        cfg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cfg_sock.bind((HOST_IP, HOST_CONFIG_PORT))
        cfg_sock.settimeout(SOCKET_TIMEOUT_S)

        ok = True
        if not DO_NOT_START_DCA:
            ok &= dca_config_only(cfg_sock, cmds)
        if USE_UART_TO_START_RADAR:
            ok &= awr_config_and_start(RADAR_CLI_PORT, RADAR_CLI_BAUD, RADAR_CFG_FILE)
        if not DO_NOT_START_DCA:
            ok &= dca_start(cfg_sock, cmds)

        cfg_sock.close()
        self.start_udp()
        self.status.showMessage("Config sent / streaming started" if ok else "Config sent with WARNINGS (check console)")

    def get_label_from_user(self) -> str | None:
        """Show a dialog to get a label from the user."""
        label, ok = QtWidgets.QInputDialog.getText(
            self,
            "Recording Label",
            "Enter a label for this recording (optional):",
            QtWidgets.QLineEdit.EchoMode.Normal,
            ""
        )
        if ok:
            # Clean the label: remove invalid characters, replace spaces with underscores
            label = label.strip()
            if label:
                # Remove any characters that are not alphanumeric, underscore, or hyphen
                label = re.sub(r'[^\w\s-]', '', label)
                # Replace spaces with underscores
                label = re.sub(r'\s+', '_', label)
                return label
        return None

    def toggle_recording(self):
        """Toggle recording on/off with a single button"""
        if not self.recording_enabled:
            # Get label from user
            label = self.get_label_from_user()
            if label is None:
                # User cancelled the dialog
                return
            
            # Create timestamp with format: DDmmHHMMSS (day month hour minute second)
            now = datetime.now()
            timestamp = now.strftime("%d%m%H%M%S")  # DDmmHHMMSS format
            
            # Create folder name based on label
            if label:
                folder_name = f"{label}"
            else:
                folder_name = f"recording_{timestamp}"
            
            # Create folder path
            folder_path = self.recording_dir / folder_name
            folder_path.mkdir(exist_ok=True)
            
            # Create filename with timestamp
            if label:
                filename = f"{label}_{timestamp}.bin"
            else:
                filename = f"capture_{timestamp}.bin"
            
            fname = folder_path / filename
            
            try:
                self.recording_fh = open(fname, "wb")
                self.recording_enabled = True
                self.btn_record_toggle.setText("Stop ⏹")
                self.status.showMessage(f"Recording to: {folder_name}/{filename}")
                print(f"[REC] Started recording to {fname}")
            except Exception as e:
                self.status.showMessage(f"Recording failed: {e}")
                print(f"[REC] Failed to start recording: {e}")
                # If we failed to open the file, reset the state
                self.recording_enabled = False
                self.btn_record_toggle.setText("Start ▶")
        else:
            # Stop recording
            try:
                if self.recording_fh is not None:
                    self.recording_fh.close()
                    self.recording_fh = None
            except Exception as e:
                print(f"[REC] Error closing file: {e}")
            
            self.recording_enabled = False
            self.btn_record_toggle.setText("Start Recording")
            self.status.showMessage("Recording stopped")
            print("[REC] Recording stopped")

    def shutdown(self):
        """
        Stop streaming cleanly:
          1) Stop UI timer
          2) Stop UDP receiver thread
          3) Best-effort: stop DCA1000 capture (STOP cmd)
          4) Best-effort: send sensorStop to radar over UART
          5) Close the window
        """
        if getattr(self, '_shutting_down', False):
            return
        self._shutting_down = True

        try:
            self.timer.stop()
        except Exception:
            pass

        # Stop UDP receiver
        try:
            if self.assembler is not None:
                self.assembler.stop()
                # give thread a moment to exit
                try:
                    self.assembler.join(timeout=0.5)
                except Exception:
                    pass
        except Exception:
            pass

        # Best-effort stop of DCA1000 (control port)
        try:
            cmds = dca_startup_commands()
            cfg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            cfg_sock.bind((HOST_IP, HOST_CONFIG_PORT))
            cfg_sock.settimeout(SOCKET_TIMEOUT_S)
            dca_stop(cfg_sock, cmds)
            cfg_sock.close()
        except Exception as e:
            print(f"[DCA] stop failed: {e}")

        # Best-effort stop of radar
        if USE_UART_TO_START_RADAR:
            awr_stop(RADAR_CLI_PORT, RADAR_CLI_BAUD)

        # Stop recording if active
        if self.recording_enabled:
            self.toggle_recording()  # This will stop recording and update button

        # Finally close the window
        self.close()

    def _apply_level_ema(self, key: str, lo: float, hi: float) -> tuple[float, float]:
        if LEVEL_EMA <= 0.0:
            return lo, hi
        a = float(LEVEL_EMA)
        if key == "rd":
            if self._rd_levels is None:
                self._rd_levels = (lo, hi)
            else:
                self._rd_levels = (a * self._rd_levels[0] + (1 - a) * lo, a * self._rd_levels[1] + (1 - a) * hi)
            return self._rd_levels
        if key == "ra":
            if self._ra_levels is None:
                self._ra_levels = (lo, hi)
            else:
                self._ra_levels = (a * self._ra_levels[0] + (1 - a) * lo, a * self._ra_levels[1] + (1 - a) * hi)
            return self._ra_levels
        if key == "md":
            if self._md_levels is None:
                self._md_levels = (lo, hi)
            else:
                self._md_levels = (a * self._md_levels[0] + (1 - a) * lo, a * self._md_levels[1] + (1 - a) * hi)
            return self._md_levels
        return lo, hi

    def on_timer(self):
        latest = None
        while True:
            try:
                latest = self.frame_q.get_nowait()
            except Empty:
                break
        if latest is None:
            return

        # ==============================
        # Raw frame recording (NO DSP impact)
        # ==============================
        if self.recording_enabled and self.recording_fh is not None:
            try:
                self.recording_fh.write(latest)
            except Exception as e:
                print(f"[REC] Write error: {e}")

        self._frame_count += 1

        # Range crop indices
        r0, r1 = RANGE_CROP
        if MICRO_RANGE_CROP is None:
            m0, m1 = r0, r1
        else:
            m0, m1 = MICRO_RANGE_CROP

        try:
            cube = bytes_to_cube_real(latest, self.cfg)
            rd_db, md_prof_db = compute_rd_and_profile(cube, self.dsp, m0, m1)
        except Exception as e:
            self.status.showMessage(f"DSP error: {e}")
            return

        # Crop RD display
        r0 = max(0, min(r0, rd_db.shape[1]))
        r1 = max(r0 + 1, min(r1, rd_db.shape[1]))
        rd_db = rd_db[:, r0:r1]

        # EMA RD
        self._rd_ema = rd_db if self._rd_ema is None else (EMA_ALPHA_RD * self._rd_ema + (1.0 - EMA_ALPHA_RD) * rd_db)
        rd_show = self._rd_ema
        if SMOOTH_KERNEL_RD > 1:
            rd_show = box_blur_2d(rd_show, SMOOTH_KERNEL_RD)

        # ====== Micro-doppler history update (FROM SAME RD, no extra FFT) ======
        self._md_ema = md_prof_db if self._md_ema is None else (EMA_ALPHA_MD * self._md_ema + (1.0 - EMA_ALPHA_MD) * md_prof_db)
        prof = self._md_ema

        # shift left, append new column at right
        self._md_hist[:, :-1] = self._md_hist[:, 1:]
        self._md_hist[:, -1] = prof

        md_show = self._md_hist
        if SMOOTH_KERNEL_MD > 1:
            md_show = box_blur_2d(md_show, SMOOTH_KERNEL_MD)

        # ====== RA update (throttled) ======
        if (self._frame_count % ANGLE_UPDATE_EVERY) == 0:
            try:
                ra = compute_ra(cube, self.dsp)
                self._last_ra = ra
            except Exception:
                pass

        ra_show = None
        if self._last_ra is not None:
            ra = self._last_ra
            r0d = r0 // RA_RANGE_DECIM
            r1d = max(r0d + 1, r1 // RA_RANGE_DECIM)
            r1d = min(r1d, ra.shape[1])
            ra = ra[:, r0d:r1d]

            self._ra_ema = ra if self._ra_ema is None else (EMA_ALPHA_RA * self._ra_ema + (1.0 - EMA_ALPHA_RA) * ra)
            ra_show = self._ra_ema
            if SMOOTH_KERNEL_RA > 1:
                ra_show = box_blur_2d(ra_show, SMOOTH_KERNEL_RA)

        # ====== Levels + draw ======
        if AUTO_LEVELS:
            rd_lo, rd_hi = levels_from_percentiles(rd_show, LEVEL_PCT_LOW_RD, LEVEL_PCT_HIGH_RD)
            md_lo, md_hi = levels_from_percentiles(md_show, LEVEL_PCT_LOW_MD, LEVEL_PCT_HIGH_MD)
        else:
            rd_lo, rd_hi = float(np.min(rd_show)), float(np.max(rd_show))
            md_lo, md_hi = float(np.min(md_show)), float(np.max(md_show))

        rd_lo, rd_hi = self._apply_level_ema("rd", rd_lo, rd_hi)
        md_lo, md_hi = self._apply_level_ema("md", md_lo, md_hi)

        self.rd_img.setImage(rd_show, autoLevels=False, levels=(rd_lo, rd_hi), lut=LUT)
        self.md_img.setImage(md_show.T, autoLevels=False, levels=(md_lo, md_hi), lut=LUT)

        if ra_show is not None:
            if AUTO_LEVELS:
                ra_lo, ra_hi = levels_from_percentiles(ra_show, LEVEL_PCT_LOW_RA, LEVEL_PCT_HIGH_RA)
            else:
                ra_lo, ra_hi = float(np.min(ra_show)), float(np.max(ra_show))
            ra_lo, ra_hi = self._apply_level_ema("ra", ra_lo, ra_hi)
            self.ra_img.setImage(ra_show, autoLevels=False, levels=(ra_lo, ra_hi), lut=LUT)

        # FPS status
        self._fps_frames += 1
        t = time.time()
        if t - self._fps_t0 > 1.0:
            fps = self._fps_frames / (t - self._fps_t0)
            self._fps_frames = 0
            self._fps_t0 = t
            md_rng = MICRO_RANGE_CROP if MICRO_RANGE_CROP is not None else RANGE_CROP
            self.status.showMessage(
                f"FPS ~ {fps:.1f} | RDfft=(D{DOPPLER_FFT},R{RANGE_FFT}) RAfft=A{ANGLE_FFT} "
                f"| RA every {ANGLE_UPDATE_EVERY}f decim={RA_RANGE_DECIM} | Micro range={md_rng}"
            )

    def closeEvent(self, event):
        # If user clicks the window X, run the same shutdown sequence.
        self.shutdown()
        event.accept()

def main():
    cfg_text = Path(RADAR_CFG_FILE).read_text(encoding="utf-8", errors="ignore")
    cfg = parse_awr_cfg(cfg_text)

    print("[CFG] Parsed:", cfg)
    print(f"[PLOTS] RD/RA range={RANGE_CROP} | Micro range={MICRO_RANGE_CROP or RANGE_CROP}")
    print(f"[FFT] DOPPLER_FFT={DOPPLER_FFT}, RANGE_FFT={RANGE_FFT}, ANGLE_FFT={ANGLE_FFT} | RA every {ANGLE_UPDATE_EVERY}f decim={RA_RANGE_DECIM}")

    app = QtWidgets.QApplication([])
    ui = RadarUI(cfg)
    ui.resize(1100, 900)
    ui.show()
    ui.start_udp()
    app.exec()

if __name__ == "__main__":
    main()