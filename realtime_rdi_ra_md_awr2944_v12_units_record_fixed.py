"""
realtime_rdi_ra_md_awr2944_v12_units_record_fixed.py
=============================================

Adds:
1) Axis units/labels for:
   - Range–Doppler (Range in meters, Doppler in m/s)
   - Range–Angle (Range in meters, Angle in degrees; assumes d = λ/2 ULA)
   - Micro-Doppler (Time in seconds, Doppler in m/s)

2) Recording controls (NO restart needed):
   - Start Recording: begins buffering raw UDP frame bytes (exact frames received)
   - Stop Recording: ends current clip (keeps clip in memory)
   - Save Recording: writes last clip to a .bin file in a chosen folder

Notes / assumptions for axes:
- Range axis uses FMCW mapping: R = c * f_b / (2*slope), with f_b from FFT bin frequency.
- Doppler axis uses: v = (λ/2) * f_d, where f_d from slow-time FFT bin frequency and λ from center freq.
- Angle axis uses ULA spatial FFT approximation with element spacing d=λ/2:
    sin(theta) ≈ 2*(k - N/2)/N, theta = asin(...)
  This yields a reasonable angle axis for visualization.

If any cfg fields are missing, axes fall back to bin indices.

Run:
  python realtime_rdi_ra_md_awr2944_v12_units_record_fixed.py
"""

from __future__ import annotations

import socket
import time
import threading as th
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
import re
from datetime import datetime

import numpy as np

from PyQt6 import QtCore, QtWidgets, QtGui
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
RADAR_CLI_PORT = "COM7"
RADAR_CLI_BAUD = 115200
RADAR_CFG_FILE = r".\awr2944_cfg_updated.cfg"

PROMPT = b"mmwDemo:/>"
CHAR_DELAY_S = 0.0015
PROMPT_TIMEOUT_S = 3.0
DO_NOT_START_DCA = False


# ==============================
# PERFORMANCE / QUALITY KNOBS
# ==============================
ENABLE_MTI = True
MTI_MODE = "mean"  # "mean" or "diff"

DOPPLER_FFT = 128
RANGE_FFT = 256
ANGLE_FFT = 64

RANGE_CROP = (5, 110)
MICRO_RANGE_CROP = None  # e.g. (30, 70)

USE_DB = True
DB_FLOOR_RD = -10.0
DB_FLOOR_RA = -10.0
DB_FLOOR_MD = -10.0

RX_COMBINE = "sum"

ENABLE_DOPPLER_NOTCH = True
DOPPLER_NOTCH_BINS = 1

ENABLE_RX_PHASE_NORM = False

EMA_ALPHA_RD = 0.65
EMA_ALPHA_RA = 0.70
EMA_ALPHA_MD = 0.35

SMOOTH_KERNEL_RDRA = 3
SMOOTH_KERNEL_MD = 1

AUTO_LEVELS = True
LEVEL_PCT_LOW_RD = 20.0
LEVEL_PCT_HIGH_RD = 99.7
LEVEL_PCT_LOW_RA = 15.0
LEVEL_PCT_HIGH_RA = 99.7
LEVEL_PCT_LOW_MD = 15.0
LEVEL_PCT_HIGH_MD = 99.7
LEVEL_EMA = 0.80

ANGLE_UPDATE_EVERY = 2
RA_RANGE_DECIM = 2

MICRO_HISTORY = 180
MD_UPDATE_EVERY = 2

PLOT_UPDATE_MS = 30


# ==============================
# Recording
# ==============================
DEFAULT_RECORD_DIR = "recordings"
DEFAULT_CLIP_BASENAME = "gesture"
MAX_CLIP_BYTES_IN_RAM = 512 * 1024 * 1024  # 512MB safety cap


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


def awr_stop(cli_port: str, baud: int) -> bool:
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


# ==============================
# Parse cfg (+ waveform params for axes)
# ==============================
@dataclass(frozen=True)
class WaveformParams:
    start_freq_ghz: float | None
    slope_mhz_per_us: float | None
    ramp_end_time_us: float | None
    idle_time_us: float | None
    sample_rate_ksps: float | None
    frame_period_ms: float | None


@dataclass(frozen=True)
class RadarFrameConfig:
    adc_samples: int
    chirps_per_frame: int
    rx_channels: int
    wf: WaveformParams

    @property
    def frame_bytes(self) -> int:
        return self.chirps_per_frame * self.adc_samples * self.rx_channels * 2


def _popcount(x: int) -> int:
    return bin(x & 0xFFFFFFFF).count("1")


def parse_awr_cfg(cfg_text: str) -> RadarFrameConfig:
    adc_samples = None
    chirps_per_frame = None
    rx_channels = None

    start_freq = None
    idle_us = None
    ramp_end_us = None
    slope = None
    sample_rate = None
    frame_period = None

    for raw in cfg_text.splitlines():
        line = raw.strip()
        if (not line) or line.startswith("%") or line.startswith("#"):
            continue

        if line.startswith("profileCfg"):
            parts = re.split(r"\s+", line)
            if len(parts) >= 12:
                start_freq = float(parts[2])
                idle_us = float(parts[3])
                ramp_end_us = float(parts[5])
                slope = float(parts[8])
                adc_samples = int(float(parts[10]))
                sample_rate = float(parts[11])  # ksps

        if line.startswith("frameCfg"):
            parts = re.split(r"\s+", line)
            if len(parts) >= 6:
                chirps_per_frame = int(float(parts[3]))
                try:
                    frame_period = float(parts[5])
                except Exception:
                    frame_period = None

        if line.startswith("channelCfg"):
            parts = re.split(r"\s+", line)
            if len(parts) >= 3:
                rx_en = int(parts[1])
                rx_channels = _popcount(rx_en & 0xF)

    if adc_samples is None or chirps_per_frame is None or rx_channels is None:
        raise ValueError("Could not parse adc_samples / chirps_per_frame / rx_channels from cfg.")

    wf = WaveformParams(
        start_freq_ghz=start_freq,
        slope_mhz_per_us=slope,
        ramp_end_time_us=ramp_end_us,
        idle_time_us=idle_us,
        sample_rate_ksps=sample_rate,
        frame_period_ms=frame_period,
    )
    return RadarFrameConfig(adc_samples=adc_samples, chirps_per_frame=chirps_per_frame, rx_channels=rx_channels, wf=wf)


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
# DSP
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


def compute_rd_and_profile(cube: np.ndarray, dsp: DSPState, r0_prof: int, r1_prof: int) -> tuple[np.ndarray, np.ndarray]:
    x = dsp.apply_windows(cube)
    x = _mti(x)

    Xr = np.fft.fft(x, n=RANGE_FFT, axis=1).astype(np.complex64, copy=False)
    Xrd = np.fft.fft(Xr, n=DOPPLER_FFT, axis=0)
    Xrd = np.fft.fftshift(Xrd, axes=0)

    mag = np.abs(Xrd).astype(np.float32, copy=False)
    rd = mag.max(axis=2) if RX_COMBINE == "max" else mag.sum(axis=2)

    if ENABLE_DOPPLER_NOTCH and DOPPLER_NOTCH_BINS > 0:
        c0 = rd.shape[0] // 2
        b = int(DOPPLER_NOTCH_BINS)
        rd[max(0, c0 - b): min(rd.shape[0], c0 + b + 1), :] = np.min(rd)

    r0p = max(0, min(r0_prof, rd.shape[1]))
    r1p = max(r0p + 1, min(r1_prof, rd.shape[1]))
    prof = rd[:, r0p:r1p].sum(axis=1)

    if USE_DB:
        rd_db = 20.0 * np.log10(rd + 1e-6)
        rd_db = np.maximum(rd_db, DB_FLOOR_RD)
        prof_db = 20.0 * np.log10(prof + 1e-6)
        prof_db = np.maximum(prof_db, DB_FLOOR_MD)
    else:
        rd_db, prof_db = rd, prof

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
# Axis calculations
# ==============================
C = 299_792_458.0  # m/s


def compute_axes(cfg: RadarFrameConfig):
    wf = cfg.wf
    range_m_full = None
    doppler_ms = None
    angle_deg = None
    frame_period_s = wf.frame_period_ms / 1000.0 if wf.frame_period_ms else None
    md_dt_s = None

    if wf.sample_rate_ksps and wf.slope_mhz_per_us:
        Fs = wf.sample_rate_ksps * 1e3
        slope_hz_per_s = wf.slope_mhz_per_us * 1e12
        fb = (np.arange(RANGE_FFT, dtype=np.float64) * Fs / RANGE_FFT)
        range_m_full = (C * fb) / (2.0 * slope_hz_per_s)

    if wf.idle_time_us is not None and wf.ramp_end_time_us is not None and wf.start_freq_ghz is not None:
        Tc = (wf.idle_time_us + wf.ramp_end_time_us) * 1e-6
        fc = wf.start_freq_ghz * 1e9
        lam = C / fc
        fd = (np.arange(DOPPLER_FFT, dtype=np.float64) - (DOPPLER_FFT / 2.0)) / (DOPPLER_FFT * Tc)
        doppler_ms = (lam / 2.0) * fd

        k = np.arange(ANGLE_FFT, dtype=np.float64) - (ANGLE_FFT / 2.0)
        sin_th = 2.0 * (k / ANGLE_FFT)
        sin_th = np.clip(sin_th, -1.0, 1.0)
        angle_deg = np.degrees(np.arcsin(sin_th))

    if frame_period_s is not None:
        md_dt_s = frame_period_s * MD_UPDATE_EVERY

    return {"range_m_full": range_m_full, "doppler_ms": doppler_ms, "angle_deg": angle_deg, "md_dt_s": md_dt_s}


def set_image_axes(img: pg.ImageItem, x_vals: np.ndarray | None, y_vals: np.ndarray | None, *, invert_y: bool = True):
    """
    Applies an affine transform so image pixels map to physical axis values.

    Important: ImageItem's row index increases downward. For most plots we want
    the Y-axis to increase upward (e.g., Doppler: negative at bottom, positive at top;
    Angle: -deg at bottom, +deg at top). Set invert_y=True (default) to flip Y.
    """
    if x_vals is None or y_vals is None:
        return
    if img.image is None:
        return

    cols = int(img.image.shape[1])
    rows = int(img.image.shape[0])
    if cols < 2 or rows < 2:
        return

    x0, x1 = float(x_vals[0]), float(x_vals[-1])
    y0, y1 = float(y_vals[0]), float(y_vals[-1])

    sx = (x1 - x0) / (cols - 1)
    sy = (y1 - y0) / (rows - 1)

    tr = QtGui.QTransform()
    if invert_y:
        # Map row 0 -> y1 (top), row rows-1 -> y0 (bottom)
        tr.translate(x0, y1)
        tr.scale(sx, -sy)
    else:
        tr.translate(x0, y0)
        tr.scale(sx, sy)

    img.setTransform(tr)

    x0, x1 = float(x_vals[0]), float(x_vals[-1])
    y0, y1 = float(y_vals[0]), float(y_vals[-1])

    sx = (x1 - x0) / (cols - 1)
    sy = (y1 - y0) / (rows - 1)

    tr = QtGui.QTransform()
    tr.translate(x0, y0)
    tr.scale(sx, sy)
    img.setTransform(tr)


# ==============================
# GUI
# ==============================
class RadarUI(QtWidgets.QMainWindow):
    def __init__(self, cfg: RadarFrameConfig):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Real-Time Radar")

        self.axes = compute_axes(cfg)

        self.frame_q: Queue[bytes] = Queue()
        self.assembler: UdpFrameAssembler | None = None
        self.dsp = DSPState(cfg.chirps_per_frame, cfg.adc_samples)

        self._rd_ema: np.ndarray | None = None
        self._ra_ema: np.ndarray | None = None
        self._md_prof_ema: np.ndarray | None = None

        self._rd_levels: tuple[float, float] | None = None
        self._ra_levels: tuple[float, float] | None = None
        self._md_levels: tuple[float, float] | None = None

        self._frame_count = 0
        self._last_ra: np.ndarray | None = None

        self._md_buf = np.full((DOPPLER_FFT, MICRO_HISTORY), DB_FLOOR_MD, dtype=np.float32)
        self._md_col = 0

        # Recording
        self._recording = False
        self._clip_buf = bytearray()
        self._last_clip: bytes | None = None
        self._clip_count = 0
        self._save_dir = Path(DEFAULT_RECORD_DIR)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # Titles
        titles_top = QtWidgets.QHBoxLayout()
        self.lbl_rd = QtWidgets.QLabel("Range-Doppler")
        self.lbl_ra = QtWidgets.QLabel("Range-Angle")
        for lbl in (self.lbl_rd, self.lbl_ra):
            f = lbl.font()
            f.setPointSize(14)
            f.setBold(True)
            lbl.setFont(f)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        titles_top.addWidget(self.lbl_rd)
        titles_top.addWidget(self.lbl_ra)
        outer.addLayout(titles_top)

        # Plots top
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

        self.rd_plot.setLabel("bottom", "Range", units="m")
        self.rd_plot.setLabel("left", "Doppler", units="m/s")
        self.ra_plot.setLabel("bottom", "Range", units="m")
        self.ra_plot.setLabel("left", "Angle", units="deg")

        self.rd_img = pg.ImageItem()
        self.ra_img = pg.ImageItem()
        self.rd_img.setAutoDownsample(False)
        self.ra_img.setAutoDownsample(False)
        self.rd_plot.addItem(self.rd_img)
        self.ra_plot.addItem(self.ra_img)
        self.rd_img.setLookupTable(LUT)
        self.ra_img.setLookupTable(LUT)

        # Micro-doppler bottom
        self.lbl_md = QtWidgets.QLabel("Micro-Doppler")
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
        self.md_plot.setLabel("bottom", "Time", units="s")
        self.md_plot.setLabel("left", "Doppler", units="m/s")

        self.md_img = pg.ImageItem()
        self.md_img.setAutoDownsample(False)
        self.md_plot.addItem(self.md_img)
        self.md_img.setLookupTable(LUT)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        outer.addLayout(btn_row)

        self.btn_send = QtWidgets.QPushButton("Send Radar Config")
        self.btn_rec_start = QtWidgets.QPushButton("Start Recording")
        self.btn_rec_stop = QtWidgets.QPushButton("Stop Recording")
        self.btn_rec_save = QtWidgets.QPushButton("Save Recording")
        self.btn_exit = QtWidgets.QPushButton("Exit")

        self.btn_send.clicked.connect(self.on_send_config)
        self.btn_rec_start.clicked.connect(self.on_record_start)
        self.btn_rec_stop.clicked.connect(self.on_record_stop)
        self.btn_rec_save.clicked.connect(self.on_record_save)
        self.btn_exit.clicked.connect(self.shutdown)

        btn_row.addWidget(self.btn_send)
        btn_row.addSpacing(10)
        btn_row.addWidget(self.btn_rec_start)
        btn_row.addWidget(self.btn_rec_stop)
        btn_row.addWidget(self.btn_rec_save)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_exit)

        # Status + timer
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self._fps_frames = 0
        self._fps_t0 = time.time()
        self._shutting_down = False

        self._update_record_buttons()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(PLOT_UPDATE_MS)

    # ---------- Recording ----------
    def _update_record_buttons(self):
        self.btn_rec_start.setEnabled(not self._recording)
        self.btn_rec_stop.setEnabled(self._recording)
        self.btn_rec_save.setEnabled((self._last_clip is not None) and (not self._recording))

    def on_record_start(self):
        self._recording = True
        self._clip_buf = bytearray()
        self._last_clip = None
        self._update_record_buttons()
        self.status.showMessage("Recording started...")

    def on_record_stop(self):
        self._recording = False
        self._last_clip = bytes(self._clip_buf) if len(self._clip_buf) > 0 else None
        nframes = (len(self._clip_buf) // self.cfg.frame_bytes) if self.cfg.frame_bytes else 0
        self._update_record_buttons()
        self.status.showMessage(f"Recording stopped. Frames: {nframes}, Bytes: {len(self._clip_buf)}")

    def on_record_save(self):
        if self._last_clip is None:
            self.status.showMessage("No clip to save. Record something first.")
            return

        start_dir = str(self._save_dir.resolve())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder to Save .bin", start_dir)
        if not folder:
            return
        self._save_dir = Path(folder)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._clip_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{DEFAULT_CLIP_BASENAME}_{self._clip_count:03d}_{ts}.bin"
        fpath = self._save_dir / fname

        try:
            with open(fpath, "wb") as f:
                f.write(self._last_clip)
            self.status.showMessage(f"Saved: {fpath} ({len(self._last_clip)} bytes)")
        except Exception as e:
            self.status.showMessage(f"Save failed: {e}")

    # ---------- Streaming ----------
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

    # ---------- Shutdown ----------
    def shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True

        try:
            self.timer.stop()
        except Exception:
            pass

        try:
            if self.assembler is not None:
                self.assembler.stop()
                try:
                    self.assembler.join(timeout=0.5)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            cmds = dca_startup_commands()
            cfg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            cfg_sock.bind((HOST_IP, HOST_CONFIG_PORT))
            cfg_sock.settimeout(SOCKET_TIMEOUT_S)
            dca_stop(cfg_sock, cmds)
            cfg_sock.close()
        except Exception as e:
            print(f"[DCA] stop failed: {e}")

        if USE_UART_TO_START_RADAR:
            awr_stop(RADAR_CLI_PORT, RADAR_CLI_BAUD)

        self.close()

    def closeEvent(self, event):
        self.shutdown()
        event.accept()

    # ---------- Levels ----------
    def _apply_level_ema(self, key: str, lo: float, hi: float) -> tuple[float, float]:
        if LEVEL_EMA <= 0.0:
            return lo, hi
        a = float(LEVEL_EMA)
        if key == "rd":
            self._rd_levels = (lo, hi) if self._rd_levels is None else (a * self._rd_levels[0] + (1 - a) * lo, a * self._rd_levels[1] + (1 - a) * hi)
            return self._rd_levels
        if key == "ra":
            self._ra_levels = (lo, hi) if self._ra_levels is None else (a * self._ra_levels[0] + (1 - a) * lo, a * self._ra_levels[1] + (1 - a) * hi)
            return self._ra_levels
        if key == "md":
            self._md_levels = (lo, hi) if self._md_levels is None else (a * self._md_levels[0] + (1 - a) * lo, a * self._md_levels[1] + (1 - a) * hi)
            return self._md_levels
        return lo, hi

    # ---------- Timer ----------
    def on_timer(self):
        latest = None
        while True:
            try:
                latest = self.frame_q.get_nowait()
            except Empty:
                break
        if latest is None:
            return

        if self._recording:
            if len(self._clip_buf) + len(latest) <= MAX_CLIP_BYTES_IN_RAM:
                self._clip_buf.extend(latest)
            else:
                self._recording = False
                self._last_clip = bytes(self._clip_buf) if len(self._clip_buf) > 0 else None
                self._update_record_buttons()
                self.status.showMessage("Recording auto-stopped: MAX_CLIP_BYTES_IN_RAM reached.")

        self._frame_count += 1

        r0, r1 = RANGE_CROP
        m0, m1 = (MICRO_RANGE_CROP if MICRO_RANGE_CROP is not None else RANGE_CROP)

        try:
            cube = bytes_to_cube_real(latest, self.cfg)
            rd_db, md_prof_db = compute_rd_and_profile(cube, self.dsp, m0, m1)
        except Exception as e:
            self.status.showMessage(f"DSP error: {e}")
            return

        r0 = max(0, min(r0, rd_db.shape[1]))
        r1 = max(r0 + 1, min(r1, rd_db.shape[1]))
        rd_db = rd_db[:, r0:r1]

        self._rd_ema = rd_db if self._rd_ema is None else (EMA_ALPHA_RD * self._rd_ema + (1.0 - EMA_ALPHA_RD) * rd_db)
        rd_show = self._rd_ema
        if SMOOTH_KERNEL_RDRA > 1:
            rd_show = box_blur_2d(rd_show, SMOOTH_KERNEL_RDRA)

        if (self._frame_count % ANGLE_UPDATE_EVERY) == 0:
            try:
                self._last_ra = compute_ra(cube, self.dsp)
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
            if SMOOTH_KERNEL_RDRA > 1:
                ra_show = box_blur_2d(ra_show, SMOOTH_KERNEL_RDRA)

        if (self._frame_count % MD_UPDATE_EVERY) == 0:
            self._md_prof_ema = md_prof_db if self._md_prof_ema is None else (EMA_ALPHA_MD * self._md_prof_ema + (1.0 - EMA_ALPHA_MD) * md_prof_db)
            prof = self._md_prof_ema
            self._md_buf[:, self._md_col] = prof
            self._md_col = (self._md_col + 1) % MICRO_HISTORY

        md_view = self._md_buf if self._md_col == 0 else np.hstack((self._md_buf[:, self._md_col:], self._md_buf[:, :self._md_col]))
        md_show = md_view
        if SMOOTH_KERNEL_MD > 1:
            md_show = box_blur_2d(md_show, SMOOTH_KERNEL_MD)

        if AUTO_LEVELS:
            rd_lo, rd_hi = levels_from_percentiles(rd_show, LEVEL_PCT_LOW_RD, LEVEL_PCT_HIGH_RD)
            md_lo, md_hi = levels_from_percentiles(md_show, LEVEL_PCT_LOW_MD, LEVEL_PCT_HIGH_MD)
        else:
            rd_lo, rd_hi = float(np.min(rd_show)), float(np.max(rd_show))
            md_lo, md_hi = float(np.min(md_show)), float(np.max(md_show))

        rd_lo, rd_hi = self._apply_level_ema("rd", rd_lo, rd_hi)
        md_lo, md_hi = self._apply_level_ema("md", md_lo, md_hi)

        self.rd_img.setImage(rd_show, autoLevels=False, levels=(rd_lo, rd_hi), lut=LUT)
        self.md_img.setImage(md_show, autoLevels=False, levels=(md_lo, md_hi), lut=LUT)

        if ra_show is not None:
            if AUTO_LEVELS:
                ra_lo, ra_hi = levels_from_percentiles(ra_show, LEVEL_PCT_LOW_RA, LEVEL_PCT_HIGH_RA)
            else:
                ra_lo, ra_hi = float(np.min(ra_show)), float(np.max(ra_show))
            ra_lo, ra_hi = self._apply_level_ema("ra", ra_lo, ra_hi)
            self.ra_img.setImage(ra_show, autoLevels=False, levels=(ra_lo, ra_hi), lut=LUT)
        # Axes transforms
        try:
            if self.axes["range_m_full"] is not None and self.axes["doppler_ms"] is not None:
                rng = self.axes["range_m_full"][r0:r1]
                dop = self.axes["doppler_ms"]
                set_image_axes(self.rd_img, x_vals=rng, y_vals=dop, invert_y=True)

            if ra_show is not None and self.axes["range_m_full"] is not None and self.axes["angle_deg"] is not None:
                rng = self.axes["range_m_full"][r0:r1]
                rng_dec = rng[::RA_RANGE_DECIM]
                ang = self.axes["angle_deg"]
                set_image_axes(self.ra_img, x_vals=rng_dec, y_vals=ang, invert_y=True)

            if self.axes["doppler_ms"] is not None and self.axes["md_dt_s"] is not None:
                dop = self.axes["doppler_ms"]
                dt = float(self.axes["md_dt_s"])
                t_axis = np.arange(MICRO_HISTORY, dtype=np.float64) * dt
                set_image_axes(self.md_img, x_vals=t_axis, y_vals=dop, invert_y=True)
        except Exception:
            # Don't crash the realtime loop if an axis calc is temporarily unavailable
            pass

        self._fps_frames += 1
        tnow = time.time()
        if tnow - self._fps_t0 > 1.0:
            fps = self._fps_frames / (tnow - self._fps_t0)
            self._fps_frames = 0
            self._fps_t0 = tnow
            rec = "REC" if self._recording else "idle"
            self.status.showMessage(f"FPS ~ {fps:.1f} | {rec} | clip_bytes={len(self._clip_buf)}")

        self._update_record_buttons()


def main():
    cfg_text = Path(RADAR_CFG_FILE).read_text(encoding="utf-8", errors="ignore")
    cfg = parse_awr_cfg(cfg_text)

    print("[CFG] Parsed:", cfg)
    print(f"[PLOTS] RD/RA range={RANGE_CROP} | Micro range={MICRO_RANGE_CROP or RANGE_CROP}")
    print(f"[FFT] DOPPLER_FFT={DOPPLER_FFT}, RANGE_FFT={RANGE_FFT}, ANGLE_FFT={ANGLE_FFT} | RA every {ANGLE_UPDATE_EVERY}f decim={RA_RANGE_DECIM}")

    try:
        Path(DEFAULT_RECORD_DIR).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    app = QtWidgets.QApplication([])
    ui = RadarUI(cfg)
    ui.resize(1150, 920)
    ui.show()
    ui.start_udp()
    app.exec()


if __name__ == "__main__":
    main()
