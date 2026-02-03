import socket
import time
import signal
import serial
from pathlib import Path

# ==============================
# NETWORK SETTINGS (DCA1000)
# ==============================
HOST_IP = "192.168.33.30"
HOST_DATA_PORT = 4098
HOST_CONFIG_PORT = 4096

DCA_IP = "192.168.33.180"
DCA_CONFIG_PORT = 4096

# ==============================
# RADAR UART SETTINGS (AWR2944)
# ==============================
RADAR_CLI_PORT = "COM10"              # CLI / application UART (you proved this is correct)
RADAR_CLI_BAUD = 115200
RADAR_CFG_FILE = r".\config\awr2944_cfg.cfg"

# ==============================
# FRAME PARAMS (fallback if you don't parse)
# ==============================
BYTES_PER_SAMPLE = 4  # typical I/Q 16-bit -> 4 bytes per complex sample
UDP_BUFFER_SIZE = 2 * 1024 * 1024

# ==============================
# DCA1000 COMMAND PACKETS
# ==============================
def make_cmd(code_hex, data=b""):
    """
    Build DCA1000 command packet
    [0xA55A][Code][Len][Data][0xEEAA]  (little-endian for header/code/len/footer)
    """
    header = (0xA55A).to_bytes(2, "little")
    footer = (0xEEAA).to_bytes(2, "little")
    code = int(code_hex, 16).to_bytes(2, "little")
    length = len(data).to_bytes(2, "little")
    return header + code + length + data + footer

def dca_startup_commands():
    return {
        "CONNECT": make_cmd("09"),
        "READ_FPGA": make_cmd("0E"),
        "CONFIG_FPGA": make_cmd("03", (0x01020102031E).to_bytes(6, "big")),
        "CONFIG_PACKET": make_cmd("0B", (0xC005350C0000).to_bytes(6, "big")),
        "START": make_cmd("05"),
        "STOP": make_cmd("06"),
    }

# ==============================
# UART HELPERS (THE IMPORTANT FIX)
# ==============================
PROMPT = b"mmwDemo:/>"

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

def _read_until_prompt(s: serial.Serial, timeout_s: float = 1.8) -> bytes:
    """
    Read until we see the CLI prompt or we time out.
    """
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

def uart_send_line_prompted(
    s: serial.Serial,
    line: str,
    *,
    char_delay_s: float = 0.0015,   # <-- KEY: slow down per character
    post_line_delay_s: float = 0.02,
    prompt_timeout_s: float = 2.0,
) -> bytes:
    """
    Send one CLI line reliably:
      - drain any pending output
      - write chars slowly (prevents dropped bytes)
      - end with CRLF
      - wait for prompt
    """
    _drain_read(s, 0.10)

    payload = (line + "\r\n").encode("ascii", errors="ignore")

    # Slow per-character write to avoid UART overrun on mmWave demo CLI
    for b in payload:
        s.write(bytes([b]))
        s.flush()
        time.sleep(char_delay_s)

    time.sleep(post_line_delay_s)
    resp = _read_until_prompt(s, prompt_timeout_s)
    return resp

def resp_text(resp: bytes) -> str:
    return resp.decode(errors="ignore")

def resp_is_problem(resp: bytes) -> bool:
    t = resp_text(resp)
    # Catch typical failure patterns
    return ("Error" in t) or ("not recognized" in t) or ("Invalid usage" in t)

# ==============================
# RADAR CONTROL (UART)
# ==============================
def start_radar_via_uart(cli_port: str, baud: int, cfg_path: str) -> bool:
    """
    Send full config over UART then sensorStart.
    Returns True if config looked clean enough to proceed.
    """
    cfg_abs = str(Path(cfg_path).resolve())
    print(f"\n[UART] Opening radar CLI {cli_port} @ {baud} ...")
    print(f"[UART] Using cfg: {cfg_abs}")

    cfg_problem_lines = 0
    cfg_sent_lines = 0

    with serial.Serial(cli_port, baudrate=baud, timeout=0.2) as s:
        # give the port time to settle
        time.sleep(0.6)
        s.reset_input_buffer()
        s.reset_output_buffer()

        # Sync to prompt: send blank line and wait
        print("[UART] Prompt sync:")
        sync = uart_send_line_prompted(s, "", char_delay_s=0.0015, prompt_timeout_s=2.5)
        print(resp_text(sync))
        print("-" * 60)

        # Always stop first
        print("[UART] sensorStop:")
        r = uart_send_line_prompted(s, "sensorStop", char_delay_s=0.0015, prompt_timeout_s=2.5)
        print(resp_text(r))
        print("-" * 60)

        print(f"[UART] Sending config file: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("%"):
                    continue

                # Do NOT send sensorStart from inside file; we will do it once at end
                if line.lower().startswith("sensorstart"):
                    continue

                cfg_sent_lines += 1
                r = uart_send_line_prompted(s, line, char_delay_s=0.0015, prompt_timeout_s=2.5)

                if resp_is_problem(r):
                    cfg_problem_lines += 1
                    print(f"[UART][CFG PROBLEM] line: {line}")
                    print(resp_text(r))
                    print("-" * 60)

        print(f"[UART] Config lines sent: {cfg_sent_lines} | problem lines: {cfg_problem_lines}")

        print("[UART] sensorStart:")
        r = uart_send_line_prompted(s, "sensorStart", char_delay_s=0.0015, prompt_timeout_s=3.5)
        print(resp_text(r))
        print("-" * 60)

        # sanity: query status
        print("[UART] queryDemoStatus (for sanity):")
        r = uart_send_line_prompted(s, "queryDemoStatus", char_delay_s=0.0015, prompt_timeout_s=2.5)
        print(resp_text(r))
        print("-" * 60)

    print("[UART] Done.")
    if cfg_problem_lines > 0:
        print("[WARN] UART cfg had problems. Sensor may not start; fix UART/cfg issues first.")
        return False
    return True

def stop_radar_via_uart(cli_port: str, baud: int):
    try:
        print(f"[UART] Opening {cli_port} to send sensorStop...")
        with serial.Serial(cli_port, baudrate=baud, timeout=0.2) as s:
            time.sleep(0.4)
            s.reset_input_buffer()
            s.reset_output_buffer()
            r = uart_send_line_prompted(s, "sensorStop", char_delay_s=0.0015, prompt_timeout_s=2.5)
            print(resp_text(r))
            print("-" * 60)
        print("[UART] sensorStop sent.")
    except Exception as e:
        print(f"[UART] Could not send sensorStop ({e}).")

# ==============================
# CLEAN SHUTDOWN
# ==============================
running = True

def handle_exit(sig, frame):
    global running
    print("\nStopping...")
    running = False

signal.signal(signal.SIGINT, handle_exit)

# ==============================
# MAIN
# ==============================
def main():
    print("DCA AWR UDP Test")

    # NOTE: you can still compute frame size yourself; leaving your cfg-derived message as-is.
    # For now just keep a default "expected" until you parse it.
    # If you already have parsing in your local version, keep that.
    FRAME_BYTES = 524288  # <-- keep what your script currently prints "from cfg" (or replace with your parser)
    print(f"Expecting {FRAME_BYTES} bytes per frame (from cfg)")

    cmds = dca_startup_commands()

    # Config socket (control channel)
    cfg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cfg_sock.bind((HOST_IP, HOST_CONFIG_PORT))
    cfg_sock.settimeout(1.0)

    # Data socket
    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.bind((HOST_IP, HOST_DATA_PORT))
    data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UDP_BUFFER_SIZE)
    data_sock.settimeout(1.0)

    def send_cmd(name):
        print(f"Sending: {name}")
        cfg_sock.sendto(cmds[name], (DCA_IP, DCA_CONFIG_PORT))
        try:
            data, _ = cfg_sock.recvfrom(2048)
            print(f"ACK ({name}): {data.hex()}")
        except socket.timeout:
            print(f"No ACK for {name}")

    # 1) DCA Config
    send_cmd("CONNECT")
    send_cmd("READ_FPGA")
    send_cmd("CONFIG_FPGA")
    send_cmd("CONFIG_PACKET")
    send_cmd("START")

    # 2) AWR Config (UART)
    ok = start_radar_via_uart(RADAR_CLI_PORT, RADAR_CLI_BAUD, RADAR_CFG_FILE)
    if not ok:
        print("[FATAL] UART config still has problems. Fix cfg/UART before expecting UDP data.")
        # continue anyway if you want, but it's almost guaranteed to be no-data

    print("\nListening for ADC data...\n")

    byte_buffer = bytearray()
    frame_count = 0
    packet_count = 0

    last_time = time.time()
    bytes_this_sec = 0

    while running:
        try:
            packet, addr = data_sock.recvfrom(UDP_BUFFER_SIZE)
        except socket.timeout:
            print(f"No data received at {time.time()}")
            continue

        print(f"Got UDP packet {len(packet)} bytes from {addr}")
        packet_count += 1

        # DCA1000 data packets typically have a 10-byte header
        payload = packet[10:]
        byte_buffer.extend(payload)
        bytes_this_sec += len(payload)

        now = time.time()
        if now - last_time >= 1.0:
            print(f"Packets/sec: {packet_count} | Bytes/sec: {bytes_this_sec}")
            packet_count = 0
            bytes_this_sec = 0
            last_time = now

        while len(byte_buffer) >= FRAME_BYTES:
            frame_count += 1
            print(f"Frame #{frame_count} assembled ({FRAME_BYTES} bytes)")
            byte_buffer = byte_buffer[FRAME_BYTES:]

    print("Stopping DCA1000 stream...")
    cfg_sock.sendto(cmds["STOP"], (DCA_IP, DCA_CONFIG_PORT))

    print("Stopping radar...")
    stop_radar_via_uart(RADAR_CLI_PORT, RADAR_CLI_BAUD)

    cfg_sock.close()
    data_sock.close()
    print("Clean exit.")

if __name__ == "__main__":
    main()
