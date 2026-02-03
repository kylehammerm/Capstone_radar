"""
AWR2944 + DCA1000 UDP Smoke Test (Improved)
------------------------------------------
Changes made vs prior script (per your request):
  1) Start AWR (UART cfg + sensorStart) BEFORE DCA START.
  2) Bind UDP data socket to 0.0.0.0 so we don't miss packets due to NIC/IP binding.
  3) Add lightweight NIC sanity prints (getsockname) and basic UDP header debug toggle.
  4) Keep everything editable at the top, keep sections segmented, keep prints minimal:
       - AWR: print "what I send" + "response"
       - DCA: print command name + ACK hex (same as before)
  5) Ctrl+C clean shutdown with a summary that clearly says if you received data or not.

Save as: scripts/transport/04_udp_smoke.py  (or 05 if you prefer)
"""

import socket
import time
import signal
import serial
from pathlib import Path

# ==============================
# EDIT THESE (NETWORK: DCA1000)
# ==============================
HOST_IP = "192.168.33.30"     # Your PC's NIC IP on the DCA subnet (used for config bind)
HOST_CONFIG_PORT = 4096       # DCA config port on host (bind)

# Data receive bind:
#   - Recommended: "0.0.0.0" to listen on all NICs (prevents missing packets due to wrong bind IP)
#   - If you insist on binding to a specific NIC IP, set this to that IP (e.g. "192.168.33.30")
HOST_DATA_BIND_IP = "0.0.0.0"
HOST_DATA_PORT = 4098         # DCA data port on host (bind)

DCA_IP = "192.168.33.180"     # DCA1000 IP
DCA_CONFIG_PORT = 4096        # DCA1000 config port (destination)

# ==============================
# EDIT THESE (UART: AWR2944 CLI)
# ==============================
RADAR_CLI_PORT = "COM10"      # AWR CLI / application UART (mmwDemo prompt)
RADAR_CLI_BAUD = 115200
RADAR_CFG_FILE = r".\config\awr2944_cfg.cfg"

# ==============================
# EDIT THESE (UART tuning)
# ==============================
PROMPT = b"mmwDemo:/>"
CHAR_DELAY_S = 0.0015         # per-character delay; increase if any lines get dropped
PROMPT_TIMEOUT_S = 3.0        # seconds to wait for prompt after each line

# ==============================
# EDIT THESE (UDP receive / stats)
# ==============================
UDP_RCVBUF = 4 * 1024 * 1024      # OS socket receive buffer (bigger can help)
UDP_READ_SIZE = 2 * 1024 * 1024   # recvfrom max bytes
SOCKET_TIMEOUT_S = 1.0

PRINT_RATE_EVERY_S = 1.0          # prints "no data yet..." or rate every N seconds
PRINT_PACKET_EVERY = 0            # 0=don't print each packet; N=print every N packets

# DCA packets often have a small header. If you're not sure, leave at 10 (common).
DCA_DATA_HEADER_BYTES = 10

# Optional: if you want to assemble frames, set this. If unknown, keep 0 (disabled).
FRAME_BYTES = 0                   # e.g. 524288; set 0 to disable frame assembly

# Optional UDP debug: print first bytes of first few packets to confirm header assumptions
DEBUG_FIRST_N_PACKETS = 3         # 0 disables


# ==============================
# DCA1000 COMMAND PACKETS
# ==============================
def make_cmd(code_hex: str, data: bytes = b"") -> bytes:
    """
    Build DCA1000 command packet
    [0xA55A][Code][Len][Data][0xEEAA]  (little-endian header/code/len/footer)
    """
    header = (0xA55A).to_bytes(2, "little")
    footer = (0xEEAA).to_bytes(2, "little")
    code = int(code_hex, 16).to_bytes(2, "little")
    length = len(data).to_bytes(2, "little")
    return header + code + length + data + footer


def dca_startup_commands():
    # Same commands you already verified produce ACKs.
    return {
        "CONNECT": make_cmd("09"),
        "READ_FPGA": make_cmd("0E"),
        "CONFIG_FPGA": make_cmd("03", (0x01020102031E).to_bytes(6, "big")),
        "CONFIG_PACKET": make_cmd("0B", (0xC005350C0000).to_bytes(6, "big")),
        "START": make_cmd("05"),
        "STOP": make_cmd("06"),
    }


# ==============================
# UART HELPERS (AWR CLI)
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


def uart_send_line_prompted(
    s: serial.Serial,
    line: str,
    *,
    char_delay_s: float,
    prompt_timeout_s: float,
) -> bytes:
    """
    Send one CLI line reliably:
      - drain pending output
      - write chars slowly (prevents dropped bytes)
      - end with CRLF
      - wait for prompt
    """
    _drain_read(s, 0.10)
    payload = (line + "\r\n").encode("ascii", errors="ignore")

    for b in payload:
        s.write(bytes([b]))
        s.flush()
        time.sleep(char_delay_s)

    return _read_until_prompt(s, timeout_s=prompt_timeout_s)


def print_awr_exchange(cmd: str, resp: bytes) -> None:
    """
    Minimal AWR print: "what I sent" and "response"
    (Response includes echoed command + Done + prompt, which is fine.)
    """
    print(cmd)
    print(resp.decode(errors="ignore").strip())
    print("-" * 60)


# ==============================
# RADAR CONTROL (UART)
# ==============================
def awr_config_and_start(cli_port: str, baud: int, cfg_path: str) -> bool:
    """
    Sends cfg over UART and runs sensorStart.
    Returns True if no obvious UART errors found.
    """
    cfg_abs = str(Path(cfg_path).resolve())
    if not Path(cfg_abs).exists():
        print(f"[FATAL] AWR cfg file not found: {cfg_abs}")
        return False

    print(f"[UART] Opening radar CLI {cli_port} @ {baud}")
    print(f"[UART] Using cfg: {cfg_abs}")
    print("-" * 60)

    ok = True
    sent_lines = 0

    with serial.Serial(cli_port, baudrate=baud, timeout=0.2) as s:
        time.sleep(0.6)
        s.reset_input_buffer()
        s.reset_output_buffer()

        # Prompt sync
        resp = uart_send_line_prompted(
            s, "", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=PROMPT_TIMEOUT_S
        )
        print("[UART] Prompt sync:")
        print(resp.decode(errors="ignore").strip())
        print("-" * 60)

        # Safe stop
        resp = uart_send_line_prompted(
            s, "sensorStop", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=PROMPT_TIMEOUT_S
        )
        print_awr_exchange("sensorStop", resp)

        # Send cfg file lines
        print("[UART] Sending cfg lines...")
        print("-" * 60)
        with open(cfg_abs, "r", encoding="utf-8", errors="ignore") as f:
            for i, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("%") or line.startswith("#"):
                    continue
                if line.lower().startswith("sensorstart"):
                    # do it once at end
                    continue

                sent_lines += 1
                resp = uart_send_line_prompted(
                    s, line, char_delay_s=CHAR_DELAY_S, prompt_timeout_s=PROMPT_TIMEOUT_S
                )
                print_awr_exchange(line, resp)

                t = resp.decode(errors="ignore")
                if ("Error" in t) or ("not recognized" in t) or ("Invalid usage" in t) or ("Unknown command" in t):
                    ok = False

        print(f"[UART] Config lines sent: {sent_lines}")
        print("-" * 60)

        # Start sensor
        resp = uart_send_line_prompted(
            s, "sensorStart", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=max(PROMPT_TIMEOUT_S, 4.0)
        )
        print_awr_exchange("sensorStart", resp)

        # Quick sanity
        resp = uart_send_line_prompted(
            s, "queryDemoStatus", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=PROMPT_TIMEOUT_S
        )
        print_awr_exchange("queryDemoStatus", resp)

        t = resp.decode(errors="ignore")
        # Optional sanity checks (non-fatal)
        if "Sensor State" not in t:
            ok = False

    return ok


def awr_stop(cli_port: str, baud: int) -> None:
    try:
        with serial.Serial(cli_port, baudrate=baud, timeout=0.2) as s:
            time.sleep(0.4)
            s.reset_input_buffer()
            s.reset_output_buffer()
            resp = uart_send_line_prompted(
                s, "sensorStop", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=PROMPT_TIMEOUT_S
            )
            print_awr_exchange("sensorStop", resp)
    except Exception as e:
        print(f"[UART] Could not send sensorStop ({e}).")


# ==============================
# DCA CONTROL (UDP config channel)
# ==============================
def dca_send_cmd(cfg_sock: socket.socket, cmds: dict, name: str) -> bool:
    print(f"[DCA] Sending: {name}")
    cfg_sock.sendto(cmds[name], (DCA_IP, DCA_CONFIG_PORT))
    try:
        data, _ = cfg_sock.recvfrom(2048)
        print(f"[DCA] ACK ({name}): {data.hex()}")
        return True
    except socket.timeout:
        print(f"[DCA] No ACK for {name}")
        return False


def dca_config_only(cfg_sock: socket.socket, cmds: dict) -> bool:
    """
    Do everything except START. (We start streaming after radar is started.)
    """
    ok = True
    ok &= dca_send_cmd(cfg_sock, cmds, "CONNECT")
    ok &= dca_send_cmd(cfg_sock, cmds, "READ_FPGA")
    ok &= dca_send_cmd(cfg_sock, cmds, "CONFIG_FPGA")
    ok &= dca_send_cmd(cfg_sock, cmds, "CONFIG_PACKET")
    return ok


def dca_start(cfg_sock: socket.socket, cmds: dict) -> bool:
    return dca_send_cmd(cfg_sock, cmds, "START")


def dca_stop(cfg_sock: socket.socket, cmds: dict) -> None:
    # Best-effort stop
    try:
        print("[DCA] Sending: STOP")
        cfg_sock.sendto(cmds["STOP"], (DCA_IP, DCA_CONFIG_PORT))
        try:
            data, _ = cfg_sock.recvfrom(2048)
            print(f"[DCA] ACK (STOP): {data.hex()}")
        except socket.timeout:
            print("[DCA] No ACK for STOP")
    except Exception as e:
        print(f"[DCA] Could not send STOP ({e}).")


# ==============================
# CLEAN SHUTDOWN (Ctrl+C)
# ==============================
running = True


def handle_exit(sig, frame):
    global running
    print("\n[CTRL+C] Stopping...")
    running = False


signal.signal(signal.SIGINT, handle_exit)


# ==============================
# MAIN
# ==============================
def main():
    print("AWR2944 + DCA1000 UDP Smoke Test (Improved)")
    print("-" * 72)

    cmds = dca_startup_commands()

    # ----- Create sockets -----
    cfg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cfg_sock.bind((HOST_IP, HOST_CONFIG_PORT))
    cfg_sock.settimeout(SOCKET_TIMEOUT_S)

    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.bind((HOST_DATA_BIND_IP, HOST_DATA_PORT))
    data_sock.settimeout(SOCKET_TIMEOUT_S)
    data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UDP_RCVBUF)

    print("[INFO] Config socket bound to:", cfg_sock.getsockname())
    print("[INFO] Data socket bound to:  ", data_sock.getsockname())
    print("-" * 72)

    # ----- DCA config (NO START YET) -----
    print("[STEP] DCA1000 config (no START yet)")
    dca_cfg_ok = dca_config_only(cfg_sock, cmds)
    print("-" * 72)

    # ----- AWR config + start -----
    print("[STEP] AWR2944 UART config + sensorStart")
    awr_ok = awr_config_and_start(RADAR_CLI_PORT, RADAR_CLI_BAUD, RADAR_CFG_FILE)
    print("-" * 72)

    # ----- Now START DCA streaming -----
    print("[STEP] DCA1000 START (after radar is running)")
    dca_start_ok = dca_start(cfg_sock, cmds)
    print("-" * 72)

    if not dca_cfg_ok:
        print("[WARN] DCA config had missing ACK(s). Data may not stream.")
    if not awr_ok:
        print("[WARN] AWR UART config had potential issues. Data may not stream.")
    if not dca_start_ok:
        print("[WARN] DCA START had no ACK. Data may not stream.")

    print("[STEP] Listening for UDP data (Ctrl+C to stop)")
    print(f"  Data bind: {HOST_DATA_BIND_IP}:{HOST_DATA_PORT}")
    print(f"  Header strip: {DCA_DATA_HEADER_BYTES} bytes")
    if FRAME_BYTES > 0:
        print(f"  Frame assembly: enabled (FRAME_BYTES={FRAME_BYTES})")
    else:
        print("  Frame assembly: disabled (FRAME_BYTES=0)")
    print("-" * 72)

    # ----- Streaming stats -----
    got_any_data = False
    first_packet_ts = None
    last_packet_ts = None

    packet_count_total = 0
    payload_bytes_total = 0

    packet_count_interval = 0
    payload_bytes_interval = 0
    last_print = time.time()

    frame_count = 0
    byte_buffer = bytearray()

    while running:
        try:
            packet, addr = data_sock.recvfrom(UDP_READ_SIZE)
        except socket.timeout:
            now = time.time()
            if (now - last_print) >= PRINT_RATE_EVERY_S:
                if packet_count_total == 0:
                    print("[UDP] No data received yet...")
                else:
                    dt = now - last_print
                    pps = packet_count_interval / dt if dt > 0 else 0.0
                    bps = payload_bytes_interval / dt if dt > 0 else 0.0
                    print(f"[UDP] packets/sec: {pps:.1f} | payload bytes/sec: {bps:.0f} | total packets: {packet_count_total}")
                    packet_count_interval = 0
                    payload_bytes_interval = 0
                last_print = now
            continue

        got_any_data = True
        now = time.time()
        if first_packet_ts is None:
            first_packet_ts = now
        last_packet_ts = now

        packet_count_total += 1
        packet_count_interval += 1

        # Debug: show first bytes of first few packets
        if DEBUG_FIRST_N_PACKETS > 0 and packet_count_total <= DEBUG_FIRST_N_PACKETS:
            head = packet[:min(64, len(packet))].hex()
            print(f"[UDP][DEBUG] First bytes (pkt {packet_count_total}, len={len(packet)} from {addr}): {head}")

        # Strip header if present
        if len(packet) > DCA_DATA_HEADER_BYTES:
            payload = packet[DCA_DATA_HEADER_BYTES:]
        else:
            payload = b""

        payload_bytes_total += len(payload)
        payload_bytes_interval += len(payload)

        if PRINT_PACKET_EVERY and (packet_count_total % PRINT_PACKET_EVERY == 0):
            print(f"[UDP] Packet #{packet_count_total}: {len(packet)} bytes from {addr} (payload {len(payload)} bytes)")

        # Optional frame assembly
        if FRAME_BYTES > 0 and len(payload) > 0:
            byte_buffer.extend(payload)
            while len(byte_buffer) >= FRAME_BYTES:
                frame_count += 1
                print(f"[UDP] Frame #{frame_count} assembled ({FRAME_BYTES} bytes)")
                byte_buffer = byte_buffer[FRAME_BYTES:]

        # Rate print on schedule
        if (now - last_print) >= PRINT_RATE_EVERY_S:
            dt = now - last_print
            pps = packet_count_interval / dt if dt > 0 else 0.0
            bps = payload_bytes_interval / dt if dt > 0 else 0.0
            print(f"[UDP] packets/sec: {pps:.1f} | payload bytes/sec: {bps:.0f} | total packets: {packet_count_total}")
            packet_count_interval = 0
            payload_bytes_interval = 0
            last_print = now

    # ----- Shutdown sequence -----
    print("-" * 72)
    print("[STEP] Shutting down...")

    dca_stop(cfg_sock, cmds)
    awr_stop(RADAR_CLI_PORT, RADAR_CLI_BAUD)

    cfg_sock.close()
    data_sock.close()

    # ----- Summary -----
    print("-" * 72)
    print("[SUMMARY]")
    print(f"  DCA config OK:   {bool(dca_cfg_ok)}")
    print(f"  AWR UART OK:     {bool(awr_ok)}")
    print(f"  DCA START OK:    {bool(dca_start_ok)}")
    print(f"  Received UDP data: {got_any_data}")
    print(f"  Total UDP packets: {packet_count_total}")
    print(f"  Total payload bytes: {payload_bytes_total}")
    if FRAME_BYTES > 0:
        print(f"  Frames assembled: {frame_count}")

    if got_any_data and first_packet_ts and last_packet_ts and last_packet_ts > first_packet_ts:
        dur = last_packet_ts - first_packet_ts
        avg_pps = packet_count_total / dur
        avg_bps = payload_bytes_total / dur
        print(f"  Duration (first->last packet): {dur:.2f}s")
        print(f"  Avg packets/sec: {avg_pps:.2f}")
        print(f"  Avg payload bytes/sec: {avg_bps:.0f}")

    print("[DONE] Clean exit.")


if __name__ == "__main__":
    main()
