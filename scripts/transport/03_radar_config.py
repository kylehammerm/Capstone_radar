import time
import serial
from pathlib import Path

# ==============================
# EDIT THESE
# ==============================
PORT = "COM10"        # CLI UART COM port (e.g. "COM10")
BAUD = 115200        # CLI UART baud rate
CFG_FILE = r".\config\awr2944_cfg.cfg"

# Optional tuning
CHAR_DELAY_S = 0.0015   # per-character delay (slow down if bytes drop)
START_WAIT_S = 0.25    # how long to let sensor run before stopping

# ==============================
# INTERNALS
# ==============================
PROMPT = b"mmwDemo:/>"

def drain_read(s: serial.Serial, seconds: float = 0.15) -> bytes:
    end = time.time() + seconds
    out = b""
    while time.time() < end:
        chunk = s.read(4096)
        if chunk:
            out += chunk
        else:
            time.sleep(0.01)
    return out

def read_until_prompt(s: serial.Serial, timeout_s: float = 2.5) -> bytes:
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

def send_line_prompted(
    s: serial.Serial,
    line: str,
    *,
    char_delay_s: float,
    post_line_delay_s: float = 0.02,
    prompt_timeout_s: float = 2.5,
) -> bytes:
    drain_read(s, 0.10)

    payload = (line + "\r\n").encode("ascii", errors="ignore")

    # Slow, per-character write to avoid dropped bytes on mmWave demo CLI
    for b in payload:
        s.write(bytes([b]))
        s.flush()
        time.sleep(char_delay_s)

    time.sleep(post_line_delay_s)
    return read_until_prompt(s, timeout_s=prompt_timeout_s)

def fmt_bytes(b: bytes, max_len: int = 300) -> str:
    if len(b) <= max_len:
        return repr(b)
    return repr(b[:max_len]) + f"... (+{len(b)-max_len} bytes)"

def looks_bad(resp: bytes) -> bool:
    t = resp.decode(errors="ignore")
    bad_markers = [
        "Error",
        "not recognized",
        "Invalid usage",
        "Unknown command",
        "Usage:",
    ]
    return any(m in t for m in bad_markers)

# ==============================
# MAIN LOGIC
# ==============================
def main():
    cfg_abs = str(Path(CFG_FILE).resolve())
    if not Path(cfg_abs).exists():
        print(f"[FATAL] cfg file not found: {cfg_abs}")
        return

    print(f"[INFO] Opening CLI UART: {PORT} @ {BAUD}")
    print(f"[INFO] Using cfg: {cfg_abs}")
    print("-" * 72)

    cfg_sent = 0
    cfg_problems = 0

    with serial.Serial(PORT, baudrate=BAUD, timeout=0.2) as s:
        time.sleep(0.6)
        s.reset_input_buffer()
        s.reset_output_buffer()

        # Sync prompt
        print("[STEP] Prompt sync")
        resp = send_line_prompted(
            s, "", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=3.0
        )
        print("  RAW:", fmt_bytes(resp))
        print("  TXT:\n", resp.decode(errors="ignore"), sep="")
        print("-" * 72)

        # Safe stop
        print("[STEP] sensorStop (pre-stop)")
        resp = send_line_prompted(
            s, "sensorStop", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=3.0
        )
        print("  RAW:", fmt_bytes(resp))
        print("  TXT:\n", resp.decode(errors="ignore"), sep="")
        print("-" * 72)

        # Send cfg file
        print("[STEP] Sending cfg lines...")
        with open(cfg_abs, "r", encoding="utf-8", errors="ignore") as f:
            for i, raw in enumerate(f, start=1):
                line = raw.strip()

                if not line:
                    continue
                if line.startswith("%") or line.startswith("#"):
                    continue
                if line.lower().startswith("sensorstart"):
                    print(f"[CFG] Skipping line {i} (sensorStart in file)")
                    continue

                cfg_sent += 1
                print(f"[CFG] ({cfg_sent}) line {i}: {line}")

                resp = send_line_prompted(
                    s, line, char_delay_s=CHAR_DELAY_S, prompt_timeout_s=3.0
                )

                print("  RAW:", fmt_bytes(resp))
                txt = resp.decode(errors="ignore")
                print("  TXT:\n", txt, sep="")

                if looks_bad(resp):
                    cfg_problems += 1
                    print("  [WARN] response looked suspicious for this line.")

                print("-" * 72)

        print(f"[INFO] cfg lines sent: {cfg_sent} | suspicious lines: {cfg_problems}")
        print("-" * 72)

        # Start sensor
        print("[STEP] sensorStart")
        resp = send_line_prompted(
            s, "sensorStart", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=4.0
        )
        print("  RAW:", fmt_bytes(resp))
        print("  TXT:\n", resp.decode(errors="ignore"), sep="")
        print("-" * 72)

        if START_WAIT_S > 0:
            print(f"[STEP] wait {START_WAIT_S:.2f}s")
            time.sleep(START_WAIT_S)
            print("-" * 72)

        # Status check
        print("[STEP] queryDemoStatus")
        resp = send_line_prompted(
            s, "queryDemoStatus", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=3.0
        )
        print("  RAW:", fmt_bytes(resp))
        print("  TXT:\n", resp.decode(errors="ignore"), sep="")
        print("-" * 72)

        # Stop sensor
        print("[STEP] sensorStop (shutdown)")
        resp = send_line_prompted(
            s, "sensorStop", char_delay_s=CHAR_DELAY_S, prompt_timeout_s=3.0
        )
        print("  RAW:", fmt_bytes(resp))
        print("  TXT:\n", resp.decode(errors="ignore"), sep="")
        print("-" * 72)

    print("[DONE] Closed UART. Smoke test complete.")
    if cfg_problems > 0:
        print("[RESULT] Completed, but some cfg lines looked suspicious.")
    else:
        print("[RESULT] Completed with no obvious cfg-line errors.")

if __name__ == "__main__":
    main()
