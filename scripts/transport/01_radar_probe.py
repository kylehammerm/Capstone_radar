import time
import serial
from serial.tools import list_ports

# Try the most common UART bauds used on TI/XDS setups
BAUDS = [115200, 921600]

# Commands that should cause readable output on a true text CLI
CMDS = [
    "help",
    "version",
    "thisIsNotACommand",
    "sensorStop",
    "sensorStart",
]

READ_WINDOW_S = 1.2
POLL_SLEEP_S = 0.01


def dump_bytes(data: bytes, hex_max_chars: int = 300, ascii_max_chars: int = 120) -> tuple[str, str]:
    """Return (hex_preview, ascii_preview)."""
    hx = data.hex(" ")
    if len(hx) > hex_max_chars:
        hx = hx[:hex_max_chars] + " ..."
    asc = "".join(chr(b) if 32 <= b <= 126 else "." for b in data)
    if len(asc) > ascii_max_chars:
        asc = asc[:ascii_max_chars] + " ..."
    return hx, asc


def read_for(s: serial.Serial, seconds: float) -> bytes:
    end = time.time() + seconds
    buf = bytearray()
    while time.time() < end:
        n = s.in_waiting
        if n:
            buf.extend(s.read(n))
        else:
            time.sleep(POLL_SLEEP_S)
    return bytes(buf)


def printable_ratio(b: bytes) -> float:
    if not b:
        return 0.0
    printable = sum(1 for x in b if 32 <= x <= 126 or x in (9, 10, 13))
    return printable / len(b)


def probe(port: str, baud: int) -> dict:
    result = {"port": port, "baud": baud, "ok": False, "responses": {}}

    try:
        with serial.Serial(port, baudrate=baud, timeout=0.1) as s:
            time.sleep(0.4)

            # Drain any boot/banner data
            s.reset_input_buffer()
            banner = read_for(s, 0.4)
            if banner:
                result["responses"]["<banner>"] = banner

            for cmd in CMDS:
                # Try CRLF first (common), then LF if CRLF yields nothing
                for ending in ("\r\n", "\n"):
                    s.write((cmd + ending).encode("ascii", errors="ignore"))
                    s.flush()
                    time.sleep(0.05)
                    data = read_for(s, READ_WINDOW_S)
                    if data:
                        result["responses"][f"{cmd}{ending.replace(chr(13),'\\\\r').replace(chr(10),'\\\\n')}"] = data
                        break  # don't try the other ending if we got something

            # Consider "ok" if we got any bytes back at all
            result["ok"] = any(len(v) > 0 for v in result["responses"].values())
            return result

    except Exception as e:
        result["error"] = repr(e)
        return result


def main():
    ports = list(list_ports.comports())
    if not ports:
        print("No COM ports found.")
        return

    print("Detected COM ports:")
    for p in ports:
        print(f"  {p.device:6}  {p.description}")

    print("\n--- Probing each port (this may take a minute) ---")
    candidates = []

    for p in ports:
        for baud in BAUDS:
            print(f"\n=== {p.device} @ {baud} ===")
            res = probe(p.device, baud)

            if "error" in res:
                print(f"  could not open: {res['error']}")
                continue

            if not res["ok"]:
                print("  no response bytes")
                continue

            # Print responses
            for k, data in res["responses"].items():
                hx, asc = dump_bytes(data)
                pr = printable_ratio(data)
                txt = data.decode("ascii", errors="ignore").strip()
                txt_preview = txt[:80].replace("\n", "\\n").replace("\r", "\\r")
                print(f"  [{k}] {len(data)} bytes | printable={pr:.2f}")
                print(f"    HEX  : {hx}")
                print(f"    ASCII: {asc}")
                if txt_preview:
                    print(f"    ASCII(clean): {txt_preview}")

            # Heuristic: likely text CLI if printable ratio is high for any command
            best_pr = max(printable_ratio(b) for b in res["responses"].values())
            if best_pr > 0.5:
                candidates.append((p.device, baud, best_pr))

    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        print("\n=== Likely TEXT CLI candidates (highest printable ratio) ===")
        for dev, baud, pr in candidates[:10]:
            print(f"  {dev} @ {baud} (printable_ratio={pr:.2f})")
    else:
        print("\nNo ports produced readable output. That usually means the CLI firmware isn't running,")
        print("or mmWave Studio/another process still has the ports open.")


if __name__ == "__main__":
    main()
