import time
import serial

PORTS = ["COM9", "COM10"]
BAUD = 115200

def sniff(port: str, seconds: float = 5.0):
    print(f"\n=== Sniffing {port} @ {BAUD} for {seconds}s ===")
    try:
        with serial.Serial(port, BAUD, timeout=0.05, rtscts=False, dsrdtr=False) as s:
            try:
                s.setDTR(True)
                s.setRTS(False)
            except Exception:
                pass

            t0 = time.time()
            buf = bytearray()
            while time.time() - t0 < seconds:
                chunk = s.read(4096)
                if chunk:
                    buf += chunk

            if not buf:
                print("  (no bytes)")
                return

            print(f"  got {len(buf)} bytes")
            print("  HEX :", bytes(buf[:80]).hex(" "), "..." if len(buf) > 80 else "")
            print("  ASCII:", bytes(buf[:400]).decode("ascii", errors="replace"))
    except Exception as e:
        print("  open failed:", e)

if __name__ == "__main__":
    for p in PORTS:
        sniff(p, 5.0)
