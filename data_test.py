import time
import serial

PORT = "COM7"         # Auxiliary/Data port
BAUD = 921600         # Common data baud; if 0 bytes, we can try others

with serial.Serial(PORT, BAUD, timeout=0.2) as s:
    time.sleep(0.2)
    s.reset_input_buffer()

    t0 = time.time()
    total = 0
    while time.time() - t0 < 3.0:
        chunk = s.read(8192)
        total += len(chunk)

    print(f"Received {total} bytes in 3 seconds on {PORT} @ {BAUD}")
