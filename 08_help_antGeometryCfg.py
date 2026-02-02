import time
import serial

PORT = "COM8"
BAUD = 115200

def read_all(s, seconds=0.6):
    end = time.time() + seconds
    buf = bytearray()
    while time.time() < end:
        n = s.in_waiting
        if n:
            buf.extend(s.read(n))
        else:
            time.sleep(0.01)
    return bytes(buf)

with serial.Serial(PORT, baudrate=BAUD, timeout=0.1) as s:
    time.sleep(0.3)
    s.reset_input_buffer()

    s.write(b"\r\n")
    time.sleep(0.1)
    read_all(s, 0.2)

    s.write(b"help antGeometryCfg\r\n")
    s.flush()
    out = read_all(s, 1.0)

print(out.decode("ascii", errors="ignore"))
