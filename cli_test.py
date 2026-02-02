import time
import serial

PORT = "COM8"      # CLI port from Device Manager
BAUD = 115200      # CLI baud is usually 115200

commands = [
    b"\r\n",
    b"version\r\n",
    b"help\r\n",
    b"?\r\n",
    b"sensorStop\r\n",
]

with serial.Serial(PORT, BAUD, timeout=1.0) as s:
    time.sleep(0.2)
    s.reset_input_buffer()

    for cmd in commands:
        s.write(cmd)
        s.flush()
        time.sleep(0.3)
        data = s.read(4096)

        print("CMD:", cmd)
        print("RAW:", data)
        print("TXT:", data.decode(errors="ignore"))
        print("-" * 60)
