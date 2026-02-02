import time
import serial

PORT = "COM8"
BAUD = 115200

def read_all(s, seconds=1.5):
    end = time.time() + seconds
    buf = b""
    while time.time() < end:
        chunk = s.read(4096)
        if chunk:
            buf += chunk
        time.sleep(0.05)
    return buf

with serial.Serial(PORT, BAUD, timeout=0.2) as s:
    time.sleep(0.5)
    s.reset_input_buffer()

    print("Listening for any boot/prompt text (1.5s)...")
    boot = read_all(s, 1.5)
    print("BOOT RAW:", boot)
    print("BOOT TXT:", boot.decode(errors="ignore"))
    print("-"*60)

    # Try a set of commands with longer waits
    cmds = [b"\r\n", b"\n", b"\r",
            b"version\r\n", b"version\n",
            b"help\r\n", b"?\r\n",
            b"sensorStop\r\n", b"sensorStart\r\n",
            b"cfg\r\n", b"status\r\n"]

    for cmd in cmds:
        s.reset_input_buffer()
        s.write(cmd)
        s.flush()
        time.sleep(0.1)
        resp = read_all(s, 1.0)
        print("CMD:", repr(cmd))
        print("RAW:", resp)
        print("TXT:", resp.decode(errors="ignore"))
        print("-"*60)
