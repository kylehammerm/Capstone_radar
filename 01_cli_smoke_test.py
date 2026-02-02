import time, serial

PORT="COM8"
BAUD=115200

def send(s, cmd):
    s.write((cmd + "\r\n").encode())
    s.flush()
    time.sleep(0.25)
    return s.read(8192)

with serial.Serial(PORT, BAUD, timeout=0.3) as s:
    time.sleep(0.2)
    s.reset_input_buffer()

    # Wake/prompt
    s.write(b"\r\n"); s.flush()
    time.sleep(0.2)
    print("BOOT:", s.read(4096).decode(errors="ignore"))

    # Test commands
    for cmd in ["sensorStop", "sensorStart"]:
        s.reset_input_buffer()
        resp = send(s, cmd)
        print(f"\nCMD: {cmd}")
        print(resp.decode(errors="ignore"))
