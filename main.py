import time, serial

PORT="COM8"
BAUD=115200

with serial.Serial(PORT, BAUD, timeout=0.2) as s:
    time.sleep(0.2)
    s.reset_input_buffer()

    # send a few enters
    for _ in range(3):
        s.write(b"\r\n")
        time.sleep(0.2)

    # try common mmWave demo command
    s.write(b"sensorStop\r\n")
    time.sleep(0.4)

    data = s.read(4096)
    print("RAW:", data)
    print("TXT:", data.decode(errors="ignore"))
