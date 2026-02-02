import time, serial

PORT="COM8"
BAUD=115200

with serial.Serial(PORT, BAUD, timeout=0.6) as s:
    time.sleep(0.2)
    s.reset_input_buffer()
    s.write(b"help procChain\r\n")
    time.sleep(0.4)
    print(s.read(20000).decode(errors="ignore"))

    s.reset_input_buffer()
    s.write(b"procChain\r\n")
    time.sleep(0.4)
    print(s.read(20000).decode(errors="ignore"))
