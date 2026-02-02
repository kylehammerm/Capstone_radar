import time, serial

PORT="COM8"; BAUD=115200
with serial.Serial(PORT, BAUD, timeout=0.5) as s:
    time.sleep(0.2)
    s.reset_input_buffer()
    s.write(b"help channelCfg\r\n")
    time.sleep(0.3)
    print(s.read(12000).decode(errors="ignore"))