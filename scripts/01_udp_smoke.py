import socket
import time 
import signal
import serial 

# ==============================
# NETWORK SETTINGS (DCA1000)
# ==============================
HOST_IP = "192.168.33.30"
HOST_DATA_PORT = 4098
HOST_CONFIG_PORT = 4096

DCA_IP = "192.168.33.180"
DCA_CONFIG_PORT = 4096

# ==============================
# RADAR UART SETTINGS (AWR2944)
# ==============================
RADAR_CLI_PORT = "COM7"              
RADAR_CLI_BAUD = 115200
RADAR_CFG_FILE = r".\config\awr2944.cfg"

# ==============================
# FRAME PARAMS (TEMP HARD-CODE)
# ==============================
ADC_SAMPLES = 64
CHRIPS_PER_FRAME = 32
RX_CHANNELS = 4
TX_CHANNELS = 1

BYTES_PER_SAMPLE = 4

FRAME_BYTES = (ADC_SAMPLES*CHRIPS_PER_FRAME*RX_CHANNELS*TX_CHANNELS*BYTES_PER_SAMPLE)

UDP_BUFFER_SIZE = 2*1024*1024

# ==============================
# DCA1000 COMMAND PACKETS
# ==============================

def make_cmd(code_hex, data=b""):
    """
    This Builds a DCA1000 command packet
    Format in bytes
    [0xA55A][Code][Len][Data][0xEEAA]
    """
    # A5 5A
    header = (0xA55A).to_bytes(2,"little")
    footer = (0xEEAA).to_bytes(2,"little")

    code = int(code_hex,16).to_bytes(2,"little")
    length = len(data).to_bytes(2,"little")

    return header + code + length + data + footer

def dca_startup_commands():
    """
    Standard set of startup commands for the DCA1000
    Returned in Hash form to know which one is doing what
    """
    return {
        "CONNECT": make_cmd("09"),
        "READ_FPGA": make_cmd("0E"),
        "CONFIG_FPGA": make_cmd("03", (0x01020102031E).to_bytes(6, "big")),
        "CONFIG_PACKET": make_cmd("0B", (0xC005350C0000).to_bytes(6, "big")),
        "START": make_cmd("05"),
        "STOP": make_cmd("06"),
    }

# ==============================
# CLEAN SHUTDOWN
# ==============================

running = True

def handle_exit(sig,frame):
    global running
    print("\n Stopping")
    running = False

signal.signal(signal.SIGINT,handle_exit)

# ==============================
# MAIN
# ==============================

def main():
    print("DCA UDP Test")
    print(f"Expecting {FRAME_BYTES} bytes")

    cmds = dca_startup_commands()

    # Config socket (control channel)
    cfg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cfg_sock.bind((HOST_IP,HOST_CONFIG_PORT))
    cfg_sock.settimeout(1.0)

    # Data socket 
    data_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    data_sock.bind((HOST_IP,HOST_DATA_PORT))
    data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UDP_BUFFER_SIZE)
    data_sock.settimeout(1.0)

    def send_cmd(name):
        print(f"Sending: {name}")
        cfg_sock.sendto(cmds[name], (DCA_IP, DCA_CONFIG_PORT))
        try:
            data, _ = cfg_sock.recvfrom(2048)
            print(f"ACK ({name}):{data.hex()}")
        except socket.timeout:
            print(f"No ACK for {name}")
        
    send_cmd("CONNECT")
    send_cmd("READ_FPGA")
    send_cmd("CONFIG_FPGA")
    send_cmd("CONFIG_PACKET")
    send_cmd("START")

    print("\n Listening for ADC data... \n")

    #_______________________________________
    # Packet recive loop

    byte_buffer = bytearray()
    frame_count = 0
    packet_count = 0

    last_time = time.time()
    bytes_this_sec = 0

    while running:
        try:
            packet,addr = data_sock.recvfrom(UDP_BUFFER_SIZE)
        except socket.timeout:
            print(f"No data recived at {time.time()}")
            continue

        packet_count+=1

        # get rid of header
        payload = packet[10:]

        byte_buffer.extend(payload)
        bytes_this_sec += len(payload)

        now = time.time()

        if now - last_time >=1.0:
            print(f"Packets/sec: {packet_count} | Bytes/sec: {bytes_this_sec}")
            packet_count = 0
            bytes_this_sec = 0
            last_time = now
        
        while len(byte_buffer) >= FRAME_BYTES:
            frame_count += 1
            print(f"Frame #{frame_count} assembled ({FRAME_BYTES} bytes)")

            # Remove frame from buffer
            byte_buffer = byte_buffer[FRAME_BYTES:]

    print("Stopping DCA1000 stream...")
    cfg_sock.sendto(cmds["STOP"], (DCA_IP, DCA_CONFIG_PORT))

    cfg_sock.close()
    data_sock.close()

    print("Clean exit.")

if __name__ =="__main__":
    main() 