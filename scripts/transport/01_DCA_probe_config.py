import socket
import time

# ==============================
# NETWORK SETTINGS (DCA1000)
# ==============================
HOST_IP = "192.168.33.30"
HOST_CONFIG_PORT = 4096

DCA_IP = "192.168.33.180"
DCA_CONFIG_PORT = 4096

# ==============================
# DCA1000 COMMAND PACKETS
# ==============================
def make_cmd(code_hex, data=b""):
    """
    Build DCA1000 command packet
    [0xA55A][Code][Len][Data][0xEEAA]
    """
    header = (0xA55A).to_bytes(2, "little")
    footer = (0xEEAA).to_bytes(2, "little")
    code = int(code_hex, 16).to_bytes(2, "little")
    length = len(data).to_bytes(2, "little")
    return header + code + length + data + footer

def dca_startup_commands():
    return {
        "CONNECT": make_cmd("09"),
        "READ_FPGA": make_cmd("0E"),
        "CONFIG_FPGA": make_cmd("03", (0x01020102031E).to_bytes(6, "big")),
        "CONFIG_PACKET": make_cmd("0B", (0xC005350C0000).to_bytes(6, "big")),
        "START": make_cmd("05"),
        "STOP": make_cmd("06"),
    }

# ==============================
# MAIN
# ==============================
def main():
    print("DCA1000 Startup Only")

    cmds = dca_startup_commands()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST_IP, HOST_CONFIG_PORT))
    sock.settimeout(1.0)

    def send_cmd(name):
        print(f"Sending: {name}")
        sock.sendto(cmds[name], (DCA_IP, DCA_CONFIG_PORT))
        try:
            data, _ = sock.recvfrom(2048)
            print(f"ACK ({name}): {data.hex()}")
        except socket.timeout:
            print(f"No ACK for {name}")
        time.sleep(0.2)

    # Standard DCA bring-up sequence
    send_cmd("CONNECT")
    send_cmd("READ_FPGA")
    send_cmd("CONFIG_FPGA")
    send_cmd("CONFIG_PACKET")
    send_cmd("START")

    print("DCA1000 should now be streaming (if radar is running)")
    sock.close()

if __name__ == "__main__":
    main()
