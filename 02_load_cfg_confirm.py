import time
import serial
import pathlib

PORT = "COM8"
BAUD = 115200
CFG_PATH = r".\config\awr2944_cfg.cfg"   # change if needed

def read_all(s, wait=0.08):
    time.sleep(wait)
    data = s.read(16384)
    return data.decode(errors="ignore")

def send(s, line, wait=0.08):
    s.write((line + "\r\n").encode("ascii", errors="ignore"))
    s.flush()
    return read_all(s, wait)

def main():
    cfg_file = pathlib.Path(CFG_PATH)
    if not cfg_file.exists():
        raise SystemExit(f"CFG file not found: {cfg_file.resolve()}")

    print(f"Opening {PORT} @ {BAUD}")
    with serial.Serial(PORT, BAUD, timeout=0.2) as s:
        time.sleep(0.3)
        s.reset_input_buffer()

        # wake prompt
        print(send(s, "", 0.15))

        # stop
        resp = send(s, "sensorStop", 0.2)
        print(resp)

        # send cfg
        print(f"Loading cfg: {cfg_file.resolve()}")
        with cfg_file.open("r", encoding="utf-8", errors="ignore") as f:
            for i, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith("%") or line.startswith("#"):
                    continue

                s.reset_input_buffer()
                resp = send(s, line, 0.03)

                # Print only when there is something interesting
                if resp.strip():
                    print(f"[line {i}] {line}")
                    print(resp.strip())

                if "Error" in resp or "Unknown" in resp or "not recognized" in resp:
                    raise SystemExit(f"\n❌ FAILED at line {i}: {line}\nResponse:\n{resp}")

        # start
        resp = send(s, "sensorStart", 0.3)
        print("\n=== sensorStart response ===")
        print(resp)

        if "Error" in resp:
            raise SystemExit("❌ sensorStart failed (cfg incomplete/invalid)")

        print("\n✅ Config loaded and sensor started.")

if __name__ == "__main__":
    main()
