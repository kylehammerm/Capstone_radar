This is a **strict, step-by-step procedure** to obtain **near‑live Doppler / micro‑Doppler visualizations** using **mmWave Studio 2G + MATLAB** with **AWR2944EVM + DCA1000**. Follow the steps **in order**. Nothing here assumes true real-time streaming.

---

# PHASE 0 — What You Are Building (Lock This In)

You are implementing:

* Short FMCW captures (100–500 ms)
* Immediate MATLAB post‑processing
* Continuous repetition → looks live

This is **batch‑live**, not streaming.

---

# PHASE 1 — Radar & Capture Configuration (mmWave Studio)

## Step 1. Launch mmWave Studio 2G

* Run **as Administrator**
* Confirm MATLAB Runtime loads
* Select **AWR2944 platform**

---

## Step 2. Connect to the Radar

**UART Tab**

* Control Port: COMx
* Data Port: COMy
* Baud Rate: 115200
* Click **Connect** → SUCCESS

---

## Step 3. Flash Firmware (if not already done)

**Firmware Tab**

1. Flash **RadarSS** image
2. Flash **MSS** image
3. Power cycle both boards

---

## Step 4. Enable RF

**Device Control Tab**

1. Click **Connect**
2. Click **Enable RF**

Do NOT proceed if RF fails.

---

# PHASE 2 — FMCW Parameter Setup (Critical)

## Step 5. Profile Configuration

Use conservative values:

* Start Frequency: 77 GHz
* Bandwidth: 1–2 GHz
* Chirp Time: 50 µs
* ADC Samples: **256**
* ADC Sampling Rate: 5 Msps

Click **Set**

---

## Step 6. Chirp Configuration

* Enable TX1 only
* Enable RX1–RX4
* Chirps per Frame: **64**

Click **Set**

---

## Step 7. Frame Configuration

* Frames per Capture: **1**
* Frame Periodicity: 40 ms
* Trigger Mode: Software

Click **Set**

---

# PHASE 3 — DCA1000 Setup

## Step 8. DCA1000 Configuration

**DCA1000 Tab**

* Capture Mode: Raw ADC
* Data Format: Complex
* LVDS Lanes: 4
* Packet Delay: 25–40 µs

Click **Configure** → SUCCESS

---

## Step 9. File Output Settings

* Enable **Save to File**
* Choose a fixed output directory
* Enable overwrite OR incremental filenames

Remember this directory.

---

# PHASE 4 — MATLAB Preparation

## Step 10. Prepare MATLAB Scripts

You need **one master script** that:

1. Watches a directory
2. Loads newest `.bin`
3. Processes ADC
4. Updates Doppler plot

Do NOT start MATLAB yet.

---

# PHASE 5 — Capture → Process Loop

## Step 11. Start MATLAB

* Open MATLAB
* Navigate to your processing folder
* Run your **live_doppler.m** script

MATLAB should now be waiting for a file.

---

## Step 12. Capture Sequence (STRICT ORDER)

In mmWave Studio:

1. Click **Arm DCA1000**
2. Click **Start Sensor**
3. Wait 100–300 ms
4. Click **Stop Sensor**
5. Click **Stop DCA1000**

A `.bin` file is written.

---

## Step 13. MATLAB Auto‑Processing

Immediately after file appears:

* MATLAB loads ADC
* Performs FFT processing
* Updates Doppler / µD plot

You now see a **live‑updated image**.

---

## Step 14. Repeat for Continuous Viewing

Repeat **Steps 12–13** continuously.

For better results:

* Capture 200 ms windows
* Use consistent gestures

---

# PHASE 6 — MATLAB Signal Processing (What Happens Internally)

## Step 15. ADC Parsing

* `.bin` → int16
* Interleaved I/Q
* Reshape → [chirps × RX × samples]

---

## Step 16. Range FFT

* FFT over ADC samples
* Output: range bins × chirps

---

## Step 17. Doppler FFT

* FFT over chirps
* FFT shift applied
* Output: Doppler × range

---

## Step 18. Micro‑Doppler (Optional)

* Select active range bins
* Stack Doppler slices over time
* Apply STFT / sliding FFT

Produces µD spectrogram.

---

# PHASE 7 — Visualization

## Step 19. Plotting

* Use `imagesc`
* Log magnitude scaling
* `drawnow` for updates

Latency: ~0.3–1 s

---

# PHASE 8 — Automation Options

## Option A: Manual (Most Stable)

* Human clicks
* Lowest failure rate

## Option B: Lua Script (Semi‑Auto)

* Loop start/stop sensor
* Fixed capture duration

MATLAB still processes each file.

---

# PHASE 9 — Validation Checklist

You are done when:

* `.bin` files are non‑zero
* Doppler image updates every capture
* Motion is visible in Doppler axis

---

# PHASE 10 — Common Failures

| Problem      | Cause            | Fix             |
| ------------ | ---------------- | --------------- |
| Empty `.bin` | DCA not armed    | Arm first       |
| Corrupt plot | Wrong reshape    | Check RX order  |
| Slow updates | Capture too long | Reduce duration |
| No Doppler   | Too few chirps   | Increase to 64  |

---

# FINAL STATE

✔ Near‑live Doppler / µD visualization
✔ mmWave Studio used correctly
✔ MATLAB processing pipeline validated
✔ Ready to align µD with CNN input

---

Next logical step:

* Feed MATLAB µD output into CNN (offline validation)
* Then migrate to Python streaming
