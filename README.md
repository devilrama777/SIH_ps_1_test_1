# ISRO PS 26172: Low Latency & Efficient Voice Activator for Edge Devices ("Aura")

An ultra-lightweight, high-accuracy TinyML Keyword Spotting (KWS) and audio streaming activator designed for edge microcontrollers (ESP32) and ground/terminal stations under strict resource boundaries.

---

## Key Metrics & Evaluation vs. ISRO Boundaries

| Parameter | ISRO Boundary | Aura Measured Value | Result |
| :--- | :--- | :--- | :---: |
| **Compute Latency** | $< 0.9\text{ ms}$ per hop | **$0.18\text{ ms} - 0.50\text{ ms}$** | **PASS** (Exceeds goal) |
| **RAM Footprint** | $< 256\text{ KB}$ | **$36.8\text{ KB}$** (Static tensor arena) | **PASS** (85% free headroom) |
| **Idle CPU Utilization**| $< 10\%$ during listening | **$0.00\% - 1.2\%$** | **PASS** (Ultra-low power) |
| **Frameworks** | Open-source TinyML only | Custom C++ DSP + INT8 DS-CNN Runtime | **PASS** (Zero proprietary SDKs) |
| **Keyword Spotting** | Custom Keyword | **"Aura"** (Wake) & **"Sleep" / "Exit"** (Stop) | **PASS** |
| **Hardware Display** | Telemetry Display & LEDs | 0.96" SSD1306 OLED + 3-LED Indicator Bar | **PASS** |

---

## System Architecture

```
[Audio Ingestion (16kHz PCM)]
           │
           ▼
[DSP Mel-Spectrogram] ──► 512-pt Radix-2 FFT with Precomputed Twiddle Tables + 16 Mel Filterbanks
           │
           ▼
[TinyKWS INT8 Engine] ──► Depthwise Separable CNN (DS-CNN) (~2,500 parameters, 10.3 KB C-Array)
           │
           ▼
[State Machine (Debounced)]
   ├── SLEEPING: Commands & noise 100% ignored. Wakes ONLY on "Aura" (2-frame debounce)
   ├── WAKE_AURA: Green LED illuminates (1.2s acknowledgement)
   ├── LISTENING_COMMAND: Blue LED illuminates. Streams continuous audio until explicit "Sleep" / "Exit"
   └── COMMAND_DONE: Red LED illuminates. Returns cleanly to SLEEPING state
```

---

## Repository Structure

```
├── .gitignore               # Excludes binaries, build artifacts, and raw audio files
├── README.md                # Project documentation & benchmark report
├── hardware/
│   ├── esp32_firmware.ino   # Production-ready ESP32 Arduino/ESP-IDF firmware sketch
│   └── WIRING_AND_SETUP.md  # Complete hardware BOM, schematic, and pin connection table
├── include/
│   ├── kws_engine.h         # Zero-allocation embedded INT8 neural network engine header
│   ├── mfcc.h               # High-speed Mel spectrogram & twiddle-optimized FFT DSP header
│   ├── miniaudio.h          # Single-header cross-platform audio capture library
│   ├── model_data.h         # Pre-trained, quantized INT8 weights and scale factors (10.3 KB)
│   └── telemetry.h          # Microsecond latency, RAM working set, and CPU percent monitor
├── scripts/
│   ├── generate_dataset.py  # SAPI synthetic speech generator & augmentor (2,000+ samples)
│   └── train_kws_model.py   # PyTorch DS-CNN training pipeline & INT8 C-header exporter
└── src/
    ├── kws_engine.cpp       # Quantized neural net forward inference implementation
    ├── main.cpp             # Terminal application: live mic streaming, state machine, OLED UI
    ├── mfcc.cpp             # Fast DSP Mel Spectrogram implementation
    └── telemetry.cpp        # Real-time latency, memory, and CPU utilization monitor
```

---

## Quick Start (Terminal Application on PC)

### Prerequisites:
* C++17 compiler (`g++` or `clang++` or `MSVC`)
* Python 3.10+ (optional, only needed for re-training)

### Compilation:
```bash
g++ -O3 -std=c++17 -Iinclude src/main.cpp src/mfcc.cpp src/kws_engine.cpp src/telemetry.cpp -o aura_agent.exe -lole32 -lwinmm -lpsapi
```

### Running:
```bash
./aura_agent.exe
```

1. **At Launch (`ASLEEP`):** All commands like *"What is the time"* are completely ignored.
2. **Say `"Aura"`:** The **Green LED** illuminates (`WAKE DETECTED`), transitioning to **Blue LED** (`ACTIVE COMMAND LISTENER`).
3. **Continuous Listening:** Aura stays awake continuously, streaming audio commands to the cloud.
4. **Say `"Sleep"`, `"Exit"`, or `"Terminate"`:** The **Red LED** illuminates and Aura returns to sleep (or closes on Exit).
5. **Quit Anytime:** Press `Q` or `Ctrl+C`.

---

## Hardware Deployment (ESP32)

Refer to [`hardware/WIRING_AND_SETUP.md`](hardware/WIRING_AND_SETUP.md) for full circuit diagrams:
* **Microcontroller:** ESP32 DevKit V1 (240MHz, 520KB SRAM)
* **Microphone:** INMP441 I2S Digital Microphone (`SCK:14`, `WS:15`, `SD:32`)
* **Display:** 0.96" SSD1306 I2C OLED (`SDA:21`, `SCL:22`)
* **Status LEDs:** Green (`GPIO 18`), Blue (`GPIO 19`), Red (`GPIO 23`)
* **Firmware:** Flash [`hardware/esp32_firmware.ino`](hardware/esp32_firmware.ino) using the Arduino IDE or ESP-IDF.
