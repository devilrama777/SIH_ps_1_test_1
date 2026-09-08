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
| **Keyword Spotting** | Custom Keyword | **"Aura"** (Wake) & **"Exit"** (Sleep) | **PASS** |
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
[State Machine (Debounced with VAD Sentence Capture)]
   ├── SLEEPING: Hums, noise & commands 100% ignored. Wakes ONLY on "Aura" (RMS gated + 2-frame debounce)
   ├── WAKE_AURA: Green LED illuminates (1.0s acknowledgement)
   ├── LISTENING_COMMAND: Blue LED illuminates. Uses VAD to capture FULL sentence without cutting off
   ├── THINKING_ANALYSING: Blue LED flashes. Telemetry displayed. Immediately loops back to LISTENING_COMMAND
   └── COMMAND_DONE: Red LED illuminates when "Exit" is spoken. Returns to SLEEPING state
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
│   ├── generate_dataset.py        # SAPI synthetic speech generator & augmentor (Windows)
│   ├── generate_indian_dataset.py # Neural Indian voice dataset generator (cross-platform, edge-tts)
│   ├── record_voice_samples.py    # Interactive microphone voice recorder for custom user keywords
│   └── train_kws_model.py         # PyTorch DS-CNN training pipeline & INT8 C-header exporter
└── src/
    ├── kws_engine.cpp       # Quantized neural net forward inference implementation
    ├── main.cpp             # Terminal application: live mic streaming, state machine, OLED UI
    ├── mfcc.cpp             # Fast DSP Mel Spectrogram implementation
    └── telemetry.cpp        # Real-time latency, memory, and CPU utilization monitor
```

---

## Quick Start (Terminal Application on PC)

### Prerequisites:
* C++14/C++17 compiler (`g++`, `clang++`, or `MSVC`)
* CMake 3.14+ or Make (optional, direct compiler commands also provided)
* Python 3.10+ (optional, only needed for re-training)

### Compilation:

#### Option A: Using CMake (Recommended across all OS)
```bash
cmake -B build
cmake --build build --config Release
```

#### Option B: Using Make
```bash
make
```

#### Option C: Direct Compiler Command

* **Windows (MinGW / GCC) - Standalone binary with static runtime (runs on any Windows desktop):**
  ```bash
  g++ -O3 -std=c++17 -Iinclude src/main.cpp src/mfcc.cpp src/kws_engine.cpp src/telemetry.cpp -o aura.exe -static -static-libgcc -static-libstdc++ -lole32 -lwinmm -lpsapi
  ```

* **Linux (Ubuntu / Debian / Fedora):**
  ```bash
  g++ -O3 -std=c++17 -Iinclude src/main.cpp src/mfcc.cpp src/kws_engine.cpp src/telemetry.cpp -o aura -lpthread -ldl -lm
  ```

* **macOS (Apple Clang / GCC):**
  ```bash
  g++ -O3 -std=c++17 -Iinclude src/main.cpp src/mfcc.cpp src/kws_engine.cpp src/telemetry.cpp -o aura -framework CoreAudio -framework AudioToolbox -framework CoreFoundation -lpthread -lm
  ```

### Running:
```bash
# Windows:
./aura.exe

# Linux / macOS:
./aura
```

1. **At Launch (`ASLEEP`):** All conversation, background hums, and commands are 100% ignored.
2. **Say `"Aura"`:** The **Green LED** illuminates (`WAKE DETECTED`), transitioning to **Blue LED** (`READY FOR COMMAND`).
3. **Speak Full Command:** Say your complete question (e.g. *"What is the time"* or *"What is today's date"*).
4. **Thinking & Analysing:** Blue LED flashes (`THINKING / ANALYSING`). The system executes the command and immediately loops back to listening for your next command.
5. **Continuous Commands:** Speak command after command without needing to wake Aura up again.
6. **Put Aura to Sleep:** Say **`"Exit"`** $\to$ The **Red LED** illuminates and Aura returns to dormant sleep.
7. **Quit Anytime:** Press `Q` or `Ctrl+C`.

---

## Model Training & Custom Keyword Fine-Tuning

Aura comes with pre-trained INT8 weights trained across 2,580+ diverse acoustic samples. You can retrain or add your own voice samples anytime:

### 1. (Optional) Record Your Own Voice Samples
Record your own voice saying "Aura" and "Exit" to tailor the activator to your exact pitch and accent:
```bash
python scripts/record_voice_samples.py
```
This prompts you to speak 5 samples of "Aura" and "Exit", saving them into `data/aura/` and `data/exit/`.

### 2. Generate Full Neural & Augmented Dataset
To regenerate the full multi-accent Indian neural dataset with realistic noise and phonetic lookalikes:
```bash
python scripts/generate_indian_dataset.py
```

### 3. Train DS-CNN & Export INT8 Weights
Run the training pipeline:
```bash
python scripts/train_kws_model.py
```
This trains the Depthwise Separable CNN, quantizes the weights to INT8, and automatically updates both:
* `include/model_data.h` (PC Terminal application)
* `hardware/model_data.h` (ESP32 Firmware)

### 4. Rebuild the Application
```bash
# Rebuild aura.exe
g++ -O3 -std=c++17 -Iinclude src/main.cpp src/mfcc.cpp src/kws_engine.cpp src/telemetry.cpp -o aura.exe -static -static-libgcc -static-libstdc++ -lole32 -lwinmm -lpsapi
```

---

## Hardware Deployment (ESP32)

Refer to [`hardware/WIRING_AND_SETUP.md`](hardware/WIRING_AND_SETUP.md) for full circuit diagrams:
* **Microcontroller:** ESP32 DevKit V1 (240MHz, 520KB SRAM)
* **Microphone:** INMP441 I2S Digital Microphone (`SCK:14`, `WS:15`, `SD:32`)
* **Display:** 0.96" SSD1306 I2C OLED (`SDA:21`, `SCL:22`)
* **Status LEDs:** Green (`GPIO 18`), Blue (`GPIO 19`), Red (`GPIO 23`)
* **Firmware:** Flash [`hardware/esp32_firmware.ino`](hardware/esp32_firmware.ino) using the Arduino IDE or ESP-IDF.
