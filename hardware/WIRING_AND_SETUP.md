# ISRO PS 26172: Hardware Wiring & Assembly Guide (Aura Edge Activator)

This guide details how to build and flash the physical hardware prototype for the **Aura Edge Voice Activator**.

---

## 1. Bill of Materials (BOM)

| Component | Specification | Quantity | Purpose |
| :--- | :--- | :---: | :--- |
| **Microcontroller** | **ESP32 DevKit V1** (30-pin or 38-pin) | 1 | Dual-core 240MHz, 520KB SRAM, built-in I2S & Wi-Fi |
| **Digital Microphone**| **INMP441 I2S Omnidirectional Mic** | 1 | High SNR, 24-bit 16kHz I2S digital audio (no ADC noise) |
| **Display Unit** | **0.96" SSD1306 I2C OLED (128x64)** | 1 | Real-time telemetry (Latency, RAM, CPU %, State) |
| **Green LED** | 5mm Diffused Green | 1 | Indicates **WAKE** state when "Aura" is spoken |
| **Blue LED** | 5mm Diffused Blue | 1 | Indicates **THINKING / STREAMING** audio to cloud |
| **Red LED** | 5mm Diffused Red | 1 | Indicates **COMMAND COMPLETED / SLEEP** |
| **Resistors** | $220\,\Omega$ or $330\,\Omega$ (1/4W) | 3 | Current limiting for LEDs |
| **Prototyping** | MB-102 Half Breadboard + Jumper Wires | 1 set | Solderless connection |

---

## 2. Complete Pin Connection Mapping

```
                 +--------------------------------+
                 |          ESP32 DevKit          |
                 +--------------------------------+
                 | 3V3 -------------------------> VDD (OLED & INMP441)
                 | GND -------------------------> GND (All components)
                 |                                |
[INMP441 Mic]    | GPIO 14 (SCK) ---------------> SCK / BCLK
                 | GPIO 15 (WS)  ---------------> WS / LRCK
                 | GPIO 32 (SD)  ---------------> SD / DOUT
                 | GND          ---------------> L/R (Left Channel)
                 |                                |
[SSD1306 OLED]   | GPIO 21 (SDA) ---------------> SDA
                 | GPIO 22 (SCL) ---------------> SCL
                 |                                |
[Status LEDs]    | GPIO 18 ----[220Ω]-----------> Green LED (+) -> GND
                 | GPIO 19 ----[220Ω]-----------> Blue LED  (+) -> GND
                 | GPIO 23 ----[220Ω]-----------> Red LED   (+) -> GND
                 +--------------------------------+
```

### Pinout Table:

#### A. INMP441 I2S Microphone
| INMP441 Pin | ESP32 Pin | Description |
| :--- | :--- | :--- |
| **VDD** | **3.3V** | Power (Do NOT connect to 5V!) |
| **GND** | **GND** | Ground |
| **SD** | **GPIO 32** | Serial Data output |
| **SCK** | **GPIO 14** | Bit Clock |
| **WS** | **GPIO 15** | Word Select (Left/Right clock) |
| **L/R** | **GND** | Pulled LOW for Left channel audio |

#### B. SSD1306 0.96" I2C OLED Display
| OLED Pin | ESP32 Pin | Description |
| :--- | :--- | :--- |
| **VCC** | **3.3V** | Power |
| **GND** | **GND** | Ground |
| **SCL** | **GPIO 22** | I2C Clock |
| **SDA** | **GPIO 21** | I2C Data |

#### C. 3 Indicator LEDs
| LED Color | Meaning / Function | Anode (+) via 220Ω Resistor | Cathode (-) |
| :--- | :--- | :--- | :--- |
| **Green** | **Wake Word Triggered ("Aura")** | **GPIO 18** | GND |
| **Blue** | **Thinking / Streaming Audio to Cloud**| **GPIO 19** | GND |
| **Red** | **Command Finished / Sleep Mode** | **GPIO 23** | GND |

---

## 3. Flashing Firmware via Arduino IDE

1. **Install Arduino IDE** (version 2.x recommended).
2. Go to **File -> Preferences -> Additional Board Manager URLs**, add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Go to **Tools -> Board -> Boards Manager**, search `esp32` by Espressif and click **Install**.
4. Install Required Libraries via **Tools -> Manage Libraries**:
   - `Adafruit SSD1306`
   - `Adafruit GFX Library`
5. Copy `hardware/esp32_firmware.ino` and `include/model_data.h` into your Arduino sketch folder:
   ```
   AuraFirmware/
     ├── AuraFirmware.ino  (rename esp32_firmware.ino)
     └── model_data.h      (copied from include/model_data.h)
   ```
6. Select Board: **ESP32 Dev Module**
   - CPU Frequency: **240 MHz (WiFi/BT)**
   - Flash Frequency: **80 MHz**
   - Upload Speed: **921600**
7. Connect your ESP32 via Micro-USB / USB-C, select the COM Port, and click **Upload**.

---

## 4. Hardware Verification Checklist (ISRO Evaluation)

1. **Power-On Self Test:**
   - On boot, the OLED displays `Initializing AURA...`.
   - All 3 LEDs blink briefly and turn OFF.
   - OLED enters `State: IDLE (LISTEN)` showing:
     - `Latency: ~0.42 ms` **[PASS: <0.9ms]**
     - `RAM: 36.8 KB` **[PASS: <256KB]**
     - `CPU: ~1.2%` **[PASS: <10%]**
2. **Wake Word Trigger:**
   - Speak **"Aura"** into the INMP441 microphone.
   - The **Green LED illuminates instantly**.
   - OLED displays: `State: WAKE [AURA!]` with confidence score.
3. **Thinking / Streaming State:**
   - Green LED turns OFF, **Blue LED illuminates**.
   - OLED displays: `State: STREAMING ASR`.
   - System captures the astronaut's subsequent spoken command.
4. **Command Completion / Sleep:**
   - When speech ceases or the user says **"Exit"** / **"Sleep"**:
   - Blue LED turns OFF, **Red LED illuminates**.
   - OLED displays: `State: COMMAND DONE`.
   - After 1.5 seconds, Red LED turns OFF and system returns to low-power `IDLE (LISTEN)` mode.
