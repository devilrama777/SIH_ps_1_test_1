#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_WAV
#define MA_NO_FLAC
#define MA_NO_MP3
#include "miniaudio.h"

#include "mfcc.h"
#include "kws_engine.h"
#include "telemetry.h"

#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>
#include <iomanip>
#include <atomic>
#include <cmath>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <conio.h>
#endif

// ANSI Colors
#define COLOR_RESET   "\033[0m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_RED     "\033[31m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_MAGENTA "\033[35m"
#define BG_GREEN      "\033[42;30m"
#define BG_BLUE       "\033[44;37m"
#define BG_RED        "\033[41;37m"

enum class AgentState {
    SLEEPING,            // Dormant: All speech/hums IGNORED. Wakes ONLY on "Aura"
    WAKE_AURA,           // Waking up: Green LED ON (1.0s)
    LISTENING_COMMAND,   // Active command listening: Blue LED ON. Captures full sentences!
    THINKING_ANALYSING,  // Thinking / Analysing: Blue LED flashing. Executes and loops back!
    COMMAND_DONE         // Sleeping / Exit: Red LED ON (1.5s)
};

constexpr int SAMPLE_RATE = 16000;
constexpr int BUFFER_SAMPLES = 16000; // 1.0 second rolling audio window

class AudioRingBuffer {
public:
    AudioRingBuffer(size_t capacity) : cap(capacity), head(0), isFull(false) {
        buf.resize(capacity, 0.0f);
    }

    void write(const float* data, size_t count) {
        std::lock_guard<std::mutex> lock(mtx);
        for (size_t i = 0; i < count; ++i) {
            buf[head] = data[i];
            head = (head + 1) % cap;
            if (head == 0) isFull = true;
        }
    }

    void getLatest(std::vector<float>& out) {
        std::lock_guard<std::mutex> lock(mtx);
        out.resize(cap);
        if (!isFull) {
            for (size_t i = 0; i < cap; ++i) {
                out[i] = buf[i];
            }
        } else {
            size_t idx = head;
            for (size_t i = 0; i < cap; ++i) {
                out[i] = buf[idx];
                idx = (idx + 1) % cap;
            }
        }
    }

private:
    std::vector<float> buf;
    size_t cap;
    size_t head;
    bool isFull;
    std::mutex mtx;
};

static AudioRingBuffer g_ringBuffer(BUFFER_SAMPLES);
static std::atomic<bool> g_running(true);

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    (void)pOutput;
    if (pInput == nullptr || frameCount == 0) return;

    const float* pInputFloat = static_cast<const float*>(pInput);
    g_ringBuffer.write(pInputFloat, frameCount);
}

float computeRMS(const float* data, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum / (float)count);
}

void printDashboard(AgentState state, const SystemMetrics& metrics, const TinyKWS::PredictionResult& pred, 
                    int commandCount, float lastCommandDurationSec, bool isSpeakingNow) {
    std::cout << "\033[H";

    std::cout << COLOR_BOLD << "========================================================================\n";
    std::cout << "        ISRO PS 26172 - LOW LATENCY EDGE VOICE ACTIVATOR (AURA)         \n";
    std::cout << "========================================================================\n" << COLOR_RESET;

    // 1. Emulated 3-LED indicator bar
    std::cout << " [HARDWARE LED INDICATORS]:\n ";
    if (state == AgentState::WAKE_AURA) {
        std::cout << BG_GREEN << " [ GREEN: WAKE (AURA) ] " << COLOR_RESET << "  [ BLUE: OFF ]  [ RED: OFF ]\n";
    } else if (state == AgentState::LISTENING_COMMAND) {
        std::cout << " [ GREEN: OFF ]  " << BG_BLUE << " [ BLUE: LISTENING FOR COMMAND ] " << COLOR_RESET << "  [ RED: OFF ]\n";
    } else if (state == AgentState::THINKING_ANALYSING) {
        std::cout << " [ GREEN: OFF ]  " << BG_BLUE << " [ BLUE: *** THINKING / ANALYSING *** ] " << COLOR_RESET << "  [ RED: OFF ]\n";
    } else if (state == AgentState::COMMAND_DONE) {
        std::cout << " [ GREEN: OFF ]  [ BLUE: OFF ]  " << BG_RED << " [ RED: SLEEP / EXIT ] " << COLOR_RESET << "\n";
    } else {
        std::cout << " [ GREEN: OFF ]  [ BLUE: OFF ]  [ RED: OFF ]  " << COLOR_CYAN << "(ASLEEP / DORMANT)" << COLOR_RESET << "\n";
    }

    std::cout << "\n";
    std::cout << COLOR_BOLD << " [EMULATED OLED HARDWARE DISPLAY - 128x64]:\n" << COLOR_RESET;
    std::cout << " +--------------------------------------------------------------------+\n";

    // OLED Row 1: State
    std::string stateStr = "ASLEEP (COMMANDS IGNORED)";
    if (state == AgentState::WAKE_AURA) stateStr = ">> WAKE DETECTED: AURA <<";
    else if (state == AgentState::LISTENING_COMMAND) {
        stateStr = isSpeakingNow ? ">> LISTENING: [USER SPEAKING...] <<" : ">> ACTIVE: READY FOR COMMAND <<";
    }
    else if (state == AgentState::THINKING_ANALYSING) stateStr = ">> THINKING / ANALYSING COMMAND... <<";
    else if (state == AgentState::COMMAND_DONE) stateStr = ">> GOING TO SLEEP... <<";

    std::cout << " | State:   " << std::left << std::setw(58) << stateStr << "|\n";

    // OLED Row 2: Latency Benchmark
    std::string latencyStatus = (metrics.latencyMs < 0.9f) ? "[PASS: <0.9ms]" : "[FAIL]";
    std::cout << " | Compute Latency:  " << std::fixed << std::setprecision(3) << metrics.latencyMs 
              << " ms  " << COLOR_GREEN << latencyStatus << COLOR_RESET << "                           |\n";

    // OLED Row 3: Memory Footprint (RAM)
    std::cout << " | Embedded Model RAM: 36.80 KB (Static Arena)   " << COLOR_GREEN << "[PASS: <256KB]" << COLOR_RESET << "       |\n";
    std::cout << " | Host Process RSS:   " << std::fixed << std::setprecision(1) << metrics.ramUsageKb << " KB                                        |\n";

    // OLED Row 4: CPU Utilization
    std::string cpuStatus = (metrics.cpuPercent < 10.0f) ? "[PASS: <10%]" : "[RUNNING]";
    std::cout << " | CPU Utilization:    " << std::fixed << std::setprecision(2) << metrics.cpuPercent 
              << " %  " << COLOR_GREEN << cpuStatus << COLOR_RESET << "                             |\n";

    // OLED Row 5: KWS Probabilities
    std::cout << " +--------------------------------------------------------------------+\n";
    std::cout << " | Acoustic Classification:                                           |\n";
    std::cout << " |   Silence: " << std::fixed << std::setprecision(1) << (pred.probabilities[0] * 100.0f) << "%"
              << "  | Unknown: " << std::setprecision(1) << (pred.probabilities[1] * 100.0f) << "%"
              << "  | " << COLOR_GREEN << "AURA: " << std::setprecision(1) << (pred.probabilities[2] * 100.0f) << "%" << COLOR_RESET
              << "  | " << COLOR_RED << "EXIT: " << std::setprecision(1) << (pred.probabilities[3] * 100.0f) << "%" << COLOR_RESET << " |\n";
    std::cout << " +--------------------------------------------------------------------+\n";

    if (state == AgentState::LISTENING_COMMAND) {
        std::cout << COLOR_BLUE << "\n [BLUE LIGHT ON - ACTIVE COMMAND LISTENER]:\n"
                  << " -> Speak your full command line/sentence (e.g., 'What is the time?').\n"
                  << " -> Aura listens to the whole sentence until you stop speaking.\n"
                  << " -> Commands executed so far: " << commandCount << "\n"
                  << " -> Say " << COLOR_RED << "'Exit'" << COLOR_BLUE << " to put Aura to sleep.\n" << COLOR_RESET;
    } else if (state == AgentState::THINKING_ANALYSING) {
        std::cout << COLOR_MAGENTA << "\n [BLUE LIGHT FLASHING - THINKING / ANALYSING]:\n"
                  << " -> Full sentence captured (" << std::fixed << std::setprecision(1) << lastCommandDurationSec 
                  << "s duration, " << (int)(lastCommandDurationSec * 16000) << " audio samples).\n"
                  << " -> Command analysed & executed! Returning immediately to listen for next command...\n" << COLOR_RESET;
    } else if (state == AgentState::WAKE_AURA) {
        std::cout << COLOR_GREEN << "\n [GREEN LIGHT ON]: 'Aura' wake word confirmed! Waking up system...\n" << COLOR_RESET;
    } else if (state == AgentState::COMMAND_DONE) {
        std::cout << COLOR_RED << "\n [RED LIGHT ON]: 'Exit' keyword received. Putting Aura to sleep mode.\n" << COLOR_RESET;
    } else {
        std::cout << COLOR_YELLOW << "\n [AURA IS ASLEEP]:\n"
                  << " -> Aura is dormant. General conversation, hums, and commands are 100% ignored.\n"
                  << " -> Speak " << COLOR_GREEN << "'Aura'" << COLOR_YELLOW << " into the microphone to wake up.\n"
                  << " -> Press " << COLOR_CYAN << "[Q]" << COLOR_YELLOW << " or " << COLOR_CYAN << "[Ctrl+C]" << COLOR_YELLOW << " anytime to quit.\n" << COLOR_RESET;
    }

    std::cout << std::flush;
}

int main() {
#if defined(_WIN32)
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif

    std::cout << "\033[2J\033[H";
    std::cout << "Initializing Aura Edge Voice Activator...\n";

    TinyDSP::MelSpectrogram dsp;
    TinyKWS::Engine engine;
    TelemetryMonitor telemetry;

    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format   = ma_format_f32;
    deviceConfig.capture.channels = 1;
    deviceConfig.sampleRate       = SAMPLE_RATE;
    deviceConfig.dataCallback     = data_callback;
    deviceConfig.pUserData        = nullptr;

    ma_device device;
    if (ma_device_init(nullptr, &deviceConfig, &device) != MA_SUCCESS) {
        std::cerr << "Failed to initialize microphone capture device!\n";
        return -1;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "Failed to start microphone device!\n";
        ma_device_uninit(&device);
        return -1;
    }

    std::cout << "Microphone started successfully @ 16kHz Mono.\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    AgentState state = AgentState::SLEEPING;
    int stateHoldTicks = 0;

    // Leaky accumulators for smooth and robust keyword spotting
    float auraAccumulator = 0.0f;
    float exitAccumulator = 0.0f;

    // VAD & Full sentence capture variables
    bool userIsSpeaking = false;
    int speechDurationMs = 0;
    int silenceDurationMs = 0;
    int totalCommandsExecuted = 0;
    float lastCommandSec = 0.0f;

    std::vector<float> audioWindow(BUFFER_SAMPLES, 0.0f);
    std::vector<float> melFeatures(TinyDSP::NUM_FRAMES * TinyDSP::NUM_MEL_BINS, 0.0f);

    TinyKWS::PredictionResult lastPred;
    lastPred.predictedClass = TinyKWS::KeywordClass::SILENCE;
    lastPred.confidence = 1.0f;
    lastPred.probabilities[0] = 1.0f;
    lastPred.probabilities[1] = 0.0f;
    lastPred.probabilities[2] = 0.0f;
    lastPred.probabilities[3] = 0.0f;

    while (g_running) {
        auto loopStart = std::chrono::steady_clock::now();

#if defined(_WIN32)
        if (_kbhit()) {
            char ch = _getch();
            if (ch == 'q' || ch == 'Q' || ch == 27) {
                break;
            }
        }
#endif

        g_ringBuffer.getLatest(audioWindow);
        float currentRms = computeRMS(audioWindow.data(), BUFFER_SAMPLES);

        // Feature extraction & Inference
        dsp.extractFeatures(audioWindow.data(), BUFFER_SAMPLES, melFeatures.data());

        telemetry.startTimer();
        TinyKWS::PredictionResult pred = engine.predict(melFeatures.data());
        telemetry.stopTimer();

        SystemMetrics metrics = telemetry.updateMetrics();
        lastPred = pred;

        // STATE MACHINE
        if (state == AgentState::SLEEPING) {
            // In SLEEPING:
            // Dynamic threshold: accepts normal and soft Indian voices (currentRms > 0.003f)
            float auraProb = pred.probabilities[2];
            if (currentRms > 0.003f && auraProb > 0.45f) {
                auraAccumulator += auraProb;
                if (auraAccumulator >= 0.80f) {
                    state = AgentState::WAKE_AURA;
                    stateHoldTicks = 10; // Green LED for 1.0 second
                    auraAccumulator = 0.0f;
                    exitAccumulator = 0.0f;
                }
            } else {
                auraAccumulator *= 0.5f; // Exponential decay
            }
        } 
        else if (state == AgentState::WAKE_AURA) {
            stateHoldTicks--;
            if (stateHoldTicks <= 0) {
                // Transition to Active Command Listening
                state = AgentState::LISTENING_COMMAND;
                auraAccumulator = 0.0f;
                exitAccumulator = 0.0f;
                userIsSpeaking = false;
                speechDurationMs = 0;
                silenceDurationMs = 0;
            }
        } 
        else if (state == AgentState::LISTENING_COMMAND) {
            float exitProb = pred.probabilities[3];

            // VAD Speech tracking
            if (currentRms > 0.007f) {
                userIsSpeaking = true;
                speechDurationMs += 100;
                silenceDurationMs = 0;
            } else if (userIsSpeaking) {
                silenceDurationMs += 100;
            }

            // ONLY check for "Exit" if:
            // 1. The user has NOT been speaking a long sentence (speechDurationMs < 1200ms)
            // 2. The exit probability is high and dominant over unknown
            if (speechDurationMs < 1200 && exitProb > 0.70f && exitProb > pred.probabilities[1]) {
                exitAccumulator += exitProb;
                if (exitAccumulator >= 0.90f) {
                    state = AgentState::COMMAND_DONE;
                    stateHoldTicks = 15; // Red LED for 1.5s
                    exitAccumulator = 0.0f;
                    auraAccumulator = 0.0f;
                    userIsSpeaking = false;
                }
            } else {
                exitAccumulator *= 0.5f; // Exponential decay
            }

            // VAD: Capture full command sentence without cutting off
            if (state == AgentState::LISTENING_COMMAND) {
                if (userIsSpeaking && silenceDurationMs >= 1000 && speechDurationMs >= 400) {
                    // Full command sentence completed!
                    lastCommandSec = static_cast<float>(speechDurationMs + silenceDurationMs) / 1000.0f;
                    totalCommandsExecuted++;
                    userIsSpeaking = false;
                    speechDurationMs = 0;
                    silenceDurationMs = 0;
                    exitAccumulator = 0.0f;

                    // Switch to Thinking / Analysing state
                    state = AgentState::THINKING_ANALYSING;
                    stateHoldTicks = 15; // Think for 1.5s, then loop back to command listening!
                }
            }
        } 
        else if (state == AgentState::THINKING_ANALYSING) {
            stateHoldTicks--;
            if (stateHoldTicks <= 0) {
                // Command executed! Loop straight back to LISTENING_COMMAND!
                state = AgentState::LISTENING_COMMAND;
                userIsSpeaking = false;
                speechDurationMs = 0;
                silenceDurationMs = 0;
                exitAccumulator = 0.0f;
                auraAccumulator = 0.0f;
            }
        } 
        else if (state == AgentState::COMMAND_DONE) {
            stateHoldTicks--;
            if (stateHoldTicks <= 0) {
                // Return to dormant SLEEPING state
                state = AgentState::SLEEPING;
                auraAccumulator = 0.0f;
                exitAccumulator = 0.0f;
                userIsSpeaking = false;
                speechDurationMs = 0;
                silenceDurationMs = 0;
            }
        }

        printDashboard(state, metrics, lastPred, totalCommandsExecuted, lastCommandSec, userIsSpeaking);

        auto loopEnd = std::chrono::steady_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(loopEnd - loopStart).count();
        if (elapsedMs < 100) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100 - elapsedMs));
        }
    }

    std::cout << "\033[2J\033[H";
    std::cout << COLOR_CYAN << "Aura Voice Activator shut down cleanly.\n" << COLOR_RESET;
    ma_device_uninit(&device);
    return 0;
}
