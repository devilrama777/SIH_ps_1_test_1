#ifndef TELEMETRY_H
#define TELEMETRY_H

#include <cstdint>
#include <chrono>

struct SystemMetrics {
    float latencyMs;     // Last inference computation time in milliseconds
    float ramUsageKb;    // Current RAM footprint in KB
    float cpuPercent;    // Current CPU usage percentage (0.0 to 100.0)
};

class TelemetryMonitor {
public:
    TelemetryMonitor();
    ~TelemetryMonitor() = default;

    // Start latency stopwatch
    void startTimer();

    // Stop latency stopwatch and record inference latency
    float stopTimer();

    // Refresh RAM and CPU measurements
    SystemMetrics updateMetrics();

private:
    std::chrono::high_resolution_clock::time_point startTime;
    float lastLatencyMs;

    uint64_t lastCpuCheckTime;
    uint64_t lastProcessCpuTime;
    int numProcessors;

    uint64_t getFileTimeAsUint64(void* fileTimePtr);
};

#endif // TELEMETRY_H
