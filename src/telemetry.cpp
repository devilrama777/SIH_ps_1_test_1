#include "telemetry.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

TelemetryMonitor::TelemetryMonitor() : lastLatencyMs(0.0f), lastCpuCheckTime(0), lastProcessCpuTime(0), numProcessors(1) {
#if defined(_WIN32)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    numProcessors = sysInfo.dwNumberOfProcessors;
    if (numProcessors < 1) numProcessors = 1;

    FILETIME nowFt, createFt, exitFt, kernelFt, userFt;
    GetSystemTimeAsFileTime(&nowFt);
    lastCpuCheckTime = getFileTimeAsUint64(&nowFt);

    if (GetProcessTimes(GetCurrentProcess(), &createFt, &exitFt, &kernelFt, &userFt)) {
        lastProcessCpuTime = getFileTimeAsUint64(&kernelFt) + getFileTimeAsUint64(&userFt);
    }
#endif
}

uint64_t TelemetryMonitor::getFileTimeAsUint64(void* fileTimePtr) {
#if defined(_WIN32)
    FILETIME* ft = static_cast<FILETIME*>(fileTimePtr);
    ULARGE_INTEGER uli;
    uli.LowPart = ft->dwLowDateTime;
    uli.HighPart = ft->dwHighDateTime;
    return uli.QuadPart;
#else
    return 0;
#endif
}

void TelemetryMonitor::startTimer() {
    startTime = std::chrono::high_resolution_clock::now();
}

float TelemetryMonitor::stopTimer() {
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = endTime - startTime;
    lastLatencyMs = duration.count();
    return lastLatencyMs;
}

SystemMetrics TelemetryMonitor::updateMetrics() {
    SystemMetrics metrics;
    metrics.latencyMs = lastLatencyMs;
    metrics.ramUsageKb = 0.0f;
    metrics.cpuPercent = 0.0f;

#if defined(_WIN32)
    // 1. RAM Usage
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        // WorkingSetSize / 1024 = KB
        metrics.ramUsageKb = static_cast<float>(pmc.WorkingSetSize) / 1024.0f;
    }

    // 2. CPU Usage
    FILETIME nowFt, createFt, exitFt, kernelFt, userFt;
    GetSystemTimeAsFileTime(&nowFt);
    uint64_t currentSystemTime = getFileTimeAsUint64(&nowFt);

    if (GetProcessTimes(GetCurrentProcess(), &createFt, &exitFt, &kernelFt, &userFt)) {
        uint64_t currentProcessTime = getFileTimeAsUint64(&kernelFt) + getFileTimeAsUint64(&userFt);

        uint64_t timeDelta = currentSystemTime - lastCpuCheckTime;
        uint64_t processDelta = currentProcessTime - lastProcessCpuTime;

        if (timeDelta > 0) {
            metrics.cpuPercent = (static_cast<float>(processDelta) / static_cast<float>(timeDelta)) * 100.0f / static_cast<float>(numProcessors);
            if (metrics.cpuPercent < 0.0f) metrics.cpuPercent = 0.0f;
            if (metrics.cpuPercent > 100.0f) metrics.cpuPercent = 100.0f;
        }

        lastCpuCheckTime = currentSystemTime;
        lastProcessCpuTime = currentProcessTime;
    }
#endif

    return metrics;
}
