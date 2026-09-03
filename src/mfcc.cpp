#include "mfcc.h"
#include <algorithm>
#include <cstring>

namespace TinyDSP {

MelSpectrogram::MelSpectrogram() {
    initHammingWindow();
    initMelFilterbank();
    initTwiddleTable();
}

void MelSpectrogram::initTwiddleTable() {
    for (int k = 0; k < FFT_SIZE / 2; ++k) {
        float angle = -2.0f * 3.141592653589793f * k / FFT_SIZE;
        twiddleReal[k] = std::cos(angle);
        twiddleImag[k] = std::sin(angle);
    }
}

void MelSpectrogram::initHammingWindow() {
    for (int i = 0; i < FRAME_LEN; ++i) {
        hammingWindow[i] = 0.54f - 0.46f * std::cos((2.0f * 3.141592653589793f * i) / (FRAME_LEN - 1));
    }
}

void MelSpectrogram::initMelFilterbank() {
    std::memset(melFilters, 0, sizeof(melFilters));

    float lowerMel = hzToMel(LOWER_FREQ);
    float upperMel = hzToMel(UPPER_FREQ);
    float melStep = (upperMel - lowerMel) / (NUM_MEL_BINS + 1);

    int numFftBins = FFT_SIZE / 2 + 1;
    float binWidth = (float)SAMPLE_RATE / FFT_SIZE;

    for (int m = 0; m < NUM_MEL_BINS; ++m) {
        float leftMel = lowerMel + m * melStep;
        float centerMel = leftMel + melStep;
        float rightMel = centerMel + melStep;

        float leftHz = melToHz(leftMel);
        float centerHz = melToHz(centerMel);
        float rightHz = melToHz(rightMel);

        for (int k = 0; k < numFftBins; ++k) {
            float freqHz = k * binWidth;
            if (freqHz >= leftHz && freqHz <= centerHz) {
                melFilters[m][k] = (freqHz - leftHz) / (centerHz - leftHz);
            } else if (freqHz > centerHz && freqHz <= rightHz) {
                melFilters[m][k] = (rightHz - freqHz) / (rightHz - centerHz);
            } else {
                melFilters[m][k] = 0.0f;
            }
        }
    }
}

// In-place Radix-2 FFT with precomputed twiddles (O(N log N) without trig calculations)
void MelSpectrogram::computeFFT(float* real, float* imag, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    for (int len = 2; len <= n; len <<= 1) {
        int halfLen = len >> 1;
        int step = FFT_SIZE / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < halfLen; ++k) {
                float w_real = twiddleReal[k * step];
                float w_imag = twiddleImag[k * step];

                float u_real = real[i + k];
                float u_imag = imag[i + k];
                float v_real = real[i + k + halfLen] * w_real - imag[i + k + halfLen] * w_imag;
                float v_imag = real[i + k + halfLen] * w_imag + imag[i + k + halfLen] * w_real;

                real[i + k] = u_real + v_real;
                imag[i + k] = u_imag + v_imag;
                real[i + k + halfLen] = u_real - v_real;
                imag[i + k + halfLen] = u_imag - v_imag;
            }
        }
    }
}

void MelSpectrogram::processFrame(const float* frame, float* outMels) {
    float real[FFT_SIZE];
    float imag[FFT_SIZE];

    // Windowing
    for (int i = 0; i < FRAME_LEN; ++i) {
        real[i] = frame[i] * hammingWindow[i];
        imag[i] = 0.0f;
    }

    computeFFT(real, imag, FFT_SIZE);

    int numFftBins = FFT_SIZE / 2 + 1;
    float powerSpectrum[FFT_SIZE / 2 + 1];
    for (int k = 0; k < numFftBins; ++k) {
        powerSpectrum[k] = (real[k] * real[k] + imag[k] * imag[k]) / (float)FFT_SIZE;
    }

    // Apply Mel filterbank and take log
    for (int m = 0; m < NUM_MEL_BINS; ++m) {
        float melEnergy = 0.0f;
        for (int k = 0; k < numFftBins; ++k) {
            melEnergy += powerSpectrum[k] * melFilters[m][k];
        }
        // Log-energy with small offset to avoid log(0)
        outMels[m] = std::log(melEnergy + 1e-6f);
    }
}

void MelSpectrogram::extractFeatures(const float* audio, int numSamples, float* outFeatures) {
    for (int frameIdx = 0; frameIdx < NUM_FRAMES; ++frameIdx) {
        int offset = frameIdx * FRAME_STEP;
        if (offset + FRAME_LEN <= numSamples) {
            processFrame(&audio[offset], &outFeatures[frameIdx * NUM_MEL_BINS]);
        } else {
            // zero pad if at boundary
            float buf[FRAME_LEN] = {0};
            int avail = numSamples - offset;
            if (avail > 0) {
                std::memcpy(buf, &audio[offset], avail * sizeof(float));
            }
            processFrame(buf, &outFeatures[frameIdx * NUM_MEL_BINS]);
        }
    }
}

} // namespace TinyDSP
