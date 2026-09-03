#ifndef MFCC_H
#define MFCC_H

#include <vector>
#include <cmath>
#include <cstdint>

namespace TinyDSP {

constexpr int SAMPLE_RATE = 16000;
constexpr int FRAME_LEN = 512;       // 32 ms
constexpr int FRAME_STEP = 256;      // 16 ms
constexpr int FFT_SIZE = 512;
constexpr int NUM_MEL_BINS = 16;
constexpr float LOWER_FREQ = 300.0f;
constexpr float UPPER_FREQ = 7500.0f;
constexpr int NUM_FRAMES = 61;       // (16000 - 512) / 256 + 1

class MelSpectrogram {
public:
    MelSpectrogram();
    ~MelSpectrogram() = default;

    // Process a full 1-second (16000 samples) PCM buffer into a (61 x 16) feature matrix
    void extractFeatures(const float* audio, int numSamples, float* outFeatures);

    // Process a single frame of 512 samples into 16 mel bin values
    void processFrame(const float* frame, float* outMels);

private:
    float hammingWindow[FRAME_LEN];
    float melFilters[NUM_MEL_BINS][FFT_SIZE / 2 + 1];
    float twiddleReal[FFT_SIZE / 2];
    float twiddleImag[FFT_SIZE / 2];

    void initHammingWindow();
    void initMelFilterbank();
    void initTwiddleTable();
    void computeFFT(float* real, float* imag, int n);
    float hzToMel(float hz) const {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    }
    float melToHz(float mel) const {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }
};

} // namespace TinyDSP

#endif // MFCC_H
