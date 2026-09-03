#ifndef KWS_ENGINE_H
#define KWS_ENGINE_H

#include <cstdint>
#include <vector>

namespace TinyKWS {

enum class KeywordClass {
    SILENCE = 0,
    UNKNOWN = 1,
    AURA = 2,
    EXIT = 3
};

struct PredictionResult {
    KeywordClass predictedClass;
    float confidence;
    float probabilities[4];
};

class Engine {
public:
    Engine();
    ~Engine() = default;

    // Run inference on (61 x 16) normalized log-mel features
    PredictionResult predict(const float* melFeatures);

private:
    // Ping-pong buffers for zero-allocation tensor operations
    // Buffer size: max channels (24) * max height (31) * max width (16) = 11,904 floats (~47 KB)
    static constexpr int MAX_BUF_SIZE = 24 * 31 * 16;
    float bufA[MAX_BUF_SIZE];
    float bufB[MAX_BUF_SIZE];

    void softmax(const float* logits, float* probs, int len);
};

} // namespace TinyKWS

#endif // KWS_ENGINE_H
