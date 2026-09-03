#include "kws_engine.h"
#include "model_data.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace TinyKWS {

Engine::Engine() {
    std::memset(bufA, 0, sizeof(bufA));
    std::memset(bufB, 0, sizeof(bufB));
}

void Engine::softmax(const float* logits, float* probs, int len) {
    float maxLogit = logits[0];
    for (int i = 1; i < len; ++i) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sum += probs[i];
    }
    if (sum > 1e-7f) {
        for (int i = 0; i < len; ++i) {
            probs[i] /= sum;
        }
    }
}

PredictionResult Engine::predict(const float* melFeatures) {
    // 1. Normalize input features into a temporary normalized view
    // Input is (61 x 16)
    constexpr int IN_H = 61;
    constexpr int IN_W = 16;

    // Layer 1: Conv1 (1 -> 16), kernel (5, 3), stride (2, 1), padding (2, 1)
    // Output: 16 x 31 x 16 -> write into bufA
    constexpr int C1_OUT_C = 16;
    constexpr int C1_OUT_H = 31;
    constexpr int C1_OUT_W = 16;

    for (int oc = 0; oc < C1_OUT_C; ++oc) {
        float bias = C1_BIAS[oc];
        for (int oh = 0; oh < C1_OUT_H; ++oh) {
            for (int ow = 0; ow < C1_OUT_W; ++ow) {
                float sum = bias;
                for (int kh = 0; kh < 5; ++kh) {
                    int ih = oh * 2 - 2 + kh;
                    if (ih < 0 || ih >= IN_H) continue;
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = ow - 1 + kw;
                        if (iw < 0 || iw >= IN_W) continue;
                        float in_val = (melFeatures[ih * IN_W + iw] - NORM_MEAN) / NORM_STD;
                        float weight = C1_WEIGHTS[(oc * 5 + kh) * 3 + kw] * C1_SCALE;
                        sum += in_val * weight;
                    }
                }
                bufA[(oc * C1_OUT_H + oh) * C1_OUT_W + ow] = std::max(0.0f, sum); // ReLU
            }
        }
    }

    // Layer 2: Depthwise Conv1 (16 -> 16), kernel (3, 3), stride 1, padding 1
    // Output: 16 x 31 x 16 -> write into bufB
    for (int c = 0; c < 16; ++c) {
        float bias = DW1_BIAS[c];
        for (int oh = 0; oh < C1_OUT_H; ++oh) {
            for (int ow = 0; ow < C1_OUT_W; ++ow) {
                float sum = bias;
                for (int kh = 0; kh < 3; ++kh) {
                    int ih = oh - 1 + kh;
                    if (ih < 0 || ih >= C1_OUT_H) continue;
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = ow - 1 + kw;
                        if (iw < 0 || iw >= C1_OUT_W) continue;
                        float weight = DW1_WEIGHTS[(c * 3 + kh) * 3 + kw] * DW1_SCALE;
                        sum += bufA[(c * C1_OUT_H + ih) * C1_OUT_W + iw] * weight;
                    }
                }
                bufB[(c * C1_OUT_H + oh) * C1_OUT_W + ow] = std::max(0.0f, sum); // ReLU
            }
        }
    }

    // Layer 3: Pointwise Conv1 (16 -> 24), kernel 1x1
    // Output: 24 x 31 x 16 -> write into bufA
    constexpr int PW1_OUT_C = 24;
    for (int oc = 0; oc < PW1_OUT_C; ++oc) {
        float bias = PW1_BIAS[oc];
        for (int oh = 0; oh < C1_OUT_H; ++oh) {
            for (int ow = 0; ow < C1_OUT_W; ++ow) {
                float sum = bias;
                for (int ic = 0; ic < 16; ++ic) {
                    float weight = PW1_WEIGHTS[oc * 16 + ic] * PW1_SCALE;
                    sum += bufB[(ic * C1_OUT_H + oh) * C1_OUT_W + ow] * weight;
                }
                bufA[(oc * C1_OUT_H + oh) * C1_OUT_W + ow] = std::max(0.0f, sum); // ReLU
            }
        }
    }

    // Layer 4: MaxPool2d 2x2, stride 2x2
    // Input is in bufA: 24 x 31 x 16
    // Output: 24 x 15 x 8 -> write into bufB
    constexpr int POOL_H = 15;
    constexpr int POOL_W = 8;
    for (int c = 0; c < PW1_OUT_C; ++c) {
        for (int oh = 0; oh < POOL_H; ++oh) {
            for (int ow = 0; ow < POOL_W; ++ow) {
                float maxVal = -1e9f;
                for (int kh = 0; kh < 2; ++kh) {
                    int ih = oh * 2 + kh;
                    for (int kw = 0; kw < 2; ++kw) {
                        int iw = ow * 2 + kw;
                        float val = bufA[(c * C1_OUT_H + ih) * C1_OUT_W + iw];
                        if (val > maxVal) maxVal = val;
                    }
                }
                bufB[(c * POOL_H + oh) * POOL_W + ow] = maxVal;
            }
        }
    }

    // Layer 5: Depthwise Conv2 (24 -> 24), kernel (3, 3), stride 1, padding 1
    // Output: 24 x 15 x 8 -> write into bufA
    for (int c = 0; c < PW1_OUT_C; ++c) {
        float bias = DW2_BIAS[c];
        for (int oh = 0; oh < POOL_H; ++oh) {
            for (int ow = 0; ow < POOL_W; ++ow) {
                float sum = bias;
                for (int kh = 0; kh < 3; ++kh) {
                    int ih = oh - 1 + kh;
                    if (ih < 0 || ih >= POOL_H) continue;
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = ow - 1 + kw;
                        if (iw < 0 || iw >= POOL_W) continue;
                        float weight = DW2_WEIGHTS[(c * 3 + kh) * 3 + kw] * DW2_SCALE;
                        sum += bufB[(c * POOL_H + ih) * POOL_W + iw] * weight;
                    }
                }
                bufA[(c * POOL_H + oh) * POOL_W + ow] = std::max(0.0f, sum); // ReLU
            }
        }
    }

    // Layer 6: Pointwise Conv2 (24 -> 32), kernel 1x1
    // Output: 32 x 15 x 8 -> write into bufB
    constexpr int PW2_OUT_C = 32;
    for (int oc = 0; oc < PW2_OUT_C; ++oc) {
        float bias = PW2_BIAS[oc];
        for (int oh = 0; oh < POOL_H; ++oh) {
            for (int ow = 0; ow < POOL_W; ++ow) {
                float sum = bias;
                for (int ic = 0; ic < 24; ++ic) {
                    float weight = PW2_WEIGHTS[oc * 24 + ic] * PW2_SCALE;
                    sum += bufA[(ic * POOL_H + oh) * POOL_W + ow] * weight;
                }
                bufB[(oc * POOL_H + oh) * POOL_W + ow] = std::max(0.0f, sum); // ReLU
            }
        }
    }

    // Layer 7: Global Average Pooling (32 x 15 x 8) -> 32 values
    float gap[PW2_OUT_C];
    constexpr float numSpatial = static_cast<float>(POOL_H * POOL_W);
    for (int c = 0; c < PW2_OUT_C; ++c) {
        float sum = 0.0f;
        for (int i = 0; i < POOL_H * POOL_W; ++i) {
            sum += bufB[c * POOL_H * POOL_W + i];
        }
        gap[c] = sum / numSpatial;
    }

    // Layer 8: Linear (32 -> 4)
    float logits[4];
    for (int oc = 0; oc < 4; ++oc) {
        float sum = FC_BIAS[oc];
        for (int ic = 0; ic < PW2_OUT_C; ++ic) {
            float weight = FC_WEIGHTS[oc * PW2_OUT_C + ic] * FC_SCALE;
            sum += gap[ic] * weight;
        }
        logits[oc] = sum;
    }

    // Softmax
    PredictionResult result;
    softmax(logits, result.probabilities, 4);

    int bestIdx = 0;
    float bestProb = result.probabilities[0];
    for (int i = 1; i < 4; ++i) {
        if (result.probabilities[i] > bestProb) {
            bestProb = result.probabilities[i];
            bestIdx = i;
        }
    }

    result.predictedClass = static_cast<KeywordClass>(bestIdx);
    result.confidence = bestProb;
    return result;
}

} // namespace TinyKWS
