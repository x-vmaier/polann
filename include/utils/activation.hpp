#pragma once

#include <cmath>

namespace utils::activation
{
    inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
    inline float tanh(float x) { return std::tanh(x); }
    inline float sigmoid(float sum) { return 1.0f / (1.0f + std::exp(-sum)); }
} // namespace utils::activation
