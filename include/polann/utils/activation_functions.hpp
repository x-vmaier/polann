#pragma once

#include <cmath>

namespace polann::utils::activation_functions
{
    [[nodiscard]] inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
    [[nodiscard]] inline float tanh(float x) { return std::tanh(x); }
    [[nodiscard]] inline float sigmoid(float sum) { return 1.0f / (1.0f + std::exp(-sum)); }
} // namespace polann::utils::activation_functions
