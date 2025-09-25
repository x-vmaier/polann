#pragma once
#include <cmath>

namespace polann::utils::activation_functions
{
    [[nodiscard]] inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
    [[nodiscard]] inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    [[nodiscard]] inline float tanh_fn(float x) { return std::tanh(x); }
    [[nodiscard]] inline float identity(float x) { return x; }

} // namespace polann::utils::activation_functions
