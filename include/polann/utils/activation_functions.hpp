#pragma once

#include <cmath>

namespace polann::utils
{
    struct Sigmoid
    {
        [[nodiscard]] static inline float compute(float x)
        {
            if (x > 500.0f)
                return 1.0f;
            if (x < -500.0f)
                return 0.0f;
            return 1.0f / (1.0f + std::exp(-x));
        }
        [[nodiscard]] static inline float derivative(float y) { return y * (1.0f - y); }
    };

    struct ReLU
    {
        [[nodiscard]] static inline float compute(float x) { return std::max(0.0f, x); }
        [[nodiscard]] static inline float derivative(float y) { return (y > 0.0f) ? 1.0f : 0.0f; }
    };

    struct Tanh
    {
        [[nodiscard]] static inline float compute(float x) { return std::tanh(x); }
        [[nodiscard]] static inline float derivative(float y) { return 1.0f - (y * y); }
    };

    struct Identity
    {
        [[nodiscard]] static inline float compute(float x) { return x; }
        [[nodiscard]] static inline float derivative(float /*y*/) { return 1.0f; }
    };

} // namespace polann::utils
