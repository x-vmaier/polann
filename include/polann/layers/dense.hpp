#pragma once

#include <span>
#include <array>
#include <vector>
#include <random>
#include <ranges>
#include <functional>
#include "polann/utils/activation_functions.hpp"

namespace polann::layers
{
    template <auto ActivationFunc = polann::utils::activation_functions::identity, size_t InputSize = 0, size_t OutputSize = 0>
    struct Dense
    {
        std::array<float, InputSize * OutputSize> weights;
        std::array<float, OutputSize> biases;

        Dense()
        {
            std::mt19937 rng;
            std::uniform_real_distribution<float> dist;
            std::ranges::generate(weights, [&] { return dist(rng); });
            std::ranges::generate(biases, [&] { return dist(rng); });
        }

        std::vector<float> forward(std::span<const float> in) const
        {
            std::vector<float> out(OutputSize, 0.0f);
            for (size_t o = 0; o < OutputSize; ++o)
            {
                float sum = biases[o];
                for (size_t i = 0; i < InputSize; ++i)
                    sum += in[i] * weights[o * InputSize + i];
                out[o] = ActivationFunc(sum);
            }
            return out;
        }
    };

} // namespace polann::layers
