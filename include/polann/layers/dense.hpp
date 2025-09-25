#pragma once

#include <span>
#include <array>
#include <random>
#include <ranges>

namespace polann::layers
{
    /**
     * @brief Fully connected layer
     *
     * @tparam ActivationFunc Activation function (e.g., ReLU, Sigmoid)
     * @tparam InputSize Number of inputs to the layer
     * @tparam OutputSize Number of neurons in the layer
     */
    template <auto ActivationFunc, size_t InputSize, size_t OutputSize>
        requires requires(float x) { { ActivationFunc(x) } -> std::convertible_to<float>; }
    struct Dense
    {
        std::array<float, InputSize * OutputSize> weights; // Flattened weight matrix [OutputSize x InputSize]
        std::array<float, OutputSize> biases;

        /**
         * @brief Initializes weights and biases with random values
         *
         * Uses std::mt19937 and std::uniform_real_distribution to fill arrays.
         */
        Dense()
        {
            std::mt19937 rng;
            std::uniform_real_distribution<float> dist;
            std::ranges::generate(weights, [&] { return dist(rng); });
            std::ranges::generate(biases, [&] { return dist(rng); });
        }

        /**
         * @brief Forward pass through the layer
         *
         * @param in Input vector of size InputSize
         * @return std::vector<float> Output vector of size OutputSize
         *
         * @todo Remove allocation of out vector and move to a double buffer appraoch
         */
        std::vector<float> forward(std::span<const float> in) const
        {
            std::vector<float> out(OutputSize, 0.0f);
            for (size_t o = 0; o < OutputSize; ++o)
            {
                float sum = biases[o];
                for (size_t i = 0; i < InputSize; ++i)
                    sum += in[i] * weights[o * InputSize + i]; // row-major
                out[o] = ActivationFunc(sum);
            }

            return out;
        }

        // Allow compile-time access
        static constexpr size_t inputSize = InputSize;
        static constexpr size_t outputSize = OutputSize;
    };

} // namespace polann::layers
