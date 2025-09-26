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
     * @tparam ActivationFunc Activation function (see polann::utils::activation_functions)
     * @tparam InputSize Number of inputs to the layer
     * @tparam OutputSize Number of neurons in the layer
     */
    template <auto ActivationFunc, size_t InputSize, size_t OutputSize>
        requires requires(float x) { { ActivationFunc(x) } -> std::convertible_to<float>; }
    struct Dense
    {
        static_assert(InputSize > 0, "Input size must be positive");
        static_assert(OutputSize > 0, "Output size must be positive");

        // Allow compile-time access
        static constexpr size_t inputSize = InputSize;
        static constexpr size_t outputSize = OutputSize;

        std::array<float, InputSize * OutputSize> weights; /// Flattened row-major weight matrix
        std::array<float, OutputSize> biases;

        /**
         * @brief Initializes weights and biases with random values
         *
         * Uses std::mt19937 and std::uniform_real_distribution to fill arrays.
         */
        Dense()
        {
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            std::ranges::generate(weights, [&] { return dist(rng); });
            std::ranges::generate(biases, [&] { return dist(rng); });
        }

        /**
         * @brief Forward pass through the layer
         *
         * @param in Input span of size InputSize
         * @param out Output span of size OutputSize
         */
        void forward(std::span<const float> in, std::span<float> out) const
        {
            for (size_t o = 0; o < OutputSize; ++o)
            {
                float sum = biases[o];
                for (size_t i = 0; i < InputSize; ++i)
                    sum += in[i] * weights[o * InputSize + i];
                out[o] = ActivationFunc(sum);
            }
        }
    };

} // namespace polann::layers
