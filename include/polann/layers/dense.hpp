#pragma once

#include <span>
#include <array>
#include <random>
#include <ranges>
#include <concepts>

namespace polann::layers
{
    /**
     * @brief Activation function concept
     *
     * @tparam Func Activation function struct with compute/derivative static methods
     */
    template <typename Func>
    concept ActivationFunction = requires(float x) {
        { Func::compute(x) } -> std::convertible_to<float>;
        { Func::derivative(x) } -> std::convertible_to<float>;
    };

    /**
     * @brief Fully connected layer
     *
     * @tparam Activation Activation function type
     * @tparam InputSize Number of inputs to the layer
     * @tparam OutputSize Number of neurons in the layer
     */
    template <ActivationFunction Activation, size_t InputSize, size_t OutputSize>
    struct Dense
    {
        static_assert(InputSize > 0, "Input size must be positive");
        static_assert(OutputSize > 0, "Output size must be positive");

        static constexpr size_t inputSize = InputSize;
        static constexpr size_t outputSize = OutputSize;

        std::array<float, InputSize * OutputSize> weights; /// Flattened row-major weight matrix
        std::array<float, OutputSize> biases;

        /**
         * @brief Initializes weights and biases with Xavier/Glorot initialization
         *
         * Uses std::mt19937 and std::uniform_real_distribution to fill weights.
         */
        Dense()
        {
            // Xavier/Glorot initialization
            float limit = std::sqrt(6.0f / (InputSize + OutputSize));

            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<float> dist(-limit, limit);

            std::ranges::generate(weights, [&]() { return dist(rng); });
            std::ranges::fill(biases, 0.0f); // Initialize biases to zero
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
                out[o] = Activation::compute(sum);
            }
        }

        template <size_t InSize, size_t OutSize>
        void forward(const std::array<float, InSize> &in, std::array<float, OutSize> &out) const
        {
            static_assert(InSize >= InputSize);
            static_assert(OutSize >= OutputSize);

            // Forward with zero overhead
            forward(std::span<const float, InSize>(in), std::span<float, OutSize>(out));
        }
    };

} // namespace polann::layers
