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

        // Gradients
        std::array<float, InputSize * OutputSize> gradWeights;
        std::array<float, OutputSize> gradBiases;

        // Last forward pass values for backprop
        mutable std::array<float, InputSize> lastInput;
        mutable std::array<float, OutputSize> lastActivation;

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
            // Store input for backward pass
            std::copy_n(in.begin(), InputSize, lastInput.begin());

            for (size_t o = 0; o < OutputSize; ++o)
            {
                float sum = biases[o];
                for (size_t i = 0; i < InputSize; ++i)
                    sum += in[i] * weights[o * InputSize + i];

                lastActivation[o] = Activation::compute(sum); // Store post-activation
                out[o] = lastActivation[o];
            }
        }

        void forward(const std::array<float, InputSize> &in, std::array<float, OutputSize> &out) const
        {
            // Forward with zero overhead
            forward(std::span<const float, InputSize>(in), std::span<float, OutputSize>(out));
        }

        /**
         * @brief Backward pass through the layer
         *
         * @param gradOutput Gradient w.r.t. this layer's output
         * @param gradInput Output: gradient w.r.t. this layer's input
         */
        void backward(std::span<const float> gradOutput, std::span<float> gradInput)
        {
            // Clear input gradients
            std::fill(gradInput.begin(), gradInput.begin() + InputSize, 0.0f);

            for (size_t o = 0; o < OutputSize; ++o)
            {
                // Apply activation derivative
                float delta = gradOutput[o] * Activation::derivative(lastActivation[o]);

                // Accumulate bias gradient
                gradBiases[o] += delta;

                for (size_t i = 0; i < InputSize; ++i)
                {
                    gradInput[i] += delta * weights[o * InputSize + i];
                    gradWeights[o * InputSize + i] += delta * lastInput[i];
                }
            }
        }

        void clearGradients()
        {
            std::fill(gradWeights.begin(), gradWeights.end(), 0.0f);
            std::fill(gradBiases.begin(), gradBiases.end(), 0.0f);
        }

        void scaleGradients(float scale)
        {
            for (auto &g : gradWeights) g *= scale;
            for (auto &g : gradBiases) g *= scale;
        }
    };

} // namespace polann::layers
