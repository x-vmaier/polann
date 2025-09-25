#pragma once

#include <tuple>
#include <vector>
#include <span>

namespace polann::models
{
    /**
     * @brief Template-based fully inlined neural network
     *
     * @tparam Layers... Layer types added via ModelBuilder
     */
    template <typename... Layers>
    class NN
    {
    public:
        /**
         * @brief Constructs the NN with given layer instances
         *
         * @param ls Layer instances, typically created by ModelBuilder
         */
        explicit NN(Layers... ls) : layers(std::move(ls)...) {}

        /**
         * @brief Performs a forward pass through the network
         *
         * @param input Input vector (flattened) for the first layer
         * @return std::vector<float> Output vector from the last layer
         */
        std::vector<float> predict(std::span<const float> input) const
        {
            return predict_impl(input, std::index_sequence_for<Layers...>{});
        }

    private:
        std::tuple<Layers...> layers;

        /**
         * @brief Internal implementation of predict using compile-time fold
         *
         * @tparam I... Indices for the tuple of layers
         * @param input Input vector for the first layer
         * @param std::index_sequence<I...> Compile-time indices for tuple iteration
         * @return std::vector<float> Output of the last layer
         */
        template <size_t... I>
        std::vector<float> predict_impl(std::span<const float> input, std::index_sequence<I...>) const
        {
            std::vector<float> x(input.begin(), input.end());

            // Fold expression: applies each layer’s forward() in order
            ((x = std::get<I>(layers).forward(x)), ...);

            return x;
        }
    };

} // namespace polann::models
