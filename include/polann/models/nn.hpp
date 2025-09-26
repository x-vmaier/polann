#pragma once

#include <tuple>
#include <span>
#include <algorithm>
#include <array>

namespace polann::models
{
    // Get size of the largest layer
    template <typename... Layers>
    constexpr size_t maxOutputSize = (std::max)({Layers::outputSize...});

    /**
     * @brief Template-based fully inlined neural network
     *
     * @tparam Layers... Layer types added via ModelBuilder
     */
    template <typename... Layers>
    class NN
    {
        using firstLayerType = std::tuple_element_t<0, std::tuple<Layers...>>;
        using finalLayerType = std::tuple_element_t<sizeof...(Layers) - 1, std::tuple<Layers...>>;

        static_assert(sizeof...(Layers) > 0, "Neural network must have at least one layer");

    public:
        static constexpr size_t maxLayerOutputSize = maxOutputSize<Layers...>; /// Maximum buffer size needed for any layer output
        static constexpr size_t layerCount = sizeof...(Layers);                /// Number of layers in the network
        static constexpr size_t outputSize = finalLayerType::outputSize;       /// Output size of the network

        /**
         * @brief Constructs the NN with given layer instances
         *
         * @param ls Layer instances, typically created by ModelBuilder
         */
        explicit NN(Layers... ls) : layers(std::move(ls)...) {}

        /**
         * @brief Performs a forward pass through the network
         *
         * @param input Fixed-size input vector for the first layer
         * @return std::array<float, outputSize> Output vector produced by the final layer
         *
         * @note Internal buffers are reused to avoid heap allocations
         */
        template <size_t InputSize>
        [[nodiscard]] std::array<float, outputSize> predict(const std::array<float, InputSize> &input) const
        {
            static_assert(InputSize == firstLayerType::inputSize);

            alignas(32) std::array<float, maxLayerOutputSize> buf1{};
            alignas(32) std::array<float, maxLayerOutputSize> buf2{};
            return predictImpl<InputSize>(input, buf1, buf2, std::index_sequence_for<Layers...>{});
        }

    private:
        std::tuple<Layers...> layers;

        template <size_t InputSize, size_t... I>
        [[nodiscard]] std::array<float, outputSize> predictImpl(
            std::span<const float, InputSize> input,
            std::array<float, maxLayerOutputSize> &buf1,
            std::array<float, maxLayerOutputSize> &buf2,
            std::index_sequence<I...>) const
        {
            // Forward through layers using fold expression
            size_t currentSize = InputSize;                      // Track the layer size
            std::copy(input.begin(), input.end(), buf1.begin()); // Use first buffer as input
            ((processLayer<I>(buf1, buf2, currentSize)), ...);

            // Determine buffer containing the final output
            constexpr size_t lastLayerIndex = sizeof...(Layers) - 1;
            const auto &outputBuffer = selectOutputBuffer < lastLayerIndex % 2 == 0 > (buf1, buf2);

            // Copy final output into a fixed-size array
            std::array<float, outputSize> result{};
            std::copy(outputBuffer.begin(), outputBuffer.begin() + outputSize, result.begin());
            return result;
        }

        template <size_t LayerIndex>
        void processLayer(
            std::array<float, maxLayerOutputSize> &buf1,
            std::array<float, maxLayerOutputSize> &buf2,
            size_t &currentSize) const
        {
            auto &layer = std::get<LayerIndex>(layers);
            auto &inBuf = selectInputBuffer < LayerIndex % 2 == 0 > (buf1, buf2);
            auto &outBuf = selectOutputBuffer < LayerIndex % 2 == 0 > (buf1, buf2);

            // Run forward pass of the current layer
            layer.forward(std::span(inBuf.data(), currentSize), std::span(outBuf.data(), layer.outputSize));
            currentSize = layer.outputSize; // Update current size to match layer's output
        }

        template <bool useBuf1>
        static constexpr std::array<float, maxLayerOutputSize> &selectInputBuffer(
            std::array<float, maxLayerOutputSize> &buf1,
            std::array<float, maxLayerOutputSize> &buf2)
        {
            if constexpr (useBuf1)
                return buf1;
            else
                return buf2;
        }

        template <bool useBuf1>
        static constexpr std::array<float, maxLayerOutputSize> &selectOutputBuffer(
            std::array<float, maxLayerOutputSize> &buf1,
            std::array<float, maxLayerOutputSize> &buf2)
        {
            if constexpr (useBuf1)
                return buf2;
            else
                return buf1;
        }
    };

} // namespace polann::models
