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
     * @brief Template-based neural network
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
        static constexpr size_t inputSize = firstLayerType::inputSize;         /// Input size of the network
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
         * @param input Fixed-size input array for the first layer
         * @return std::array<float, outputSize> Output array produced by the final layer
         *
         * @note Internal buffers are reused to avoid heap allocations
         */
        template <size_t InputSize>
        [[nodiscard]] std::array<float, outputSize> predict(const std::array<float, InputSize> &input) const
        {
            static_assert(InputSize == inputSize, "Input size mismatch");

            alignas(32) std::array<float, maxLayerOutputSize> buf1{};
            alignas(32) std::array<float, maxLayerOutputSize> buf2{};
            std::copy(input.begin(), input.end(), buf1.begin()); // Use first buffer as input

            return predictImpl(buf1, buf2, std::index_sequence_for<Layers...>{});
        }

        /**
         * @brief Trains the model using mini-batch gradient descent
         *
         * @tparam Dataset Dataset type
         * @tparam Optimizer Optimizer type. Must implement step(layer)
         * @tparam LossFunction Loss function type. Must provide static compute() and gradient()
         *
         * @param dataset Training dataset
         * @param optimizer Optimizer instance (e.g., SGD)
         * @param epochs Number of full passes over dataset
         * @param batchSize Number of samples per training batch
         * @param shuffle Whether to shuffle dataset each epoch
         * @param verbose Whether to print training progress
         */
        template <typename Dataset, typename Optimizer, typename LossFunction = polann::loss::MSE>
        void fit(Dataset &dataset, Optimizer &optimizer, int epochs = 1, int batchSize = 32, bool shuffle = true, bool verbose = true)
        {
            for (size_t epoch = 0; epoch < epochs; epoch++)
            {
                if (shuffle) // Shuffling helps generalizing the model
                    dataset.shuffle();

                float epochLoss = 0.0f;
                size_t numBatches = dataset.numBatches(batchSize);
                size_t totalSamples = 0;

                for (size_t batch = 0; batch < numBatches; batch++)
                {
                    // Get data batches from the dataset
                    auto [batchInputs, batchLabels] = dataset.getBatch(batch, batchSize);
                    size_t currentBatchSize = batchInputs.size() / inputSize;

                    if (currentBatchSize == 0)
                        continue;

                    // Zero gradients at start of batch
                    std::apply([](auto &...layer) { ((layer.clearGradients()), ...); }, layers);

                    float batchLoss = 0.0f;

                    // Accululate gradients over all samples
                    for (size_t sample = 0; sample < currentBatchSize; sample++)
                    {
                        std::array<float, inputSize> input{};
                        std::array<float, outputSize> target{};

                        std::copy_n(batchInputs.data() + sample * inputSize, inputSize, input.begin());
                        std::copy_n(batchLabels.data() + sample * outputSize, outputSize, target.begin());

                        // Forward pass
                        auto prediction = predict(input);

                        // Loss
                        std::span<const float> predSpan(prediction);
                        std::span<const float> targetSpan(target);
                        batchLoss += LossFunction::compute(predSpan, targetSpan);

                        // Gradients
                        std::array<float, outputSize> dLoss{};
                        LossFunction::gradient(predSpan, targetSpan, std::span(dLoss));

                        // Backward pass
                        backward(dLoss);
                    }

                    // Scale gradients by 1/batchSize and update weights
                    float scale = 1.0f / currentBatchSize;
                    std::apply([&](auto &...layer) { ((layer.scaleGradients(scale)), ...); }, layers);
                    std::apply([&](auto &...layer) { ((optimizer.step(layer)), ...); }, layers);

                    epochLoss += batchLoss;
                    totalSamples += currentBatchSize;
                }

                if (totalSamples > 0)
                    epochLoss /= totalSamples;

                if (verbose && (epoch % 10 == 0 || epoch == epochs - 1))
                    std::cout << "Epoch " << epoch << "/" << epochs << ", Loss: " << epochLoss << std::endl;
            }
        }

    private:
        std::tuple<Layers...> layers;

        template <size_t... I>
        [[nodiscard]] std::array<float, outputSize> predictImpl(
            std::array<float, maxLayerOutputSize> &buf1,
            std::array<float, maxLayerOutputSize> &buf2,
            std::index_sequence<I...>) const
        {
            // Forward through layers using fold expression
            ((forwardLayer<I>(buf1, buf2)), ...);

            // Determine buffer containing the final output
            constexpr size_t lastLayerIndex = sizeof...(Layers) - 1;
            const auto &outputBuffer = selectOutputBuffer < lastLayerIndex % 2 == 0 > (buf1, buf2);

            // Copy final output into a fixed-size array
            std::array<float, outputSize> result{};
            std::copy(outputBuffer.begin(), outputBuffer.begin() + outputSize, result.begin());
            return result;
        }

        template <size_t LayerIndex>
        void forwardLayer(
            std::array<float, maxLayerOutputSize> &buf1,
            std::array<float, maxLayerOutputSize> &buf2) const
        {
            auto &layer = std::get<LayerIndex>(layers);

            // Alternate between buf1 and buf2 to avoid extra memory allocations
            // At each layer, one buffer serves as input and the other as output
            auto &inBuf = selectInputBuffer < LayerIndex % 2 == 0 > (buf1, buf2);
            auto &outBuf = selectOutputBuffer < LayerIndex % 2 == 0 > (buf1, buf2);

            // Run forward pass of the current layer
            layer.forward(inBuf, outBuf);
        }

        template <size_t OutputSize>
        void backward(const std::array<float, OutputSize> &dLoss)
        {
            static_assert(OutputSize == outputSize, "Input size mismatch");

            alignas(32) std::array<float, maxLayerOutputSize> buf1{};
            alignas(32) std::array<float, maxLayerOutputSize> buf2{};
            std::copy(dLoss.begin(), dLoss.end(), buf1.begin()); // Start with loss gradients in first buffer

            return backwardImpl(buf1, buf2, std::index_sequence_for<Layers...>{});
        }

        template <size_t... I>
        void backwardImpl(
            std::array<float, maxLayerOutputSize> &buf1,
            std::array<float, maxLayerOutputSize> &buf2,
            std::index_sequence<I...>)
        {
            // Process layers using reverse fold
            ((backwardLayer<sizeof...(Layers) - 1 - I>(buf1, buf2)), ...);
        }

        template <size_t LayerIndex>
        void backwardLayer(
            std::array<float, maxLayerOutputSize> &buf1,
            std::array<float, maxLayerOutputSize> &buf2)
        {
            auto &layer = std::get<LayerIndex>(layers);

            // For backward pass, gradient flows from output to input
            constexpr size_t reverseIndex = sizeof...(Layers) - 1 - LayerIndex;
            auto &gradOut = selectInputBuffer < reverseIndex % 2 == 0 > (buf1, buf2);
            auto &gradIn = selectOutputBuffer < reverseIndex % 2 == 0 > (buf1, buf2);

            // Run backward pass of the current layer
            layer.backward(gradOut, gradIn);
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
