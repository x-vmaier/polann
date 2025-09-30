#pragma once

#include <span>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <filesystem>

namespace polann::core
{
    /**
     * @brief Dataset structure for neural network training
     *
     * @tparam InputSize Number of features per input sample
     * @tparam OutputSize Number of features per output sample
     */
    template <size_t InputSize, size_t OutputSize>
    struct Dataset
    {
        std::vector<float> inputs;   /// Flattened row-major input matrix: numSamples * InputSize
        std::vector<float> outputs;  /// Flattened row-major output matrix: numSamples * OutputSize
        std::vector<size_t> indices; /// Shuffled indices for batching
        size_t numSamples = 0;

        // Batch buffers to avoid repeated allocations
        mutable std::vector<float> batchInputBuffer;
        mutable std::vector<float> batchOutputBuffer;

        // File I/O interface
        virtual void fromFile(const std::filesystem::path &path) {}
        virtual void toFile(const std::filesystem::path &path) const {}

        /**
         * @brief Add a sample using spans
         *
         * @param in Span of input features (must be InputSize)
         * @param out Span of output features (must be OutputSize)
         */
        void addSample(std::span<const float> in, std::span<const float> out)
        {
            if (in.size() != InputSize || out.size() != OutputSize)
                throw std::invalid_argument("Input/output size mismatch");

            inputs.insert(inputs.end(), in.begin(), in.end());
            outputs.insert(outputs.end(), out.begin(), out.end());
            indices.push_back(numSamples++);
        }

        /**
         * @brief Add a sample using compile-time sized arrays
         *
         * @tparam InSize Input array size (must be InputSize)
         * @tparam OutSize Output array size (must be OutputSize)
         */
        template <size_t InSize, size_t OutSize>
        void addSample(const std::array<float, InSize> &in, const std::array<float, OutSize> &out)
        {
            static_assert(InSize == InputSize, "Input size mismatch");
            static_assert(OutSize == OutputSize, "Output size mismatch");

            inputs.insert(inputs.end(), in.begin(), in.end());
            outputs.insert(outputs.end(), out.begin(), out.end());
            indices.push_back(numSamples++);
        }

        void shuffle()
        {
            std::random_device rd;
            std::default_random_engine gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        void shuffle(unsigned int seed)
        {
            std::default_random_engine gen(seed);
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        size_t size() const { return numSamples; }

        /**
         * @brief Compute the number of batches for a given batch size
         *
         * @param batchSize Number of samples per batch
         * @return Number of batches needed to cover the dataset
         */
        size_t numBatches(size_t batchSize) const
        {
            if (batchSize == 0)
                throw std::invalid_argument("Batch size cannot be zero");
            return (numSamples + batchSize - 1) / batchSize;
        }

        /**
         * @brief Get a batch of inputs and outputs as contiguous spans
         *
         * @param batchIndex Index of the batch (0-based)
         * @param batchSize Maximum number of samples in this batch
         * @return Pair of flattened row-major spans: (inputs, outputs)
         */
        std::pair<std::span<const float>, std::span<const float>> getBatch(size_t batchIndex, size_t batchSize) const
        {
            if (batchSize == 0)
                throw std::invalid_argument("Batch size cannot be zero");

            if (batchIndex >= numBatches(batchSize))
                throw std::out_of_range("Batch index out of range");

            size_t startSample = batchIndex * batchSize;
            size_t endSample = std::min(startSample + batchSize, numSamples);
            size_t actualBatchSize = endSample - startSample;

            // Resize buffers if needed
            size_t requiredInputSize = actualBatchSize * InputSize;
            size_t requiredOutputSize = actualBatchSize * OutputSize;

            if (batchInputBuffer.size() < requiredInputSize)
                batchInputBuffer.resize(requiredInputSize);

            if (batchOutputBuffer.size() < requiredOutputSize)
                batchOutputBuffer.resize(requiredOutputSize);

            // Gather samples according to shuffled indices
            for (size_t i = 0; i < actualBatchSize; ++i)
            {
                size_t sampleIdx = indices[startSample + i];
                std::copy_n(inputs.data() + sampleIdx * InputSize, InputSize, batchInputBuffer.data() + i * InputSize);
                std::copy_n(outputs.data() + sampleIdx * OutputSize, OutputSize, batchOutputBuffer.data() + i * OutputSize);
            }

            return {
                std::span(batchInputBuffer.data(), requiredInputSize),
                std::span(batchOutputBuffer.data(), requiredOutputSize)};
        }

        /**
         * @brief Reserve memory for expected number of samples
         */
        void reserve(size_t expectedSamples)
        {
            inputs.reserve(expectedSamples * InputSize);
            outputs.reserve(expectedSamples * OutputSize);
            indices.reserve(expectedSamples);
        }
    };

} // namespace polann::core
