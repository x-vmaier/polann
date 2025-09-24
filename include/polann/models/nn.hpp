#pragma once

#include <span>
#include <vector>
#include <random>
#include <stdexcept>
#include "polann/core/layer.hpp"
#include "polann/utils/activation_functions.hpp"

namespace polann::models
{
    class NN
    {
    public:
        NN();
        ~NN() = default;

        void addLayer(core::LayerType type, float (*activationFunction)(float), int neuronCount);
        std::vector<float> predict(const std::vector<float> &inputs);

    private:
        std::mt19937 rng_;
        std::uniform_real_distribution<float> dist_;

        std::vector<float> weights_;                // Indexing: weights_[layerOffset + neuronOffset + input]
        std::vector<float> biases_;                 // Indexing: biases_[layerOffset + neuronOffset]
        std::vector<float (*)(float)> activations_; // Indexing: activations_[layerOffset + neuronOffset]
        std::vector<size_t> layerWeightOffsets_;    // Fast access for layer offsets in the weights_ vector
        std::vector<core::Layer> layers_;

        // Compute index into flat weights_ vector
        [[nodiscard]] inline size_t weightIndex(size_t layerOffset, size_t neuronOffset, size_t inputOffset) const
        {
            size_t prevLayerSize = layers_.at(layerOffset - 1).neuronCount;
            size_t base = layerWeightOffsets_.at(layerOffset);
            return base + neuronOffset * prevLayerSize + inputOffset;
        }

        // Compute index into flat bias vector
        [[nodiscard]] inline size_t biasIndex(size_t layerOffset, size_t neuronOffset) const
        {
            size_t idx = neuronOffset;
            for (size_t i = 1; i < layerOffset; i++)
                idx += layers_.at(i).neuronCount;
            return idx;
        }

        // Compute index into flat activation vector
        [[nodiscard]] inline size_t activationIndex(size_t layerOffset, size_t neuronOffset) const
        {
            size_t idx = neuronOffset;
            for (size_t i = 1; i < layerOffset; i++)
                idx += layers_.at(i).neuronCount;
            return idx;
        }

        std::vector<float> handleDenseLayer(const std::vector<float> &inputs, size_t layerOffset);
    };
} // namespace polann::models
