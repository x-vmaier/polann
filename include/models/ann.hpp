#pragma once

#include <span>
#include <vector>
#include <random>
#include <stdexcept>
#include "core/layer.hpp"
#include "utils/activation_functions.hpp"

namespace models
{
    class ANN
    {
    public:
        ANN() : rng_(std::random_device{}()),
                dist_(0.0f, 1.0f) {}

        ~ANN() = default;

        void addLayer(core::LayerType type, float (*activationFunction)(float), int neuronCount)
        {
            // Check for model initialization bugs
            if (type == core::LayerType::INPUT_LAYER && !layers_.empty())
                throw std::logic_error("Model only allows one input layer!");

            layers_.emplace_back(core::Layer{type, neuronCount});

            // Offset bookkeeping for weights
            size_t offset = weights_.size();
            layerWeightOffsets_.push_back(offset);

            if (type == core::LayerType::INPUT_LAYER)
                return; // Input layer has no linear or non-linear component

            // Initialize neurons
            for (int n = 0; n < neuronCount; n++)
            {
                biases_.push_back(dist_(rng_));
                activations_.push_back(activationFunction);

                // Neuron has number of neurons in previous layer weights
                for (int p = 0; p < layers_.at(layers_.size() - 2).neuronCount; p++)
                    weights_.push_back(dist_(rng_));
            }
        }

        std::vector<float> predict(const std::vector<float> &inputs)
        {
            // Check if input layer exists
            if (layers_.empty() || layers_.at(0).type != core::LayerType::INPUT_LAYER)
                throw std::runtime_error("No Input Layer specified!");

            // Outputs of the previous layer created outside the loop to
            // persist long enough for the next iteration.
            // First layer passes inputs to each neuron in the following layer,
            // thus the outputs for the next layer is the input passed by the function call.
            std::vector<float> outputs = inputs;

            // Feed forward pass
            for (size_t layerOffset = 1; layerOffset < layers_.size(); layerOffset++)
            {
                switch (layers_.at(layerOffset).type)
                {
                case core::LayerType::DENSE_LAYER:
                    outputs = handle_dense_layer(outputs, layerOffset);
                    break;

                case core::LayerType::INPUT_LAYER:
                    throw std::logic_error("More than one input layer defined!");

                default:
                    throw std::logic_error("Layer type not implemented yet!");
                }
            }

            // Return outputs from last layer (= output layer)
            return outputs;
        }

    private:
        std::mt19937 rng_;
        std::uniform_real_distribution<float> dist_;

        std::vector<float> weights_;                // Indexing: weights_[layerOffset + neuronOffset + input]
        std::vector<float> biases_;                 // Indexing: biases_[layerOffset + neuronOffset]
        std::vector<float (*)(float)> activations_; // Indexing: activations_[layerOffset + neuronOffset]
        std::vector<size_t> layerWeightOffsets_;    // Fast access for layer offsets in the weights_ vector
        std::vector<core::Layer> layers_;

        // Compute index into flat weights_ vector
        [[nodiscard]] inline size_t weight_index(size_t layerOffset, size_t neuronOffset, size_t inputOffset) const
        {
            size_t prevLayerSize = layers_.at(layerOffset - 1).neuronCount;
            size_t base = layerWeightOffsets_.at(layerOffset);
            return base + neuronOffset * prevLayerSize + inputOffset;
        }

        // Compute index into flat bias vector
        [[nodiscard]] inline size_t bias_index(size_t layerOffset, size_t neuronOffset) const
        {
            size_t idx = neuronOffset;
            for (size_t i = 1; i < layerOffset; i++)
                idx += layers_.at(i).neuronCount;
            return idx;
        }

        // Compute index into flat activation vector
        [[nodiscard]] inline size_t activation_index(size_t layerOffset, size_t neuronOffset) const
        {
            size_t idx = neuronOffset;
            for (size_t i = 1; i < layerOffset; i++)
                idx += layers_.at(i).neuronCount;
            return idx;
        }

        std::vector<float> handle_dense_layer(const std::vector<float> &inputs, size_t layerOffset)
        {
            std::vector<float> outputs(layers_.at(layerOffset).neuronCount); // Avoid allocation per push_back
            for (size_t neuronOffset = 0; neuronOffset < outputs.size(); neuronOffset++)
            {
                // Add bias to sum
                float sum = biases_.at(bias_index(layerOffset, neuronOffset));

                // Add weighted inputs to sum
                for (size_t input = 0; input < inputs.size(); input++)
                {
                    size_t weight_idx = weight_index(layerOffset, neuronOffset, input);
                    sum += inputs.at(input) * weights_.at(weight_idx);
                }

                // Add neuron activation to the outputs
                outputs[neuronOffset] = activations_.at(activation_index(layerOffset, neuronOffset))(sum);
            }

            return outputs;
        }
    };
} // namespace models
