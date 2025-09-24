#pragma once

#include <span>
#include <vector>
#include <random>
#include <stdexcept>
#include "core/layer.hpp"
#include "utils/activation.hpp"

namespace models
{
    class ANN
    {
    public:
        ANN() : _rng(std::random_device{}()),
                _dist(0.0f, 1.0f) {}

        ~ANN() = default;

        void addLayer(core::LayerType type, float (*activationFunction)(float), int neuronCount)
        {
            // Check for model initialization bugs
            if (type == core::LayerType::INPUT_LAYER && !_layers.empty())
                throw std::logic_error("Model only allows one input layer!");

            _layers.emplace_back(core::Layer{type, neuronCount});

            // Offset bookkeeping for weights
            size_t offset = _weights.size();
            _layerWeightOffsets.push_back(offset);

            if (type == core::LayerType::INPUT_LAYER)
                return; // Input layer has no linear or non-linear component

            // Initialize neurons
            for (int n = 0; n < neuronCount; n++)
            {
                _biases.push_back(_dist(_rng));
                _activations.push_back(activationFunction);

                // Neuron has number of neurons in previous layer weights
                for (int p = 0; p < _layers.at(_layers.size() - 2).neuronCount; p++)
                    _weights.push_back(_dist(_rng));
            }
        }

        std::vector<float> predict(const std::vector<float> &inputs)
        {
            // Check if input layer exists
            if (_layers.empty() || _layers.at(0).type != core::LayerType::INPUT_LAYER)
                throw std::runtime_error("No Input Layer specified!");

            // Outputs of the previous layer created outside the loop to
            // persist long enough for the next iteration.
            // First layer passes inputs to each neuron in the following layer,
            // thus the outputs for the next layer is the input passed by the function call.
            std::vector<float> outputs = inputs;

            // Feed forward pass
            for (size_t layerOffset = 1; layerOffset < _layers.size(); layerOffset++)
            {
                switch (_layers.at(layerOffset).type)
                {
                case core::LayerType::DENSE_LAYER:
                    outputs = handleDenseLayer(outputs, layerOffset);
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
        std::mt19937 _rng;
        std::uniform_real_distribution<float> _dist;

        std::vector<float> _weights;                // Indexing: _weights[layerOffset + neuronOffset + input]
        std::vector<float> _biases;                 // Indexing: _biases[layerOffset + neuronOffset]
        std::vector<float (*)(float)> _activations; // Indexing: _activations[layerOffset + neuronOffset]
        std::vector<size_t> _layerWeightOffsets;    // Fast access for layer offsets in the _weights vector
        std::vector<core::Layer> _layers;

        // Compute index into flat _weights vector
        inline size_t weightIndex(size_t layerOffset, size_t neuronOffset, size_t inputOffset) const
        {
            size_t prevLayerSize = _layers.at(layerOffset - 1).neuronCount;
            size_t base = _layerWeightOffsets.at(layerOffset);
            return base + neuronOffset * prevLayerSize + inputOffset;
        }

        // Compute index into flat _bias vector
        inline size_t biasIndex(size_t layerOffset, size_t neuronOffset) const
        {
            size_t idx = neuronOffset;
            for (size_t i = 1; i < layerOffset; i++)
                idx += _layers.at(i).neuronCount;
            return idx;
        }

        // Compute index into flat _activation vector
        inline size_t activationIndex(size_t layerOffset, size_t neuronOffset) const
        {
            size_t idx = neuronOffset;
            for (size_t i = 1; i < layerOffset; i++)
                idx += _layers.at(i).neuronCount;
            return idx;
        }

        std::vector<float> handleDenseLayer(const std::vector<float> &inputs, size_t layerOffset)
        {
            std::vector<float> outputs(_layers.at(layerOffset).neuronCount); // Avoid allocation per push_back
            for (size_t neuronOffset = 0; neuronOffset < outputs.size(); neuronOffset++)
            {
                // Add bias to sum
                float sum = _biases.at(biasIndex(layerOffset, neuronOffset));

                // Add weighted inputs to sum
                for (size_t input = 0; input < inputs.size(); input++)
                {
                    size_t wIdx = weightIndex(layerOffset, neuronOffset, input);
                    sum += inputs.at(input) * _weights.at(wIdx);
                }

                // Add neuron activation to the outputs
                outputs[neuronOffset] = _activations.at(activationIndex(layerOffset, neuronOffset))(sum);
            }

            return outputs;
        }
    };

} // namespace models
