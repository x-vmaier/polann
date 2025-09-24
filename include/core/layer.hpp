#pragma once

namespace core
{
    enum class LayerType
    {
        INPUT_LAYER,
        DENSE_LAYER
    };

    struct Layer
    {
        LayerType type;  // Behaviour and role of this layer
        int neuronCount; // Number of neurons in this layer

        Layer(LayerType type, int neuronCount) : type(type), neuronCount(neuronCount) {}
    };
} // namespace core
