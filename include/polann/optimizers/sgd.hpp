#pragma once

#include <vector>

namespace polann::optimizers
{
    /**
     * @brief Stochastic Gradient Descent (SGD) optimizer
     *
     * Updates weights and biases in the opposite direction
     * of their gradients, scaled by learning rate.
     */
    struct SGD
    {
        float learningRate;

        explicit SGD(float lr) : learningRate(lr) {}

        template <typename Layer>
        void step(Layer &layer)
        {
            // Update weights
            for (size_t i = 0; i < layer.weights.size(); ++i)
                layer.weights[i] -= learningRate * layer.gradWeights[i];

            // Update biases
            for (size_t i = 0; i < layer.biases.size(); ++i)
                layer.biases[i] -= learningRate * layer.gradBiases[i];
        }
    };

} // namespace polann::optimizers
