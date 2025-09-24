#include <iostream>
#include <vector>
#include "polann/models/nn.hpp"
#include "polann/utils/activation_functions.hpp"
#include "polann/utils/io.hpp"
#include "polann/config.h"

int main()
{
    // Create a simple neural network
    polann::models::NN network;
    network.addLayer(polann::core::LayerType::INPUT_LAYER, nullptr, 2);                                      // Add input layer
    network.addLayer(polann::core::LayerType::DENSE_LAYER, polann::utils::activation_functions::relu, 4);    // Add hidden layer
    network.addLayer(polann::core::LayerType::DENSE_LAYER, polann::utils::activation_functions::sigmoid, 1); // Add output layer

    // Test with some input
    std::vector<float> inputs = {0.5f, -0.2f, 0.8f};
    std::cout << "Input: " << inputs << std::endl;

    // Make prediction
    std::vector<float> outputs = network.predict(inputs);
    std::cout << "Output: " << outputs << std::endl;

    return 0;
}