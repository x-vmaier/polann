#include <vector>
#include <iostream>
#include "polann/layers/dense.hpp"
#include "polann/core/model_builder.hpp"
#include "polann/utils/activation_functions.hpp"
#include "polann/utils/io.hpp"

using namespace polann::core;
using namespace polann::layers;
using namespace polann::utils;

int main()
{
    // Create a simple neural network
    auto model = ModelBuilderRoot()
                     .addLayer<Dense<activation_functions::relu, 2, 128>>()
                     .addLayer<Dense<activation_functions::relu, 128, 64>>()
                     .addLayer<Dense<activation_functions::sigmoid, 64, 1>>()
                     .build();

    // Test with some input
    std::vector<float> inputs = {0.43f, 0.22f};
    std::cout << "Input: " << inputs << std::endl;

    // Make prediction
    std::vector<float> outputs = model.predict(inputs);
    std::cout << "Output: " << outputs << std::endl;

    return 0;
}
