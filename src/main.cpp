#include <iostream>
#include "models/ann.hpp"
#include "utils/activation.hpp"
#include "utils/io.hpp"

int main(int argc, char *argv[])
{
    models::ANN model;
    model.addLayer(core::LayerType::INPUT_LAYER, nullptr, 2);
    model.addLayer(core::LayerType::DENSE_LAYER, utils::activation::sigmoid, 6);
    model.addLayer(core::LayerType::DENSE_LAYER, utils::activation::sigmoid, 1);

    // Predict with untrained model
    std::vector<float> inputData = {0.164f, 0.493f};
    std::cout << "Prediction: " << model.predict(inputData) << std::endl;
}
