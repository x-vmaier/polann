#include <iostream>
#include <matplot/matplot.h>
#include "polann/core/dataset.hpp"
#include "polann/layers/dense.hpp"
#include "polann/optimizers/sgd.hpp"
#include "polann/core/model_builder.hpp"
#include "polann/utils/activation_functions.hpp"
#include "polann/loss/mse.hpp"
#include "polann/utils/io.hpp"

using namespace polann::core;
using namespace polann::layers;
using namespace polann::optimizers;
using namespace polann::utils;
using namespace polann::loss;

/**
 * @brief Generates a dataset of points labeled inside or outside a circle
 *
 * @param radius Circle radius
 * @param range Max noise applied to coordinates
 * @param samples Number of samples to generate
 *
 * @return Dataset with InputSize = 2, OutputSize = 1
 */
Dataset<2, 1> circleDataset(float radius, float range, size_t samples)
{
    Dataset<2, 1> dataset;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(-range, range);

    for (size_t i = 0; i < samples; ++i)
    {
        float x = dist(rng);
        float y = dist(rng);

        std::array<float, 2> coordinate = {x, y};
        float distance = std::sqrt(x * x + y * y);
        std::array<float, 1> inCircle = {distance < radius ? 1.0f : 0.0f};

        dataset.addSample(coordinate, inCircle);
    }

    return dataset;
}

int main()
{
    // Generate dataset
    auto dataset = circleDataset(0.6f, 0.7f, 1000);

    // Create a simple neural network
    auto model = ModelBuilderRoot()
                     .addLayer<Dense<ReLU, 2, 64>>()
                     .addLayer<Dense<ReLU, 64, 32>>()
                     .addLayer<Dense<Sigmoid, 32, 1>>()
                     .build();

    // Train model on dataset
    SGD optimizer(0.1f);
    model.fit(dataset, optimizer, 100, 32);

    // Test with some input
    std::array<float, 2> inputs = {0.43f, 0.22f};
    std::cout << "Input: " << inputs << std::endl;

    // Make prediction
    std::array<float, 1> outputs = model.predict(inputs);
    std::cout << "Output: " << outputs << std::endl;

    const int N = 100; // grid resolution
    std::vector<std::vector<double>> Z(N, std::vector<double>(N, 0.0));

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double x = -1.0 + 2.0 * i / (N - 1); // range -1 to 1
            double y = -1.0 + 2.0 * j / (N - 1);

            std::array<float, 2> input = {static_cast<float>(x), static_cast<float>(y)};
            std::array<float, 1> output = model.predict(input);

            Z[j][i] = static_cast<double>(output[0]);
        }
    }

    matplot::imagesc(Z);
    matplot::colorbar();
    matplot::xlabel("x");
    matplot::ylabel("y");
    matplot::show();

    return 0;
}
