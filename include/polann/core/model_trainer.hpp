#pragma once

#include <algorithm>
#include <random>

namespace polann::core
{
    template <typename Optimizer>
    struct ModelTrainer
    {
        size_t batchSize;
        float learningRate;
        size_t epochs;
        Optimizer optimizer;

        template <typename Model, typename Dataset>
        void train(Model &model, const Dataset &dataset)
        {
            // 1. Randomly shuffle dataset into batches
            // 2. Forward + loss
            // 3. Backward
            // 4. Optimizer step
        }
    };

} // namespace polann::core
