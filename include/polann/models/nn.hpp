#pragma once

#include <tuple>
#include <vector>
#include <span>

namespace polann::models
{
    template <typename... Layers>
    class NN
    {
    public:
        explicit NN(Layers... ls) : layers(std::move(ls)...) {}

        std::vector<float> predict(std::span<const float> input) const
        {
            return predict_impl(input, std::index_sequence_for<Layers...>{});
        }

    private:
        std::tuple<Layers...> layers;

        template <size_t... I>
        std::vector<float> predict_impl(std::span<const float> input, std::index_sequence<I...>) const
        {
            std::vector<float> x(input.begin(), input.end());
            ((x = std::get<I>(layers).forward(x)), ...); // fold expression = frozen switch
            return x;
        }
    };

} // namespace polann::models
