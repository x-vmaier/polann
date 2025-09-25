#pragma once

#include <tuple>
#include <utility>
#include "polann/models/nn.hpp"

namespace polann::core
{
    template <typename... Layers>
    struct ModelBuilder
    {
        std::tuple<Layers...> layers;

        // constructor from tuple
        explicit ModelBuilder(std::tuple<Layers...> ls)
            : layers(std::move(ls)) {}

        // default empty builder
        ModelBuilder() = default;

        template <typename NewLayer, typename... Args>
        auto addLayer(Args &&...args)
        {
            auto newLayer = NewLayer(std::forward<Args>(args)...);                        // Forward args to layer
            auto newTuple = std::tuple_cat(layers, std::make_tuple(std::move(newLayer))); // Extend model builder with layer
            return ModelBuilder<Layers..., NewLayer>(std::move(newTuple));                // Return extended model builder
        }

        auto build()
        {
            return std::apply(
                [](auto &&...ls)
                { return polann::models::NN<Layers...>(ls...); },
                layers);
        }
    };

    // Start empty builder
    inline auto ModelBuilderRoot()
    {
        return ModelBuilder<>();
    }

} // namespace polann::core
