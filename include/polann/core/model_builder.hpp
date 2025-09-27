#pragma once

#include <tuple>
#include <utility>
#include "polann/models/nn.hpp"

namespace polann::core
{
    /**
     * @brief Template-based builder for neural networks
     *
     * @tparam Layers... Types of layers already added to the builder
     */
    template <typename... Layers>
    class ModelBuilder
    {
    public:
        /**
         * @brief Construct from existing layers tuple
         *
         * @param ls Tuple of already constructed layers
         */
        explicit ModelBuilder(std::tuple<Layers...> ls)
            : layers(std::move(ls)) {}

        // default empty builder
        ModelBuilder() = default;

        /**
         * @brief Adds a new layer to the model architecture
         *
         * All arguments are forwarded to the constructor of NewLayer.
         *
         * @tparam NewLayer Struct defining the layer
         * @tparam Args Parameter pack forwarded to NewLayer constructor
         * @param args Constructor arguments for NewLayer
         * @return ModelBuilder<Layers..., NewLayer> containing the extended model
         */
        template <typename NewLayer, typename... Args>
        [[nodiscard]] auto addLayer(Args &&...args)
        {
            auto newLayer = NewLayer(std::forward<Args>(args)...);                        // Forward args to layer
            auto newTuple = std::tuple_cat(layers, std::make_tuple(std::move(newLayer))); // Extend tuple with new layer
            return ModelBuilder<Layers..., NewLayer>(std::move(newTuple));                // Return new builder with added layer
        }

        /**
         * @brief Builds the final neural network
         *
         * @return NN<Layers...> Fully inlined neural network
         */
        [[nodiscard]] auto build()
        {
            return std::apply(
                [](auto &&...ls)
                { return polann::models::NN<Layers...>(ls...); },
                layers);
        }

    private:
        std::tuple<Layers...> layers;
    };

    /**
     * @brief Creates an empty ModelBuilder
     *
     * @return ModelBuilder<> Empty builder
     */
    [[nodiscard]] inline auto ModelBuilderRoot()
    {
        return ModelBuilder<>();
    }

} // namespace polann::core
