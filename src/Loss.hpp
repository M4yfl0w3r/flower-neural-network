#pragma once

#include "Tensor.hpp"

#include <algorithm>
#include <ranges>
#include <limits>

namespace Mayflower
{
    enum class Loss 
    {
        CategoricalCrossEntropy
    };

    // template <std::size_t Rows, std::size_t Cols>
    // [[nodiscard]] constexpr auto oneHotEncoding(const Tensor<std::size_t, Rows, 1u>& tensor)
    // {
    //     auto result = Tensor<std::size_t, Rows, Cols>{ 0 };
    //
    //     for (auto i = 0u; i < Rows; ++i)
    //         result.fillAt(i, tensor.at(i, 0u), 1u);
    //
    //     return result;
    // }

    [[nodiscard]] constexpr auto accuracy(const auto& input, const auto& labels) 
    {
        const auto rows = labels.shape().first;
        auto maxIndices = std::vector<std::size_t>{};

        for (const auto& row : input.data()) 
        {
            maxIndices.push_back(static_cast<std::size_t>(std::ranges::distance(std::begin(row), 
                                                          std::ranges::max_element(row))));
        }


        auto correctPredictions = 0u;

        for (const auto& [index, arg] : maxIndices | std::views::enumerate) 
        {
            if (arg == labels.at(index, 0u))
                ++correctPredictions; 
        }

        return correctPredictions / rows;
    }

    template <typename Type, std::size_t Rows, std::size_t Cols>
    class CategoricalCrossEntropy final
    {
        using Inputs    = Tensor<Type, Rows, Cols>;
        using Labels    = Tensor<std::size_t, Rows, 1u>;
        using Gradients = Tensor<Type, Rows, Cols>;

    public:
        constexpr auto forward(const Inputs& predictions, const Labels& trueLabels) 
        {
            auto confidences = Tensor<Type, Rows, 1u>{};

            for (const auto& [index, row] : predictions.data() | std::views::enumerate) 
            {
                confidences.fillAt(index, 0u, row.at(trueLabels.at(index, 0u)));
            }

            // Clip to prevent log(0.0)
            confidences.clip(std::numeric_limits<Type>::min(), 1 - std::numeric_limits<Type>::min());
            confidences.log();
            confidences.negative();
            return confidences.mean();
        }
    };
}

