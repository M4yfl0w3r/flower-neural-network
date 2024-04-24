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
    //     auto result = Tensor<std::size_t, Rows, Cols>();
    //     result.fill(0.0f);

    //     for (auto i = 0u; i < Rows; ++i)
    //         result.fillAt(i, tensor.at(i, 0u), 1u);

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

        for (const auto& [index, arg] : maxIndices | std::views::enumerate )
        {
            if (arg == labels.at(index, 0u))
                ++correctPredictions; 
        }

        return correctPredictions / rows;
    }

    template <typename Type, std::size_t Rows, std::size_t Cols>
    class CategoricalCrossEntropy
    {
        using Inputs    = Tensor<Type, Rows, Cols>;
        using Labels    = Tensor<std::size_t, Rows, 1>;
        using Gradients = Tensor<Type, Rows, 1>;

    public:
        [[nodiscard]] constexpr auto forward(const Inputs& input, const Labels& labels)
        {
            m_forwardInput = input;

            auto confidences = Tensor<Type, Rows, 1u>{};

            for (const auto& [index, row] : input.data() | std::views::enumerate) 
            {
                confidences.fillAt(index, 0u, row.at(labels.at(index, 0u)));
            }

            // Clip to prevent log(0.0)
            confidences.clip(std::numeric_limits<Type>::min(), 1 - std::numeric_limits<Type>::min());
            confidences.log();
            confidences.negative();

            return confidences.mean();
        }
        
        // [[nodiscard]] constexpr auto backward(const Gradients& gradients, const Labels& labels)
        // {
        //
        // }


    private:
        Tensor<Type, Rows, 1> m_targetClasses;
        Tensor<Type, Rows, Cols> m_forwardInput;
    };
}

