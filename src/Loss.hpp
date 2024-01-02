#pragma once

#include "Tensor.hpp"

#include <algorithm>
#include <limits>

namespace Mayflower
{
    enum class Loss 
    {
        CategoricalCrossEntropy
    };

    template <std::size_t Rows, std::size_t Cols>
    [[nodiscard]] constexpr auto oneHotEncoding(const Tensor<std::size_t, Rows, 1u>& tensor)
    {
        auto result = Tensor<std::size_t, Rows, Cols>();
        result.fill(0.0f);

        for (auto i = 0u; i < Rows; ++i)
            result.fillAt(i, tensor.at(i, 0u), 1u);

        return result;
    }

    template <typename Type, std::size_t Rows, std::size_t Cols>
    [[nodiscard]] constexpr auto accuracy(const Tensor<Type, Rows, Cols>& input,
                                          const Tensor<std::size_t, Rows, 1u>& labels)
    {
        auto maxIndices = std::array<std::size_t, Rows>();

        // TODO: Change to std::views::zip when available
        for (auto i = 0u; const auto& row : input.data())        
        {
            maxIndices.at(i++) = static_cast<std::size_t>(std::ranges::distance(std::begin(row), std::ranges::max_element(row)));
        }

        auto correctPredictions = 0u;
        for (auto i = 0u; const auto& arg : maxIndices)
        {
            if (arg == labels.at(i, 0u))
                ++correctPredictions; 
        }

        return correctPredictions / Rows;
    }

    template <typename Type, std::size_t Rows, std::size_t Cols>
    class CategoricalCrossEntropy
    {
    public:
        [[nodiscard]] constexpr auto forward(const Tensor<Type, Rows, Cols>& input, 
                                             const Tensor<std::size_t, Rows, 1>& labels);
        
    private:
        Tensor<Type, Rows, 1> targetClasses;
        Tensor<Type, Rows, Cols> m_forwardInput;
    };


    template <typename Type, std::size_t Rows, std::size_t Cols>
    constexpr auto CategoricalCrossEntropy<Type, Rows, Cols>::forward(const Tensor<Type, Rows, Cols>& input,
                                                                      const Tensor<std::size_t, Rows, 1u>& labels)
    {
        m_forwardInput = input;

        // TODO: Change to std::views::zip function when C++23 available
        auto confidences = Tensor<Type, Rows, 1u>{};
        for (auto labelIndex = 0u; auto& row : input.data())
        {
            confidences.fillAt(labelIndex, 0u, row.at(labels.at(labelIndex, 0u)));
            ++labelIndex;
        }

        // Clip to prevent log(0.0)
        confidences.clip(std::numeric_limits<Type>::min(), 1 - std::numeric_limits<Type>::min());
        confidences.log();
        confidences.negative();
        return confidences.mean();
    }
}

