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

        for (auto i = 0u; const auto& row : input.data())        
        {
            maxIndices.at(i++) = static_cast<std::size_t>(std::ranges::distance(std::begin(row), 
                                                          std::ranges::max_element(row)));
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
        using InputTensor = Tensor<Type, Rows, Cols>;
        using LabelsTensor = Tensor<std::size_t, Rows, 1>;

    public:
        [[nodiscard]] constexpr auto forward(const InputTensor& input, const LabelsTensor& labels)
        {
            m_forwardInput = input;

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
        
        [[nodiscard]] constexpr auto backward();

    private:
        Tensor<Type, Rows, 1> m_targetClasses;
        Tensor<Type, Rows, Cols> m_forwardInput;
    };
}

