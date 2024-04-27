module;

import tensor;
import config;

#include <algorithm>
#include <ranges>
#include <limits>

export module loss;

export namespace Loss 
{
    enum class Loss
    {
        CategoricalCrossEntropy
    };

    static constexpr auto oneHotEncoding = []<typename T, std::size_t R, std::size_t C>(const auto& labels) {
        auto result = Tensor<T, R, C>{ 0 };

        // TODO: views::to?
        for (auto i = 0u; i < R; ++i) {
            result.fillAt(i, labels.at(i, 0u), static_cast<T>(1));
        }

        return result;
    };

    [[nodiscard]] constexpr auto accuracy(const auto& input, const auto& labels) {
        const auto rows = labels.shape().first;
        auto maxIndices = std::vector<std::size_t>{};

        // TODO: Merge it to one for
        // for (const auto& row : input.data()) {
        //     maxIndices.push_back(static_cast<std::size_t>(std::ranges::distance(std::begin(row), std::ranges::max_element(row))));
        // }

        auto correctPredictions = 0u;

        for (const auto& [index, arg] : maxIndices | std::views::enumerate) {
            if (arg == labels.at(index, 0u))
                ++correctPredictions; 
        }

        return correctPredictions / rows;
    }

    template <typename Type, std::size_t Rows, std::size_t Cols>
    class CategoricalCrossEntropy final
    {
        using Inputs = Tensor<Type, Rows, Cols>;
        using Labels = Tensor<std::size_t, Rows, 1u>;

    public:
        [[nodiscard]] constexpr auto value(const Inputs& predictions, const Labels& trueLabels) {
            m_trueLabels     = trueLabels;
            auto confidences = Tensor<Type, Rows, 1u>{};

            for (const auto& [index, row] : predictions.data() | std::views::enumerate) {
                confidences.fillAt(index, 0u, row.at(trueLabels.at(index, 0u)));
            }

            // Clip to prevent log(0.0)
            confidences.clip(std::numeric_limits<Type>::min(), 1 - std::numeric_limits<Type>::min());
            confidences.log();
            confidences.negative();
            return confidences.mean();
        }

        [[nodiscard]] constexpr auto backward(const Inputs& gradients) {
            auto labels = oneHotEncoding.operator()<float, 1, Config::numClasses>(m_trueLabels);
            auto output = labels / gradients;
            output.negative(); // TODO: Add - operator to the Tensor class
            return output;
        }

    private:
        Labels m_trueLabels;
    };
}

