module;

import std;
import tensor;
import config;

export module loss;

export namespace Loss 
{
    enum class Loss
    {
        CategoricalCrossEntropy
    };

    inline constexpr auto oneHotEncoding = []<typename T, std::size_t R, std::size_t C>(const auto& labels) {
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
        for (const auto& row : input.data()) {
            maxIndices.push_back(static_cast<std::size_t>(std::ranges::distance(std::begin(row), std::ranges::max_element(row))));
        }

        auto correctPredictions = 0u;

        for (auto i = 0u; const auto& arg : maxIndices) {
            if (arg == labels.at(i, 0u))
                ++correctPredictions;
            ++i;
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

            for (auto i = 0u; auto& row : predictions.data()) {
                confidences.fillAt(i, 0u, row.at(trueLabels.at(i, 0u)));
            }

            // Clip to prevent log(0.0)
            confidences.clip(std::numeric_limits<Type>::min(), 1 - std::numeric_limits<Type>::min());
            confidences.log();
            confidences.negative();
            return confidences.mean();
        }

        [[nodiscard]] constexpr auto backward(const Inputs& gradients) {
            auto labels = oneHotEncoding.operator()<float, 1, Mayflower::Config::numClasses>(m_trueLabels);
            auto output = labels / gradients;
            output.negative(); // TODO: Add - operator to the Tensor class
            return output;
        }

    private:
        Labels m_trueLabels;
    };
}

