module;

import std;
import tensor;
import config;
import dense_layer;

export module loss;

namespace Loss 
{
    enum class LossFunction
    {
        CategoricalCrossEntropy
    };

    static constexpr auto oneHotEncoding = []<typename T, std::size_t R, std::size_t C>(const auto& labels) {
        auto result = Tensor<T, TensorParams{ R, C }>{ T{} };

        for (auto i = 0u; i < R; ++i) {
            result.fillAt(i, labels.at(i, 0u), static_cast<T>(1));
        }

        return result;
    };

    export [[nodiscard]] constexpr auto accuracy(const auto& input, const auto& labels) {
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

    // That's bad, will change it for sure
    export template<typename T>
    class CategoricalCrossEntropy final
    {
    public:
        template<LayerParams prevLayer>
        [[nodiscard]] constexpr auto forward(
            const Tensor<T, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }>& predictions, 
            const Tensor<std::size_t, TensorParams{ prevLayer.Inputs, 1u }>& trueLabels
        )
        {
            m_trueLabels     = trueLabels;
            auto confidences = Tensor<T, TensorParams{ prevLayer.Inputs, 1u }>{};

            for (auto i = 0u; auto& row : predictions.data()) {
                confidences.fillAt(i, 0u, row.at(trueLabels.at(i, 0u)));
            }

            // Clip to prevent log(0.0)
            confidences.clip(std::numeric_limits<T>::min(), 1 - std::numeric_limits<T>::min());
            confidences.log();
            confidences.negative();
            return confidences.mean();
        }

        template<LayerParams nextLayer>
        [[nodiscard]] constexpr auto backward(
            const Tensor<T, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }>& gradients
        ) 
        {
            auto labels = oneHotEncoding.operator()<float, 1, Mayflower::Config::numClasses>(m_trueLabels);
            auto output = labels / gradients;
            output.negative(); // TODO: Add - operator to the Tensor class
            return output;
        }

    private:
        Tensor<std::size_t, TensorParams{ 1u, 1u }> m_trueLabels;
    };
}

