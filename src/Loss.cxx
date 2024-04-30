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

    static constexpr auto oneHotEncoding = []<std::size_t R, std::size_t C>(const auto& labels) {
        auto result = Tensor<float, TensorParams{ R, C }>{ 0.0f };

        for (auto i = 0u; i < R; ++i) {
            result.fillAt(i, labels.at(i, 0u), 1.0f);
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

    export class CategoricalCrossEntropy final
    {
    public:
        template<LayerParams prevLayer>
        [[nodiscard]] constexpr auto forward(
            const Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }>& predictions, 
            const Tensor<std::size_t, TensorParams{ prevLayer.Inputs, 1u }>& trueLabels
        )
        {
            m_trueLabels     = trueLabels;
            auto confidences = Tensor<float, TensorParams{ prevLayer.Inputs, 1u }>{};

            for (auto i = 0u; auto& row : predictions.data()) {
                confidences.fillAt(i, 0u, row.at(trueLabels.at(i, 0u)));
            }

            // Clip to prevent log(0.0)
            confidences.clip(std::numeric_limits<float>::min(), 1 - std::numeric_limits<float>::min());
            confidences.log();
            confidences.negative();
            return confidences.mean();
        }

        template<LayerParams nextLayer>
        [[nodiscard]] constexpr auto backward(
            const Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }>& gradients
        ) 
        {
            auto labels = oneHotEncoding.operator()<1, Mayflower::Config::numClasses>(m_trueLabels);
            auto output = labels / gradients;
            output.negative(); // TODO: Add - operator to the Tensor class
            return output;
        }

    private:
        Tensor<std::size_t, TensorParams{ 1u, 1u }> m_trueLabels;
    };
}

