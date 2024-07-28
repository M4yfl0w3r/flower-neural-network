module;

import std;
import tensor;
import config;
import dense_layer;

export module loss;

namespace Loss
{
    static constexpr auto OneHotEncoding = []<int R, int C> [[nodiscard]] (const auto& labels)
    {
        auto result = Tensor<float, { R, C }>{ 0.0f };

        for (auto i : std::ranges::views::iota(0, R)) {
            result.FillAt(i, labels.At(i, 0), 1.0f);
        }

        return result;
    };

    export [[nodiscard]] constexpr auto Accuracy(const auto* input, const auto& labels)
    {
        const auto rows         = labels.Shape().first;
        const auto predictions  = input->ArgMax();
        auto correctPredictions = 0;

        for (auto i : std::ranges::views::iota(0, rows)) {
            if (labels.At(i) == predictions.At(i)) {
                ++correctPredictions;
            }
        }

        return static_cast<float>(correctPredictions) / static_cast<float>(rows);
    }

    export class CategoricalCrossEntropy final
    {
    public:
        template<LayerParams prevLayer>
        [[nodiscard]] constexpr auto Forward (
            const Tensor<float, { prevLayer.Inputs, prevLayer.Neurons }>& input,
            const Tensor<int, { prevLayer.Inputs, 1 }>& trueLabels
        )
        {
            m_trueLabels     = trueLabels;
            auto confidences = Tensor<float, { prevLayer.Inputs, 1 }>{};

            // TODO: Change to std::views::enumerate when available
            for (auto i = 0; const auto& row : input.Data()) {
                confidences.FillAt(i, 0, row.at(trueLabels.At(i)));
                ++i;
            }

            static constexpr auto minFloat = std::numeric_limits<float>::min();

            confidences.Clip(minFloat, 1.0f - minFloat); // Clip to prevent log(0.0f)
            confidences.Log();
            confidences.Negative();

            return confidences.Mean();
        }

        template<LayerParams nextLayer>
        [[nodiscard]] constexpr auto Backward (
            const Tensor<float, { nextLayer.Inputs, nextLayer.Neurons }>& gradients
        )
        {
            auto labels = OneHotEncoding.operator()<Config::BatchSize, Config::NumClasses>(m_trueLabels);
            auto output = labels / gradients;
            output.Negative();
            output.MultiplyEachElementBy( 1.0f / static_cast<float>(nextLayer.Inputs) );
            return output;
        }

    private:
        Tensor<int, { Config::BatchSize, 1 }> m_trueLabels;
    };
}
