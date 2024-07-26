module;

import std;
import tensor;
import config;
import dense_layer;

export module loss;

namespace Loss
{
    static constexpr auto OneHotEncoding = []<std::size_t R, std::size_t C> [[nodiscard]] (const auto& labels)
    {
        auto result = Tensor<float, TensorParams{ R, C }>{ 0.0f };

        for (auto i : std::ranges::views::iota(0uz, R)) {
            result.FillAt(i, labels.At(i, 0uz), 1.0f);
        }

        return result;
    };

    export [[nodiscard]] constexpr auto Accuracy(const auto* input, const auto& labels)
    {
        const auto rows         = labels.Shape().first;
        const auto predictions  = input->ArgMax();
        auto correctPredictions = 0;

        for (auto i : std::ranges::views::iota(0uz, rows)) {
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
            const Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }>& input,
            const Tensor<std::size_t, TensorParams{ prevLayer.Inputs, 1uz }>& trueLabels
        )
        {
            m_trueLabels     = trueLabels;
            auto confidences = Tensor<float, TensorParams{ prevLayer.Inputs, 1uz }>{};

            // TODO: Change to std::views::enumerate when available
            for (auto i = 0uz; auto& row : input.Data())
            {
                confidences.FillAt(i, 0uz, row.at(trueLabels.At(i)));
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
            const Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }>& gradients
        )
        {
            auto labels = OneHotEncoding.operator()<Config::batchSize, Config::numClasses>(m_trueLabels);
            auto output = labels / gradients;
            output.Negative();
            output.MultiplyEachElementBy( 1.0f / static_cast<float>(nextLayer.Inputs) );
            return output;
        }

    private:
        Tensor<std::size_t, TensorParams{ Config::batchSize, 1uz }> m_trueLabels;
    };
}
