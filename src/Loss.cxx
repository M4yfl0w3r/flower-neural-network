module;

import std;
import tensor;
import config;
import dense_layer;

export module loss;

namespace Loss 
{
    using namespace Operators;
    
    static constexpr auto oneHotEncoding = []<std::size_t R, std::size_t C>(const auto& labels) 
    {
        auto result = Tensor<float, TensorParams{ R, C }>{ 0.0f };

        for (auto i : std::ranges::iota_view(0uz, R))
            result.fillAt(i, labels.at(i, 0uz), 1.0f);

        return result;
    };

    export [[nodiscard]] constexpr auto accuracy(const auto* input, const auto& labels)
    {
        const auto rows         = labels.shape().first;
        const auto predictions  = input->argMax();
        auto correctPredictions = 0uz;

        for (auto i : std::ranges::iota_view(0uz, rows))
            if (labels.at(i) == predictions.at(i))
                ++correctPredictions;

        return static_cast<float>(correctPredictions) / static_cast<float>(rows);
    }

    export class CategoricalCrossEntropy final
    {
    public:
        template<LayerParams prevLayer>
        [[nodiscard]] constexpr auto forward(
            const Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }>& input, 
            const Tensor<std::size_t, TensorParams{ prevLayer.Inputs, 1uz }>& trueLabels
        )
        {
            m_trueLabels     = trueLabels;
            auto confidences = Tensor<float, TensorParams{ prevLayer.Inputs, 1uz }>{};

            // TODO: Change to std::views::enumerate when available
            for (auto i = 0uz; auto& row : input.data()) 
            {
                confidences.fillAt(i, 0uz, row.at(trueLabels.at(i)));
                ++i;
            }
 
            static constexpr auto minFloat = std::numeric_limits<float>::min();

            confidences.clip(minFloat, 1.0f - minFloat); // Clip to prevent log(0.0f)
            confidences.log();
            confidences.negative();

            return confidences.mean();
        }

        template<LayerParams nextLayer>
        [[nodiscard]] constexpr auto backward(
            const Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }>& gradients
        ) 
        {
            auto labels = oneHotEncoding.operator()<Config::batchSize, 
                                                    Config::numClasses>(m_trueLabels);
            auto output = labels / gradients;
            output.negative();
            output.scaleEachValue( 1.0f / static_cast<float>(nextLayer.Inputs) );
            return output;
        }

    private:
        Tensor<std::size_t, TensorParams{ Config::batchSize, 1uz }> m_trueLabels;
    };
}

