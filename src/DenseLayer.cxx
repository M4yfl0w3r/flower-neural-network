module;

import std;
import tensor;
import config;

export module dense_layer;

export struct LayerParams
{
    std::size_t Inputs;
    std::size_t Neurons;
};

export template<LayerParams params, LayerParams prevLayer, LayerParams nextLayer>
class DenseLayer final
{
    using PreviousLayerTensor = Tensor<float, TensorParams{ prevLayer.Inputs,
                                                            prevLayer.Neurons }>;

    using NextLayerTensor = Tensor<float, TensorParams{ nextLayer.Inputs,
                                                        nextLayer.Neurons}>;

public:
    constexpr DenseLayer()
    {
        m_weights = Tensor<float, TensorParams{ params.Inputs, params.Neurons } >();
        m_biases  = Tensor<float, TensorParams{ 1uz, params.Neurons } >();

        m_weights.FillWithRandomValues({ -1.0f, 1.0f });
        m_biases.FillWithRandomValues({ 0.0f, 0.0f });

        m_weights.MultiplyEachElementBy(0.01f);
    }

    [[nodiscard]] constexpr auto ForwardReLU(const PreviousLayerTensor& input)
    {
        m_forwardInput = input;
        auto output    = (input * m_weights) + m_biases;

        m_forwardActivationInput = output;
        output.ReLU();

        m_forwardOutput = output;

        return output;
    }

    [[nodiscard]] constexpr auto ForwardSoftmax(const PreviousLayerTensor& input)
    {
        auto output = (input * m_weights) + m_biases;
        output.SubtractMaxFromEachRow();
        auto expValues    = output.Exp();
        auto expValuesSum = expValues.SumEachRow();
        output = expValues / expValuesSum;
        m_forwardInput = input;
        m_forwardOutput = output;

        return output;
    }

    [[nodiscard]] constexpr auto BackwardReLU(const NextLayerTensor& gradients)
    {
        auto result = NextLayerTensor{ gradients };

        static constexpr auto lessThanZero = [](auto& el) { return el <= 0.0f; };
        auto lessThanZeroMask = m_forwardActivationInput.Where(lessThanZero);

        result.Mask(lessThanZeroMask, 0.0f);

        const auto weightsT = Transpose(m_weights);
        const auto inputsT  = Transpose(m_forwardInput);

        auto output        = result * weightsT;
        auto m_weightsGrad = inputsT * gradients;
        auto m_biasesGrad  = gradients.SumEachColumn();

        m_weightsGrad.MultiplyEachElementBy( -Config::learningRate );
        m_weights = m_weights + m_weightsGrad;

        m_biasesGrad.MultiplyEachElementBy( -Config::learningRate );
        m_biases = m_biases + m_biasesGrad;

        return output;
    }

    [[nodiscard]] constexpr auto BackwardSoftmax(const NextLayerTensor& gradients)
    {
        auto result = NextLayerTensor{ 0.0f };
        auto index  = 0;

        for (const auto& [output, gradient] : std::views::zip(m_forwardOutput.Data(), gradients.Data())) {
            auto jacobian = Tensor<float, TensorParams{ Config::numClasses,
                                                        Config::numClasses }>{};
            auto [R, C]   = jacobian.Shape();

            for (auto i : std::ranges::iota_view(0uz, R)) {
                for (auto j : std::ranges::iota_view(0uz, C)) {
                    auto ith_output = output.at(i);
                    if (i == j)
                        jacobian.FillAt(i, j, ith_output * (1.0f - ith_output));
                    else
                        jacobian.FillAt(i, j, - ith_output * output.at(j));
                }
            }

            auto gradTensor   = Tensor1D(gradient);
            auto gradTensorT  = Transpose(gradTensor);
            auto dotProduct   = jacobian * gradTensorT;
            auto untransposed = Transpose(dotProduct);

            result.ExchangeRow(index++, untransposed.Data());
        }

        const auto weightsT = Transpose(m_weights);
        const auto inputsT  = Transpose(m_forwardInput);

        auto output         = result * weightsT;
        auto m_weightsGrad  = inputsT * gradients;
        auto m_biasesGrad   = gradients.SumEachColumn();

        m_weightsGrad.MultiplyEachElementBy( -Config::learningRate );
        m_weights = m_weights + m_weightsGrad;

        m_biasesGrad.MultiplyEachElementBy( -Config::learningRate );
        m_biases = m_biases + m_biasesGrad;

        return output;
    }

private:
    PreviousLayerTensor m_forwardInput;

    NextLayerTensor m_forwardActivationInput;
    NextLayerTensor m_forwardOutput;

    Tensor<float, TensorParams{ params.Inputs, params.Neurons}> m_weightsGrad;
    Tensor<float, TensorParams{ params.Inputs, params.Neurons}> m_weights;
    Tensor<float, TensorParams{ 1uz, params.Neurons }>          m_biasesGrad;
    Tensor<float, TensorParams{ 1uz, params.Neurons }>          m_biases;
};
