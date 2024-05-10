module;

import std;
import tensor;

export module dense_layer;

export enum class Activation 
{
    ReLU,
    Softmax
};

export struct LayerParams
{
    std::size_t Inputs;
    std::size_t Neurons;
};

export template<LayerParams params, LayerParams prevLayer, LayerParams nextLayer>
class DenseLayer final
{
public:
    // TODO: Move activation somewhere else
    explicit constexpr DenseLayer(Activation activation) 
        : m_activation{activation}
    {
        m_weights = Tensor<float, TensorParams{ params.Inputs, params.Neurons } >();
        m_biases  = Tensor<float, TensorParams{ 1uz, params.Neurons } >();

        m_weights.fillWithRandomValues({ -1.0f, 1.0f });
        m_biases.fillWithRandomValues({ 0.0f, 0.0f });      // TODO: For now, all biases = 0

        m_weights.scaleEachValue(0.01f);
    }

    constexpr auto update(float learningRate) {
        // m_weightsGrad.forEachElement( [=](auto& el){ el *= learningRate; } );
        // m_weights = m_weights + m_weightsGrad;
    }

    [[nodiscard]] constexpr auto forward(
        const Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }>& input
    )
    {
        m_forwardInput = input;
        auto output    = (input * m_weights) + m_biases;
        
        switch (m_activation) {
            using enum Activation;

            case ReLU:
            {
                m_forwardActivationInput = output;
                output.relu();
                break;
            }
            
            case Softmax:
            {
                output.subtractMaxFromEachRow();
                auto expValues    = output.exp();
                auto expValuesSum = expValues.sumEachRow();
                output = expValues / expValuesSum;
                break;
            }
        }

        m_forwardOutput = output;

        return output;
    }

    [[nodiscard]] constexpr auto backwardWithReLU(
        const Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons} >& gradients
    )
    {
        auto result = Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }>{ gradients };

        static constexpr auto lessThanZero = [](auto& el) { return el <= 0.0f; };
        auto lessThanZeroMask = m_forwardActivationInput.where(lessThanZero);

        result.mask(lessThanZeroMask);

        const auto weightsT = transpose(m_weights);
        const auto inputsT  = transpose(m_forwardInput);

        auto output        = result * weightsT;
        auto m_weightsGrad = inputsT * gradients;
        auto m_biasesGrad  = gradients.sumEachColumn();

        return output;
    }

    [[nodiscard]] constexpr auto backwardWithSoftmax(
        const Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons} >& gradients
    )
    {
        auto result   = Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }>{ 0.0f };
        auto jacobian = Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }>{};
        auto [R, C]   = jacobian.shape();

        for (auto index = 0uz; 
            const auto& [output, gradient] : std::views::zip(m_forwardOutput.data(), gradients.data())) 
        {
            for (auto i : std::ranges::iota_view(0uz, R)) {
                for (auto j : std::ranges::iota_view(0uz, C)) {
                    const auto ith_output = output.at(i);
                    if (i == j)
                        jacobian.fillAt(i, j, ith_output * (1.0f - ith_output));
                    else 
                        jacobian.fillAt(i, j, - ith_output * output.at(j));
                }
            }

            auto gradTensor   = Tensor1D(gradient);
            auto gradTensorT  = transpose(gradTensor);
            auto dotProduct   = jacobian * gradTensorT;
            auto untransposed = transpose(dotProduct);

            result.exchangeRow(index++, untransposed.data());
        }

        const auto weightsT = transpose(m_weights);
        auto output = result * weightsT;

        return output;
    }

private:
    const Activation m_activation;

    Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }> m_forwardInput;
    
    Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }> m_forwardOutput;
    Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }> m_forwardActivationInput;;

    Tensor<float, TensorParams{ params.Inputs, params.Neurons}>        m_weightsGrad;
    Tensor<float, TensorParams{ params.Inputs, params.Neurons}>        m_weights;
    Tensor<float, TensorParams{ 1uz, params.Neurons }>                 m_biasesGrad;
    Tensor<float, TensorParams{ 1uz, params.Neurons }>                 m_biases;
};
