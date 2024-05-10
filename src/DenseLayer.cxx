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
public:
    explicit constexpr DenseLayer() 
    {
        m_weights = Tensor<float, TensorParams{ params.Inputs, params.Neurons } >();
        m_biases  = Tensor<float, TensorParams{ 1uz, params.Neurons } >();

        m_weights.fillWithRandomValues({ -1.0f, 1.0f });
        m_biases.fillWithRandomValues({ 0.0f, 0.0f });      // TODO: For now, all biases = 0

        m_weights.scaleEachValue(0.01f);
    }

    [[nodiscard]] constexpr auto forwardReLU(
        const Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }>& input
    )
    {
        m_forwardInput = input;
        auto output    = (input * m_weights) + m_biases;
                
        m_forwardActivationInput = output;
        output.relu();
        
        m_forwardOutput = output;

        return output;
    }

    [[nodiscard]] constexpr auto forwardSoftmax(
        const Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }>& input
    )
    {
        m_forwardInput = input;
        auto output    = (input * m_weights) + m_biases;
            
        output.subtractMaxFromEachRow();
        auto expValues    = output.exp();
        auto expValuesSum = expValues.sumEachRow();
        output = expValues / expValuesSum;

        return output;
    }

    // TODO: Combine backward functions
    [[nodiscard]] constexpr auto backwardReLU(
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

        m_weightsGrad.forEachElement( [=](auto& el){ el *= -Mayflower::Config::learningRate; } );
        m_weights = m_weights + m_weightsGrad;

        m_biasesGrad.forEachElement( [=](auto& el){ el *= -Mayflower::Config::learningRate; });
        m_biases = m_biases + m_biasesGrad;

        return output;
    }

    [[nodiscard]] constexpr auto backwardSoftmax(
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
        const auto inputsT  = transpose(m_forwardInput);

        auto output         = result * weightsT;
        auto m_weightsGrad  = inputsT * gradients;
        auto m_biasesGrad   = gradients.sumEachColumn();
        
        m_weightsGrad.forEachElement( [=](auto& el){ el *= -Mayflower::Config::learningRate; } );
        m_weights = m_weights + m_weightsGrad;

        m_biasesGrad.forEachElement( [=](auto& el){ el *= -Mayflower::Config::learningRate; });
        m_biases = m_biases + m_biasesGrad;

        return output;
    }

private:
    Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }> m_forwardInput;

    Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }> m_forwardActivationInput;;
    Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }> m_forwardOutput;

    Tensor<float, TensorParams{ params.Inputs, params.Neurons}>        m_weightsGrad;
    Tensor<float, TensorParams{ params.Inputs, params.Neurons}>        m_weights;
    Tensor<float, TensorParams{ 1uz, params.Neurons }>                 m_biasesGrad;
    Tensor<float, TensorParams{ 1uz, params.Neurons }>                 m_biases;
};
