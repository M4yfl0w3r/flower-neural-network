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

    constexpr auto printWeights() const {
        m_weights.print();
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

    [[nodiscard]] constexpr auto backward(
        const Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons} >& gradients
    )
    {
        const auto weightsT = transpose(m_weights);
        auto output         = gradients * weightsT;

        // const auto inputsT  = transpose(m_forwardInput);
        // auto test = inputsT * gradients;
        // m_weightsGrad = test;

        using enum Activation;

        switch (m_activation) {
            using enum Activation;

            case ReLU:
            {
                output.forEachElement( [](auto& el){ el <= 0.0f ? el = 0.0f : el; } );
                break;
            }

            case Softmax:
            {
                // TODO: Move it to a separate function
                auto jacobian = Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }>{};
                auto [R, C]   = jacobian.shape();

                for (const auto& [output, gradient] : std::views::zip(m_forwardOutput.data(), gradients.data())) 
                {
                    for (auto i = 0uz; i < R; ++i) {
                        for (auto j = 0uz; j < C; ++j) {
                            const auto ith_output = output.at(i);
                            if (i == j)
                                jacobian.fillAt(i, j, ith_output * (1.0f - ith_output));
                            else 
                                jacobian.fillAt(i, j, - ith_output * output.at(j));
                        }
                    }
                }

                break;
            }
        }

        return output;
    }

private:
    const Activation m_activation;

    Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }> m_forwardInput;
    Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons }> m_forwardOutput;
    Tensor<float, TensorParams{ params.Inputs, params.Neurons}>        m_weightsGrad;
    Tensor<float, TensorParams{ params.Inputs, params.Neurons}>        m_weights;
    Tensor<float, TensorParams{ 1uz, params.Neurons }>                 m_biasesGrad;
    Tensor<float, TensorParams{ 1uz, params.Neurons }>                 m_biases;
};
