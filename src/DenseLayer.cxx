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

export template<LayerParams params>
class DenseLayer final
{
public:
    explicit constexpr DenseLayer(Activation activation) 
        : m_activation{activation}
    {
        m_weights = Tensor<float, TensorParams{ params.Inputs, params.Neurons } >();
        m_biases  = Tensor<float, TensorParams{ 1, params.Neurons } >();

        m_weights.fillWithRandomValues({ 0.0f, 1.0f });
        m_biases.fillWithRandomValues({ 0.0f, 1.0f });
    }

    constexpr auto update(float learningRate) {
        m_weights.forEachElement([=](auto& el) { el = -el * learningRate; });
        m_biases.forEachElement([=](auto& el) { el = -el * learningRate; });
    }

    constexpr auto printWeights() const {
        m_weights.print();
    }

    template<LayerParams prevLayer>
    [[nodiscard]] constexpr auto forward(
        const Tensor<float, TensorParams{ prevLayer.Inputs, prevLayer.Neurons }>& input
    )
    {
        m_forwardOutput = (input * m_weights) + m_biases;

        switch (m_activation) {
            case Activation::ReLU:
                m_forwardOutput.relu();
                break;
            
            case Activation::Softmax:
                const auto expValues    = m_forwardOutput.exp();
                const auto expValuesSum = expValues.sum();
                m_forwardOutput = expValues / expValuesSum;
                break;
        }

        return m_forwardOutput;
    }

    template<LayerParams nextLayer>
    [[nodiscard]] constexpr auto backward(
        const Tensor<float, TensorParams{ nextLayer.Inputs, nextLayer.Neurons} >& gradients
    ) 
    {
        const auto transposedWeights = transpose(m_weights);
        const auto result = gradients * transposedWeights;
        return result;
    }

private:
    const Activation  m_activation;

    Tensor<float, TensorParams{ 1, params.Neurons }>             m_forwardOutput;
    Tensor<float, TensorParams{ params.Inputs, params.Neurons }> m_weights;
    Tensor<float, TensorParams{ 1, params.Neurons }>             m_biases;
};
