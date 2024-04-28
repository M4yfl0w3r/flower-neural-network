module;

import std;
import tensor;

export module dense_layer;

export enum class Activation 
{
    ReLU,
    Softmax
};

export template<typename Type, std::size_t Inputs, std::size_t Neurons>
class DenseLayer final
{
public:
    explicit constexpr DenseLayer(Activation activation) 
        : m_numInputs{Inputs}, m_numNeurons{Neurons}, m_activation{activation} 
    {
        m_weights = Tensor<Type, Inputs, Neurons>();
        m_biases  = Tensor<Type, 1, Neurons>();

        m_weights.fillWithRandomValues({ 0.0f, 1.0f });
        m_biases.fillWithRandomValues({ 0.0f, 1.0f });
    }

    constexpr auto weights() const { return m_weights; }

    [[nodiscard]] constexpr auto forward(const Tensor<Type, 1, Inputs>& input) {
        m_forwardInput  = input;
        m_forwardOutput = (m_forwardInput * m_weights) + m_biases;

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

    template<std::size_t GradRows, std::size_t GradCols>
    [[nodiscard]] constexpr auto backward(const Tensor<Type, GradRows, GradCols>& gradients) {
        const auto transposedWeights = transpose(m_weights);
        const auto result = gradients * transposedWeights;
        return result;
    }

private:
    const std::size_t m_numInputs;
    const std::size_t m_numNeurons;
    const Activation  m_activation;

    Tensor<Type, 1, Inputs>       m_forwardInput;
    Tensor<Type, 1, Neurons>      m_forwardOutput;
    Tensor<Type, Inputs, Neurons> m_weights;
    Tensor<Type, 1, Neurons>      m_biases;
};
