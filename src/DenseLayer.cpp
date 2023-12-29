#include "DenseLayer.hpp"

namespace Mayflower
{
    DenseLayer::DenseLayer(unsigned numInputs, unsigned numNeurons)
        : m_numInputs{numInputs}, m_numNeurons{numNeurons}
    {
        m_weights = Tensor(std::pair{m_numInputs, m_numNeurons});
        m_biases = Tensor(std::pair{1, m_numNeurons});

        m_weights.fillRandomValues({0.0f, 1.0f});
        m_biases.fillRandomValues({0.0f, 1.0f});
    }

    auto DenseLayer::forward(const Tensor& input) -> Tensor
    {
        m_forwardInput = input;
        return input * m_weights + m_biases;
    }
}

