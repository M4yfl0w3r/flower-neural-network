#pragma once

#include "Tensor.hpp"

namespace Mayflower
{
    class DenseLayer
    {
    public:
        DenseLayer(unsigned numInputs, unsigned numNeurons);

        auto forward(const Tensor&) -> Tensor;
        auto backward(const Tensor&) -> Tensor;

    private:
        unsigned m_numInputs;
        unsigned m_numNeurons;

        Tensor m_forwardInput;

        Tensor m_weights;
        Tensor m_biases;
    };
}

