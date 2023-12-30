#pragma once

#include "Tensor.hpp"

namespace Mayflower
{
    class ReLU 
    {
    public:
        auto forward(const Tensor&) -> Tensor;
        auto backward(const Tensor&) -> Tensor;

    private:
        Tensor m_forwardInput;
    };
}
