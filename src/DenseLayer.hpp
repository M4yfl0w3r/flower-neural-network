#pragma once

#include "Tensor.hpp"

namespace Mayflower
{

class DenseLayer
{
    DenseLayer(int num_inputs, int num_outputs);

    auto forward(const Tensor& input) -> Tensor;
  
};

}
