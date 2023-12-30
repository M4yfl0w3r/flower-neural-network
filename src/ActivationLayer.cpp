#include "ActivationLayer.hpp"

#include <algorithm>

namespace Mayflower
{
    auto ReLU::forward(const Tensor& input) -> Tensor
    {
        auto output = input;
        output.forEachElement([](auto& el){ el = std::max(0.0f, el); });
        return output;
    }

}
