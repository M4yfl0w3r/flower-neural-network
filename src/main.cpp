#include "Tensor.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"

auto main() -> int
{
    using namespace Mayflower;

    auto layer = DenseLayer(1, 3);
    auto relu = ReLU();

    auto input = Tensor(std::vector{1.0f, 2.0f});

    auto o1 = layer.forward(input);
    auto o2 = relu.forward(o1);

}

