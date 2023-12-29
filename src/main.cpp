#include "Tensor.hpp"
#include "DenseLayer.hpp"

auto main() -> int
{
    using namespace Mayflower;

    auto layer = DenseLayer(1, 3);
    auto input = Tensor(std::vector{1.0f, 2.0f});

    auto output = layer.forward(input);

    output.print();
}

