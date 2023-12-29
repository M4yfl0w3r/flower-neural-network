#include <iostream>

#include "Tensor.hpp"

auto main() -> int
{
    namespace MFL = Mayflower;

    auto tensor = MFL::Tensor<float, 3, 4>();
    tensor.print();
}

