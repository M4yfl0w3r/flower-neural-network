#include <iostream>

#include "Tensor.hpp"

auto main() -> int
{
    namespace MFL = Mayflower;

    auto st_tensor = MFL::Tensor<int, 1, 3>();
    auto nd_tensor = MFL::Tensor<int, 3, 1>();

    st_tensor.fill(2);
    nd_tensor.fill(3);

    auto res = dot(st_tensor, nd_tensor);
    res.print();
}

