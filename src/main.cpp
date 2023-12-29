#include <iostream>

#include "Tensor.hpp"

auto main() -> int
{
    namespace MFL = Mayflower;

    auto st_tensor = MFL::Tensor<int, 1, 3>();
    auto nd_tensor = MFL::Tensor<int, 3, 1>();
    auto rd_tensor = MFL::Tensor<int, 1, 3>();

    st_tensor.fill(2);
    nd_tensor.fill(3);
    rd_tensor.fill(5);

    auto res = st_tensor + rd_tensor;
    res.print();
}

