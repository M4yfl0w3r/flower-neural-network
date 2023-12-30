#include "Tensor.hpp"

auto main() -> int
{
    using namespace Mayflower;

    auto tensor = Tensor<float, 2, 3>();
    tensor.fillRandomValues({-1.0f, 1.0f});
    tensor.print();

    // auto layer1 = DenseLayer(2, 3, Activation::ReLU);
    // auto layer2 = DenseLayer(3, 4, Activation::Softmax);

    // auto input = Tensor(std::vector{1.0f, 2.0f}, {1, 2});
    // auto o1 = layer1.forward(input);
    // o1.print("1st DenseLayer output");
    // auto o2 = layer2.forward(o1);
    // o2.print("2nd DenseLayer output");
    
}

