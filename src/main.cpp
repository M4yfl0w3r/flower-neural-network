#include "Tensor.hpp"
#include "DenseLayer.hpp"

auto main() -> int
{
    using namespace Mayflower;

    auto st = DenseLayer<float, 2, 3>(Activation::ReLU);
    auto nd = DenseLayer<float, 3, 3>(Activation::Softmax);

    auto input = Tensor<float, 1, 2>();
    input.fill(2.0f);

    auto o1 = st.forward(input);
    auto o2 = nd.forward(o1);
    
    std::cout << "\nForward pass output\n";
    o2.print();
   
}

