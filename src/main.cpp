#include "Tensor.hpp"
#include "DenseLayer.hpp"

auto main() -> int
{
    using namespace Mayflower;

    auto st = DenseLayer<float, 2, 3>(Activation::ReLU);

    auto input = Tensor<float, 1, 2>();
    input.fill(2.0f);

    st.printBiases();
    st.printWeights();

    // auto o1 = layer1.forward(input);
    // o1.print("1st DenseLayer output");
    // auto o2 = layer2.forward(o1);
    // o2.print("2nd DenseLayer output");
    
}

