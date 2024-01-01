#include "Tensor.hpp"
#include "DenseLayer.hpp"

auto main() -> int
{
    using namespace Mayflower;

    auto st = DenseLayer<float, 2, 3>(Activation::Softmax);

    auto input = Tensor<float, 1, 2>();
    input.fill(2.0f);

    auto o1 = st.forward(input);
    
    std::cout << "\nDenseLayer output\n";
    o1.print();
   
    // std::cout << "Input = \n";
    // input.print();

    // std::cout << "Weights = \n";
    // st.printWeights();

    // std::cout << "1st DenseLayer output\n";
    // o1.print();
}

