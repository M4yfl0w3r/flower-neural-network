#include "Loss.hpp"
#include "Tensor.hpp"
#include "DenseLayer.hpp"

auto main() -> int
{
    using namespace Mayflower;

    auto st = DenseLayer<float, 2u, 3u>(Activation::ReLU);
    auto nd   = DenseLayer<float, 3u, 3u>(Activation::Softmax);
    auto loss = CategoricalCrossEntropy<float, 1u, 3u>();
    
    auto data   = Tensor<float, 1u, 2u>();
    auto labels = Tensor<std::size_t, 1u, 1u>( std::array<std::array<std::size_t, 1u>, 1u>({{0u}}));
    
    data.fill(2.0f);
    
    auto o1 = st.forward(data);
    auto o2 = nd.forward(o1);

    const auto lossValue = loss.forward(o2, labels);
    const auto accValue  = accuracy<float, 1u, 3u>(o2, labels);

    o2.print("\nForward pass output");

    std::cout << "Loss : " << lossValue << '\n';
    std::cout << "Accuracy : " << accValue * 100 << "%\n";
}

