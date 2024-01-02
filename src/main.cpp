#include "Utils.hpp"
#include "Loss.hpp"
#include "Tensor.hpp"
#include "Dataset.hpp"
#include "DenseLayer.hpp"

auto main() -> int
{
    using namespace Mayflower;

    const auto data = Dataset(Datasets::Iris, 
                              "/home/hope/Repositories/flower-neural-network/assets/iris/iris.data",
                              100u);

    auto st = DenseLayer<float, 2u, 3u>(Activation::ReLU);
    auto nd = DenseLayer<float, 3u, 3u>(Activation::Softmax);
    auto loss = CategoricalCrossEntropy<float, 1u, 3u>();

    auto input = Tensor<float, 1u, 2u>();
    auto labels = Tensor<std::size_t, 1u, 1u>(std::array<std::array<std::size_t, 1u>, 1u>({{0u}}));

    input.fill(2.0f);

    auto o1 = st.forward(input);
    auto o2 = nd.forward(o1);

    const auto lossValue = loss.forward(o2, labels);
    const auto accValue = accuracy<float, 1u, 3u>(o2, labels);

    std::cout << "\nForward pass output\n";
    o2.print();

    std::cout << "\nLoss : " << lossValue << '\n';
    std::cout << "\nAccuracy : " << accValue * 100 << "%\n";
}

