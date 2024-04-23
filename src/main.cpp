#include "Loss.hpp"
#include "Tensor.hpp"
#include "Dataset.hpp"
#include "DenseLayer.hpp"

auto main() -> int
{
    using namespace Mayflower;

    const auto dataset = Dataset("/home/hope/Repositories/flower-neural-network/assets/iris/iris.data");
    [[maybe_unused]] const auto& [data, labels] = dataset.read();

    [[maybe_unused]] auto st   = DenseLayer<float, 4u, 3u>(Activation::ReLU);
    [[maybe_unused]] auto nd   = DenseLayer<float, 3u, 3u>(Activation::Softmax);
    [[maybe_unused]] auto loss = CategoricalCrossEntropy<float, 1u, 3u>();
    
    auto firstRow = Tensor<float, 1u, 4u>{ { data.data().at(0) } };
    auto firstCol = Tensor<std::size_t, 1u, 1u>{ { labels.data().at(0).at(0) } };

    auto o1 = st.forward(firstRow);
    auto o2 = nd.forward(o1);
     
    const auto lossValue = loss.forward(o2, firstCol);
    const auto accValue  = accuracy<float, 1u, 3u>(o2, firstCol);
    
    std::cout << "Loss : " << lossValue << '\n';
    std::cout << "Accuracy : " << accValue * 100 << "%\n";
}

