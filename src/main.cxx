import std;
import loss;
import dataset;
import config;
import tensor;
import dense_layer;

auto main() -> int
{
    using namespace Mayflower;

    const auto& [data, labels] = Dataset::readFile(Config::irisPath);
    
    static constexpr auto rowTensorParams = TensorParams{ .Rows = 1u, .Cols = 4u };
    static constexpr auto colTensorParams = TensorParams{ .Rows = 1u, .Cols = 1u };

    auto row  = Tensor<float, rowTensorParams>{ { data.at(0) } };
    auto col  = Tensor<std::size_t, colTensorParams>{ {{ labels.at(0).at(0) }} };

    auto loss = Loss::CategoricalCrossEntropy<float, 1u, 3u>();

    static constexpr auto inputLayerParams = LayerParams{ .Inputs = 1u, .Neurons = 4u };
    static constexpr auto stLayerParams    = LayerParams{ .Inputs = 4u, .Neurons = 3u };
    static constexpr auto ndLayerParams    = LayerParams{ .Inputs = 3u, .Neurons = 3u };
    static constexpr auto lossLayerParams  = LayerParams{ .Inputs = 1u, .Neurons = 3u };

    auto st = DenseLayer<stLayerParams>{ Activation::ReLU };
    auto nd = DenseLayer<ndLayerParams>{ Activation::Softmax };

    auto o1 = st.forward<inputLayerParams>(row);

    static constexpr auto firstLayerOutputParams = LayerParams{ .Inputs = 1u, .Neurons = 3u };

    auto o2 = nd.forward<firstLayerOutputParams>(o1);

    const auto lossValue = loss.value(o2, col);
    const auto accValue  = Loss::accuracy(o2, col);
    std::cout << "Loss = " << lossValue << " | Accuracy = " << accValue * 100 << "%\n";

    auto o3 = loss.backward(o2);
    std::cout << "Loss backward output = " << o3 << '\n';

    auto o4 = nd.backward<lossLayerParams>(o3);
    std::cout << "2nd backward output = " << o4 << '\n';

    auto o5 = st.backward<firstLayerOutputParams>(o4);
    std::cout << "1st backward output = " << o5 << '\n';
}
