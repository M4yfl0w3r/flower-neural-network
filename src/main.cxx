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

    std::array<std::array<float, 4u>, 1u> batchRows { data.at(0) };
    std::array<std::array<std::size_t, 1u>, 1u> batchCols { {{ labels.at(0).at(0) }} };
    
    // std::array<std::array<float, 4u>, 3u> batchRows { data.at(0), data.at(54), data.at(119) };
    // std::array<std::array<std::size_t, 1u>, 3u> batchCols { labels.at(0), labels.at(54), labels.at(119) };

    auto rows = Tensor<float, rowTensorParams>{ batchRows };
    auto cols = Tensor<std::size_t, colTensorParams>{ batchCols };

    // rows.print();
    // cols.print();

    auto loss = Loss::CategoricalCrossEntropy<float>();

    static constexpr auto inputLayerParams = LayerParams{ .Inputs = 1u, .Neurons = 4u };
    static constexpr auto stLayerParams    = LayerParams{ .Inputs = 4u, .Neurons = 3u };
    static constexpr auto ndLayerParams    = LayerParams{ .Inputs = 3u, .Neurons = 3u };
    static constexpr auto lossLayerParams  = LayerParams{ .Inputs = 1u, .Neurons = 3u };

    auto st = DenseLayer<stLayerParams>{ Activation::ReLU };
    auto nd = DenseLayer<ndLayerParams>{ Activation::Softmax };
    
    static constexpr auto stLayerOutputParams = LayerParams{ .Inputs = 1u, .Neurons = 3u };
    static constexpr auto ndLayerOutputParams = LayerParams{ .Inputs = 1u, .Neurons = 3u };

    for (auto i = 0u; i < Config::epochs; ++i) {
        auto o1 = st.forward<inputLayerParams>(rows);
        auto o2 = nd.forward<stLayerOutputParams>(o1);


        const auto lossValue = loss.forward<ndLayerOutputParams>(o2, cols);
        const auto accValue  = Loss::accuracy(o2, cols);
        std::cout << "Loss = " << lossValue << " | Accuracy = " << accValue * 100 << "%\n";


        auto o3 = loss.backward<ndLayerOutputParams>(o2);
        auto o4 = nd.backward<lossLayerParams>(o3);
        auto o5 = st.backward<stLayerOutputParams>(o4);

        st.update(Config::learningRate);
        nd.update(Config::learningRate);
    }
}
