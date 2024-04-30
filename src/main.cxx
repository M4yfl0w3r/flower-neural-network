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
    
    static constexpr auto rowTensorParams = TensorParams{ .Rows = 1uz, .Cols = 4uz };
    static constexpr auto colTensorParams = TensorParams{ .Rows = 1uz, .Cols = 1uz };

    std::array<std::array<float, 4u>, 1u> batchRows { data.at(0) };
    std::array<std::array<std::size_t, 1u>, 1u> batchCols { {{ labels.at(0).at(0) }} };
    
    // std::array<std::array<float, 4u>, 3u> batchRows { data.at(0), data.at(54), data.at(119) };
    // std::array<std::array<std::size_t, 1u>, 3u> batchCols { labels.at(0), labels.at(54), labels.at(119) };

    auto rows = Tensor<float, rowTensorParams>{ batchRows };
    auto cols = Tensor<std::size_t, colTensorParams>{ batchCols };

    // rows.print();
    // cols.print();

    auto loss = Loss::CategoricalCrossEntropy();

    auto st = DenseLayer< LayerParams{ .Inputs = 4uz, .Neurons = 3uz } >{ Activation::ReLU };
    auto nd = DenseLayer< LayerParams{ .Inputs = 3uz, .Neurons = 3uz } >{ Activation::Softmax };
    
    for (auto i = 0uz; i < Config::epochs; ++i) {
        
        auto o1 = st.forward< LayerParams{ .Inputs = 1uz, .Neurons = 4uz } >(rows);
        auto o2 = nd.forward< LayerParams{ .Inputs = 1uz, .Neurons = 3uz } >(o1);


        const auto lossValue = loss.forward< LayerParams{ .Inputs = 1uz, .Neurons = 3uz } >(o2, cols);
        const auto accValue  = Loss::accuracy(o2, cols);
        std::cout << "Loss = " << lossValue << " | Accuracy = " << accValue * 100 << "%\n";


        auto o3 = loss.backward< LayerParams{ .Inputs = 1uz, .Neurons = 3uz } >(o2);
        auto o4 = nd.backward<   LayerParams{ .Inputs = 1uz, .Neurons = 3uz } >(o3);
        auto o5 = st.backward<   LayerParams{ .Inputs = 1uz, .Neurons = 3uz } >(o4);

        st.update(Config::learningRate);
        nd.update(Config::learningRate);
    }
}
