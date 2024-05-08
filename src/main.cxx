import std;
import loss;
import dataset;
import config;
import tensor;
import dense_layer;

auto main() -> int
{
    using namespace Mayflower;
    using namespace Loss;

    const auto& [data, labels] = Dataset::readFile(Config::irisPath);

    std::array<std::array<float, 4u>, 3u> batchRows { data.at(0), data.at(54), data.at(119) };
    std::array<std::array<std::size_t, 1u>, 3u> batchCols { labels.at(0), labels.at(54), labels.at(119) };

    auto rows = Tensor<float, TensorParams{ .Rows = 3uz, .Cols = 4uz }>{ batchRows };
    auto cols = Tensor<std::size_t, TensorParams{ .Rows = 3uz, .Cols = 1uz }>{ batchCols };

    auto st   = DenseLayer< 
                            LayerParams{ .Inputs = 4uz, .Neurons = 3uz }, // layer params
                            LayerParams{ .Inputs = 3uz, .Neurons = 4uz }, // prev layer params
                            LayerParams{ .Inputs = 3uz, .Neurons = 3uz }  // next layer params
                          > { Activation::ReLU };
                        
    auto nd   = DenseLayer<
                            LayerParams{ .Inputs = 3uz, .Neurons = 3uz }, // layer params
                            LayerParams{ .Inputs = 3uz, .Neurons = 3uz }, // prev layer params
                            LayerParams{ .Inputs = 3uz, .Neurons = 3uz }
                          > { Activation::Softmax };
    
    auto loss = Loss::CategoricalCrossEntropy();
    
    // for (auto i = 0uz; i < Config::epochs; ++i) {
        auto o1 = st.forward(rows);
        auto o2 = nd.forward(o1);

        const auto lossValue = loss.forward< LayerParams{ .Inputs = 3uz, .Neurons = 3uz } >(o2, cols);
        const auto accValue  = accuracy(&o2, cols);
        std::cout << std::setprecision(4) << "Loss = " << lossValue << " | Accuracy = " << accValue * 100uz << "%\n";

        // auto o3 = loss.backward< LayerParams{ .Inputs = 3uz, .Neurons = 3uz } >(o2);
        // auto o4 = nd.backward(o3);
        // auto o5 = st.backward(o4);

        // st.update(Config::learningRate);
        // nd.update(Config::learningRate);
    // }
}
