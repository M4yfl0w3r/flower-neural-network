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

    std::array<std::array<float, 4u>, 10u> batchRows { data.at(0), 
                                                       data.at(1), 
                                                       data.at(2), 
                                                       data.at(3), 
                                                       data.at(4), 
                                                       data.at(5), 
                                                       data.at(6), 
                                                       data.at(7), 
                                                       data.at(54), 
                                                       data.at(119) };

    std::array<std::array<std::size_t, 1u>, 10u> batchCols { labels.at(0), 
                                                             labels.at(1), 
                                                             labels.at(2), 
                                                             labels.at(3), 
                                                             labels.at(4), 
                                                             labels.at(5), 
                                                             labels.at(6), 
                                                             labels.at(7), 
                                                             labels.at(54), 
                                                             labels.at(119) };

    auto rows = Tensor<float, TensorParams{ .Rows = 10uz, .Cols = 4uz }>{ batchRows };
    auto cols = Tensor<std::size_t, TensorParams{ .Rows = 10uz, .Cols = 1uz }>{ batchCols };

    auto st   = DenseLayer< 
                            LayerParams{ .Inputs = 4uz, .Neurons = 10uz }, // layer params
                            LayerParams{ .Inputs = 10uz, .Neurons = 4uz }, // prev layer params
                            LayerParams{ .Inputs = 10uz, .Neurons = 10uz}  // next layer params
                          > {};
                        
    auto nd   = DenseLayer<
                            LayerParams{ .Inputs = 10uz, .Neurons = 3uz },  // layer params
                            LayerParams{ .Inputs = 10uz, .Neurons = 10uz }, // prev layer params
                            LayerParams{ .Inputs = 10uz, .Neurons = 3uz }   // next layer params
                          > {};
    
    auto loss = Loss::CategoricalCrossEntropy();
    
    for (auto i : std::ranges::iota_view(0uz, Config::epochs)) 
    {
        auto o1 = st.forwardReLU(rows);
        auto o2 = nd.forwardSoftmax(o1);
        
        const auto lossValue = loss.forward< LayerParams{ .Inputs = 10uz, .Neurons = 3uz } >(o2, cols);
        const auto accValue  = accuracy(&o2, cols) * 100uz;

        std::cout << std::setprecision(4) << "Loss = " << lossValue << " | " << 
                                             "Accuracy = " << accValue << "%\n";

        auto o3 = loss.backward< LayerParams{ .Inputs = 10uz, .Neurons = 3uz } >(o2);
        auto o4 = nd.backwardSoftmax(o3);
        auto o5 = st.backwardReLU(o4);
    }
}
