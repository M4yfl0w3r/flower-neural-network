import std;
import loss;
import dataset;
import config;
import tensor;
import dense_layer;

auto main() -> int
{
    using namespace Loss;

    auto dataset = Dataset(Config::irisPath);
    const auto& [data, labels] = dataset.getRandomBatch();

    auto rows = Tensor<float, TensorParams{ .Rows = 10uz, .Cols = 4uz }>{ data };
    auto cols = Tensor<std::size_t, TensorParams{ .Rows = 10uz, .Cols = 1uz }>{ labels };

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

        std::println("Loss = {0} | Accuracy = {1}%", lossValue.at(0, 0), accValue);

        auto o3 = loss.backward< LayerParams{ .Inputs = 10uz, .Neurons = 3uz } >(o2);
        
        auto o4 = nd.backwardSoftmax(o3);
        auto o5 = st.backwardReLU(o4);
    }
}
