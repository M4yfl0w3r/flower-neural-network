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
    const auto& [data, labels] = dataset.GetRandomBatch();

    auto rows = Tensor<float, {10, 4}>( data );
    auto cols = Tensor<int, {10, 1}>( labels );

    auto st   = DenseLayer<
                            LayerParams{ .Inputs = 4, .Neurons = 10 }, // layer params
                            LayerParams{ .Inputs = 10, .Neurons = 4 }, // prev layer params
                            LayerParams{ .Inputs = 10, .Neurons = 10 }  // next layer params
                          > {};

    auto nd   = DenseLayer<
                            LayerParams{ .Inputs = 10, .Neurons = 3 },  // layer params
                            LayerParams{ .Inputs = 10, .Neurons = 10 }, // prev layer params
                            LayerParams{ .Inputs = 10, .Neurons = 3 }   // next layer params
                          > {};

    auto loss = Loss::CategoricalCrossEntropy();

    for (auto i : std::ranges::iota_view(0, Config::epochs))
    {
        auto o1 = st.ForwardReLU(rows);
        auto o2 = nd.ForwardSoftmax(o1);

        const auto lossValue = loss.Forward< LayerParams{ .Inputs = 10, .Neurons = 3 } >(o2, cols);
        const auto accValue  = Accuracy(&o2, cols) * 100;

        std::println("Loss = {0} | Accuracy = {1}%", lossValue.At(0, 0), accValue);

        auto o3 = loss.Backward< LayerParams{ .Inputs = 10, .Neurons = 3 } >(o2);

        auto o4 = nd.BackwardSoftmax(o3);
        auto o5 = st.BackwardReLU(o4);
    }
}
