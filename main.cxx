import std;
import loss;
import dataset;
import config;
import tensor;
import dense_layer;

auto main() -> int
{
    using namespace Loss;
    namespace rv = std::ranges::views;

    const auto dataset{ Dataset(Config::irisPath) };
    const auto& [data, labels] { dataset.GetRandomBatch() };
    const auto rows{ Tensor<float, { Config::batchSize, Config::dataCols }>(data) };
    const auto cols{ ColumnTensor<Config::batchSize>(labels) };

    auto st { DenseLayer< LayerParams{ .Inputs =   4, .Neurons = 10 },
                          LayerParams{ .Inputs =  10, .Neurons =  4 },
                          LayerParams{ .Inputs =  10, .Neurons = 10 }
                        > {}
    };

    auto nd { DenseLayer< LayerParams{ .Inputs = 10, .Neurons =  3 },  // 3 - num classes
                          LayerParams{ .Inputs = 10, .Neurons = 10 },  // prev layer params
                          LayerParams{ .Inputs = 10, .Neurons =  3 }   // next layer params
                        > {}
    };

    auto loss{ CategoricalCrossEntropy() };

    for (auto i : rv::iota(0, Config::epochs)) {
        auto o1{ st.ForwardReLU(rows) };
        auto o2{ nd.ForwardSoftmax(o1) };

        const auto lossValue{ loss.Forward< LayerParams{ .Inputs = 10, .Neurons = 3 } >(o2, cols) };
        const auto accValue{ Accuracy(&o2, cols) * 100 };

        std::println("Loss = {0} | Accuracy = {1}%", lossValue.At(0, 0), accValue);

        auto o3{ loss.Backward< LayerParams{ .Inputs = 10, .Neurons = 3 } >(o2) };

        auto o4{ nd.BackwardSoftmax(o3) };
        auto o5{ st.BackwardReLU(o4) };
    }
}
