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

    std::array<std::array<float, 4u>, 3u> batchRows { data.at(0), data.at(54), data.at(119) };
    std::array<std::array<std::size_t, 1u>, 3u> batchCols { labels.at(0), labels.at(54), labels.at(119) };

    auto rows = Tensor<float, TensorParams{ .Rows = 3uz, .Cols = 4uz }>{ batchRows };
    auto cols = Tensor<std::size_t, TensorParams{ .Rows = 3uz, .Cols = 1uz }>{ batchCols };

    auto st = DenseLayer< LayerParams{ .Inputs = 4uz, .Neurons = 3uz } >{ Activation::ReLU };
    auto nd = DenseLayer< LayerParams{ .Inputs = 3uz, .Neurons = 3uz } >{ Activation::Softmax };
    
    auto loss = Loss::CategoricalCrossEntropy();
    
    auto o1 = st.forward< LayerParams{ .Inputs = 3uz, .Neurons = 4uz } >(rows);
    auto o2 = nd.forward< LayerParams{ .Inputs = 3uz, .Neurons = 3uz } >(o1);
    
    std::cout << o2;
}
