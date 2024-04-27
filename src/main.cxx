import dataset;
import tensor;
import config;
import dense_layer;

auto main() -> int
{
    using namespace Mayflower;

    const auto& [data, labels] = Dataset::readFile(Config::irisPath);

    auto row = Tensor<float, 1, 4>{ { data.at(0) } };
    auto st  = DenseLayer<float, 4u, 3u>(Activation::ReLU);
    auto nd  = DenseLayer<float, 3u, 3u>(Activation::Softmax);

    auto o1 = st.forward(row);
    auto o2 = nd.forward(o1);

    o2.print();
}
