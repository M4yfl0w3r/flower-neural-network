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
    
    auto row  = Tensor<float, 1u, 4u>{ { data.at(0) } };
    auto col  = Tensor<std::size_t, 1u, 1u>{ {{ labels.at(0).at(0) }} };
    auto loss = Loss::CategoricalCrossEntropy<float, 1u, 3u>();

    auto st   = DenseLayer<float, 4u, 3u>(Activation::ReLU);
    auto nd   = DenseLayer<float, 3u, 3u>(Activation::Softmax);


    auto o1 = st.forward(row);
    std::cout << "1st forward output = " << o1 << '\n';


    auto o2 = nd.forward(o1);
    std::cout << "2nd forward output = " << o2 << '\n';


    const auto lossValue = loss.value(o2, col);
    const auto accValue  = Loss::accuracy(o2, col);
    std::cout << "Loss = " << lossValue << " | Accuracy = " << accValue * 100 << "%\n";


    auto o3 = loss.backward(o2);
    std::cout << "Loss backward output = " << o3 << '\n';


    auto o4 = nd.backward<o3.rows(), o3.cols()>(o3);
    std::cout << "2nd backward output = " << o4 << '\n';


    auto o5 = st.backward<o4.rows(), o4.cols()>(o4);
    std::cout << "1st backward output = " << o5 << '\n';
}
