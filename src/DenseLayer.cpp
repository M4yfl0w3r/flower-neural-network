#include "DenseLayer.hpp"
#include "Utils.hpp"

// namespace Mayflower
// {
//     DenseLayer::DenseLayer(unsigned numInputs, unsigned numNeurons, Activation activation)
//         : m_numInputs{numInputs}, m_numNeurons{numNeurons}, m_activation{activation}
//     {
//         m_weights = Tensor(std::pair{m_numInputs, m_numNeurons});
//         m_biases = Tensor(std::pair{1, m_numNeurons});
//
//         m_weights.fillRandomValues({0.0f, 1.0f});
//         m_biases.fillRandomValues({0.0f, 1.0f});
//
//         m_weights.print("CONSTUCTOR WEIGHTS = ");
//         m_biases.print("CONSTUCTOR BIASES = ");
//     }
//
//     auto DenseLayer::forward(const Tensor& input) -> Tensor
//     {
//         m_forwardInput = input;
//
//         switch (m_activation)
//         {
//         case Activation::ReLU:
//             m_forwardOutput.forEachElement([](auto& el){ el = std::max(0.0f, el); });
//    
//         case Activation::Softmax:
//             break;
//         }
//         
//         return m_forwardOutput;
//     }
// }

