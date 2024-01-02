#pragma once

#include "Tensor.hpp"

namespace Mayflower
{
    enum class Activation
    {
        ReLU,
        Softmax
    };

    template <typename Type, std::size_t Inputs, std::size_t Neurons>
    class DenseLayer
    {
    public:
        constexpr DenseLayer(Activation activation);

        constexpr auto printWeights() const;
        constexpr auto printBiases() const;

        [[nodiscard]] constexpr auto forward(const Tensor<Type, 1, Inputs>&);

    private:
        const std::size_t m_numInputs;
        const std::size_t m_numNeurons;
        const Activation m_activation;

        Tensor<Type, 1, Inputs> m_forwardInput;
        Tensor<Type, 1, Neurons> m_forwardOutput;

        Tensor<Type, Inputs, Neurons> m_weights;
        Tensor<Type, 1, Neurons> m_biases;
    };

    template <typename Type, std::size_t Inputs, std::size_t Neurons>
    constexpr DenseLayer<Type, Inputs, Neurons>::DenseLayer(Activation activation)
        : m_numInputs{Inputs}, m_numNeurons{Neurons}, m_activation{activation}
    {
        m_weights = Tensor<Type, Inputs, Neurons>();
        m_weights.fillRandomValues( {0.0f, 1.0f} );

        m_biases= Tensor<Type, 1, Neurons>();
        m_biases.fillRandomValues( {0.0f, 1.0f} );
    }
        
    template <typename Type, std::size_t Inputs, std::size_t Neurons>
    constexpr auto DenseLayer<Type, Inputs, Neurons>::printWeights() const
    {
        m_weights.print();
    }
    
    template <typename Type, std::size_t Inputs, std::size_t Neurons>
    constexpr auto DenseLayer<Type, Inputs, Neurons>::printBiases() const
    {
        m_biases.print();
    }

    template <typename Type, std::size_t Inputs, std::size_t Neurons>
    constexpr auto DenseLayer<Type, Inputs, Neurons>::forward(const Tensor<Type, 1, Inputs>& input)
    {
        m_forwardInput = input;
        m_forwardOutput = (m_forwardInput * m_weights) + m_biases;

        switch (m_activation)
        {
            case Activation::ReLU:
                m_forwardOutput.forEachElement([](auto& el){ el = std::max(Type{}, el); });
                break;
            
            case Activation::Softmax:
                const auto expValues = m_forwardOutput.exp();
                const auto expValuesSum = expValues.sum();
                m_forwardOutput = expValues / expValuesSum;
                break;
        }

        return m_forwardOutput;
    }

}


