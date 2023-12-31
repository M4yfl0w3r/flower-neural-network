#pragma once

#include "Tensor.hpp"

namespace Mayflower
{
    enum class Activation
    {
        ReLU,
        Softmax
    };

    template <typename Type, unsigned Inputs, unsigned Neurons>
    class DenseLayer
    {
    public:
        constexpr DenseLayer(Activation activation);

        constexpr auto printWeights() const;
        constexpr auto printBiases() const;

        [[nodiscard]] constexpr auto forward(const Tensor<Type, Inputs, Neurons>&) const;

    private:
        const unsigned m_numInputs;
        const unsigned m_numNeurons;
        const Activation m_activation;

        Tensor<Type, Inputs, Neurons> m_weights;
        Tensor<Type, 1, Neurons> m_biases;
    };

    template <typename Type, unsigned Inputs, unsigned Neurons>
    constexpr DenseLayer<Type, Inputs, Neurons>::DenseLayer(Activation activation)
        : m_numInputs{Inputs}, m_numNeurons{Neurons}, m_activation{activation}
    {
        m_weights = Tensor<Type, Inputs, Neurons>();
        m_weights.fillRandomValues( {0.0f, 1.0f} );

        m_biases= Tensor<Type, 1, Neurons>();
        m_biases.fillRandomValues( {0.0f, 1.0f} );
    }
        
    template <typename Type, unsigned Inputs, unsigned Neurons>
    constexpr auto DenseLayer<Type, Inputs, Neurons>::printWeights() const
    {
        m_weights.print();
    }
    
    template <typename Type, unsigned Inputs, unsigned Neurons>
    constexpr auto DenseLayer<Type, Inputs, Neurons>::printBiases() const
    {
        m_biases.print();
    }

    template <typename Type, unsigned Inputs, unsigned Neurons>
    constexpr auto DenseLayer<Type, Inputs, Neurons>::forward(const Tensor<Type, Inputs, Neurons>&) const
    {
        

    }

}


