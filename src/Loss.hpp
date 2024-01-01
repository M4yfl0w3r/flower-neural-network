#pragma once

#include "Tensor.hpp"

namespace Mayflower
{
    enum class Loss 
    {
        CategoricalCrossEntropy
    };

    template <typename Type, unsigned Rows, unsigned Cols>
    class CategoricalCrossEntropy
    {
    public:
        [[nodiscard]] constexpr auto forward(const Tensor<Type, Rows, Cols>& input, 
                                             const Tensor<Type, Rows, 1>& labels);
        
    private:
        Tensor<Type, Rows, 1> targetClasses;

        Tensor<Type, Rows, Cols> m_forwardInput;
        // Tensor m_forwardOutput;
    };

    
    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto CategoricalCrossEntropy<Type, Rows, Cols>::forward(const Tensor<Type, Rows, Cols>& input,
                                                                      const Tensor<Type, Rows, 1>& labels)
    {
        m_forwardInput = input;
    }
}

