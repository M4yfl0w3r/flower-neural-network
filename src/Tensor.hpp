#pragma once

#include <iostream>
#include <array>

namespace Mayflower
{
    template <typename Type, std::size_t Rows, std::size_t Cols>
    class Tensor
    {
    public:
        Tensor()
        {

        }

        auto empty(unsigned size) -> void
        {
            auto test = size;
        }

        auto print() const -> void
        {
            std::cout << "[";

            for (const auto& element : m_array)
            {
                 std::cout << element << ", ";
            }

            std::cout << "]";
        }

    private:
        std::array<Type, Rows * Cols> m_array;
    };
}

