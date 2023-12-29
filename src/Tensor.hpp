#pragma once

#include "Utils.hpp"

#include <string_view>
#include <algorithm>
#include <iostream>
#include <array>

using namespace std::string_view_literals;

namespace Mayflower
{
    template <typename Type, std::size_t Rows, std::size_t Cols>
    class Tensor
    {
    public:
        constexpr Tensor()
        {
            fill(Type{});
        }

        explicit constexpr Tensor(const std::array<Type, Rows * Cols>& array)
            : m_array{array}
        {
        }

        constexpr auto fill(Type value) -> void
        {
            std::ranges::fill(m_array, value);
        }

        constexpr auto print(std::string_view message = ""sv) const -> void
        {
            if (!message.empty()) std::cout << message << "\n";

            for (auto i = 0u; i < Rows; ++i)
            {
                std::cout << "\t";
                for (auto j = 0u; j < Cols; ++j)
                    std::cout << m_array[i * Cols + j] << " ";
                
                std::cout << "\n";
            }
        }

        constexpr auto data() const -> std::array<Type, Rows * Cols>
        {
            return m_array;
        }

        auto fillRandomValues(std::pair<Type, Type> range) -> void
        {
            std::ranges::generate(m_array, [&](){ return Utils::randomNumber(range); });
        }

    private:
        std::array<Type, Rows * Cols> m_array{};
    };

    template <typename Type, 
              std::size_t RowsA, std::size_t ColsA,
              std::size_t RowsB, std::size_t ColsB>
    auto dot(const Tensor<Type, RowsA, ColsA>& a, 
             const Tensor<Type, RowsB, ColsB>& b)
    {
        std::array<Type, RowsA * ColsA> array;
        std::ranges::transform(a.data(), b.data(), array.begin(), std::multiplies<Type>());
        return Tensor<Type, RowsA, ColsA>(array);
    }
}

