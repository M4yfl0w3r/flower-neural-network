#pragma once

#include "Utils.hpp"

#include <algorithm>
#include <cassert>
#include <array>

namespace Mayflower
{
    template <typename Type, unsigned Rows, unsigned Cols>
    class Tensor
    {
    public:
        constexpr Tensor() = default; 
        constexpr Tensor(std::array<std::array<Type, Cols>, Rows> data);
        
        [[nodiscard]] constexpr auto at(unsigned x, unsigned y) const -> Type;

        constexpr auto fill(Type value) -> void;
        constexpr auto print() const -> void;

        auto fillRandomValues(std::pair<Type, Type> range) -> void;

    private:
        std::array<std::array<Type, Cols>, Rows> m_data;
    };


    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr Tensor<Type, Rows, Cols>::Tensor(std::array<std::array<Type, Cols>, Rows> data)
        : m_data{data}
    {
    }
    
    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::at(unsigned x, unsigned y) const -> Type
    {
        return m_data.at(x).at(y);
    }
    
    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::fill(Type value) -> void
    {
        std::ranges::for_each(m_data, [=](auto& row) { std::ranges::fill(row, value); });
    }

    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::print() const -> void
    {
        for (const auto& row : m_data)
        {
            for (const auto& el : row)
                std::cout << el << ' ';
            std::cout << '\n';
        }
    }
    
    template <typename Type, unsigned Rows, unsigned Cols>
    auto Tensor<Type, Rows, Cols>::fillRandomValues(std::pair<Type, Type> range) -> void
    {
        std::ranges::for_each(m_data, [=](auto& row) { 
            std::ranges::generate(row, [&](){ return Utils::randomNumber(range); } );
        } );
    }

    template <typename Type, unsigned RowsA, unsigned ColsA, unsigned RowsB, unsigned ColsB>
    [[nodiscard]] constexpr auto operator+(const Tensor<Type, RowsA, ColsA>& one, const Tensor<Type, RowsB, ColsB>& other)
    {
        if constexpr (RowsA == RowsB && ColsA == ColsB)
        {
            const auto rows = RowsA;
            const auto cols = ColsA;
            
            std::array<std::array<Type, cols>, rows> result{};

            for (auto i = 0u; i < rows; ++i)
            {
                for (auto j = 0u; j < cols; ++j)
                    result.at(i).at(j) = one.at(i, j) + other.at(i, j);
            }

            return Tensor<Type, rows, cols>(result);
        }
    }

    template <typename Type, unsigned RowsA, unsigned ColsA, unsigned RowsB, unsigned ColsB>
    [[nodiscard]] constexpr auto operator*(const Tensor<Type, RowsA, ColsA>& one, const Tensor<Type, RowsB, ColsB>& other)
    {
        static_assert(ColsA == RowsB);

        std::array<std::array<Type, ColsB>, RowsA> result{};

        // TODO: Strassen algorithm
        for (auto i = 0u; i < RowsA; ++i)
        {
            for (auto j = 0u; j < ColsB; ++j)
            {
                Type sum{};
                for (auto k = 0u; k < ColsA; ++k)
                    sum += one.at(i, k) * other.at(k, j);
                result.at(i).at(j) = sum;
            }
        }
        
        return Tensor<Type, RowsA, ColsB>(result);
    }
}

