#pragma once

#include "Utils.hpp"

#include <string_view>
#include <functional>
#include <algorithm>
#include <iostream>
#include <utility>
#include <numeric>
#include <cmath>
#include <array>

namespace Mayflower
{
    template<typename Type, std::size_t Rows, std::size_t Cols>
    class Tensor final
    {
    public:
        constexpr Tensor() = default;

        constexpr Tensor(std::array<std::array<Type, Cols>, Rows> data) 
            : m_data{ data }
        { 

        }

        [[nodiscard]] constexpr auto at(std::size_t x, std::size_t y) const 
        { 
            return m_data.at(x).at(y); 
        }

        [[nodiscard]] constexpr auto data() const 
        { 
            return m_data; 
        }

        constexpr auto forEachElement(std::function<void(Type&)> func) 
        {
            for (auto& row : m_data) 
                std::ranges::for_each(row, func);
        }

        [[nodiscard]] constexpr auto mean() const
        {
            auto sum = Type{};

            for (const auto& row : m_data) {
                // TODO: std::ranges::accumulate(row, Type{}); when available
                sum += std::accumulate(std::begin(row), std::end(row), Type{});
            }

            return Tensor1D(sum / (Rows * Cols));
        }
        
        [[nodiscard]] constexpr auto sum() const
        {
            auto sum = Type{};

            for (const auto& row : m_data) {
                // TODO: std::ranges::accumulate(row, Type{}); when available
                sum += std::accumulate(std::begin(row), std::end(row), Type{});
            }

            return Tensor1D(sum);
        }

        [[nodiscard]] constexpr auto exp()
        {
            // TODO: Do not use forEachElement
            auto result = Tensor<Type, Rows, Cols>(m_data);
            result.forEachElement([](auto& el){ el = std::exp(el); });
            return result;
        }

        constexpr auto log() 
        { 
            forEachElement([=](auto& el){ el = std::log(el); }); 
        }
        
        constexpr auto clip(Type min, Type max) 
        { 
            forEachElement([=](auto& el){ el = std::clamp(el, min, max); }); 
        }

        constexpr auto fillAt(std::size_t i, std::size_t j, Type value) 
        { 
            m_data.at(i).at(j) = value; 
        }

        constexpr auto negative() 
        { 
            forEachElement([](auto& el){ el = -el;}); 
        }

        constexpr auto fill(Type value) 
        {
            std::ranges::for_each(m_data, [=](auto& row) { std::ranges::fill(row, value); });
        }

        auto fillRandomValues(std::pair<Type, Type> range) -> void
        {
            std::ranges::for_each(m_data, [=](auto& row) { 
                std::ranges::generate(row, [&](){ return Utils::randomNumber(range); } );
            });
        }

    private:
        constexpr auto Tensor1D(auto value) const {
            return Tensor<Type, 1u, 1u>(std::array<std::array<Type, 1u>, 1u>({ value }));
        }

        std::array<std::array<Type, Cols>, Rows> m_data;
    };
    
    constexpr auto print(const auto& tensor) 
    {
        for (const auto& row : tensor.data()) {
            for (const auto& el : row)
                std::cout << el << ' ';
            std::cout << '\n';
        }
    }

    template <typename T, std::size_t Rows, std::size_t Cols>
    [[nodiscard]] constexpr auto operator+(const Tensor<T, Rows, Cols>& one,
                                           const Tensor<T, Rows, Cols>& other)
    {
        auto result = Tensor<T, Rows, Cols>{};

        for (auto i = 0u; i < Rows; ++i)
            for (auto j = 0u; j < Cols; ++j)
                result.fillAt(i, j, one.at(i, j) + other.at(i, j));

        return result;
    }

    template <typename T, std::size_t RowsA, std::size_t ColsA, std::size_t RowsB, std::size_t ColsB>
    [[nodiscard]] constexpr auto operator*(const Tensor<T, RowsA, ColsA>& one,
                                           const Tensor<T, RowsB, ColsB>& other)
    {
        static_assert(ColsA == RowsB);

        auto result = Tensor<T, RowsA, ColsB>{};

        // TODO: Strassen algorithm
        for (auto i = 0u; i < RowsA; ++i)
        {
            for (auto j = 0u; j < ColsB; ++j)
            {
                T sum{};
                for (auto k = 0u; k < ColsA; ++k)
                    sum += one.at(i, k) * other.at(k, j);
                result.fillAt(i, j, sum);
            }
        }
        
        return result;
    }

    template <typename T, std::size_t RowsA, std::size_t ColsA, std::size_t RowsB, std::size_t ColsB>
    [[nodiscard]] constexpr auto operator/(const Tensor<T, RowsA, ColsA>& one, 
                                           const Tensor<T, RowsB, ColsB>& other)
    {
        if constexpr (RowsB == 1 && ColsB == 1)
        {
            auto result = one;
            result.forEachElement([&other](auto& el){ el /= other.at(0u, 0u); });
            return result;
        }
    }
}
