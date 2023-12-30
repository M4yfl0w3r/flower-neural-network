#pragma once

#include "Utils.hpp"

#include <string_view>
#include <functional>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <array>

using namespace std::string_view_literals;

namespace Mayflower
{
    template <typename Type, unsigned Rows, unsigned Cols>
    class Tensor
    {
    public:
        constexpr auto fill(Type value) -> void;
        constexpr auto print() -> void;
        auto fillRandomValues(std::pair<Type, Type> range) -> void;

    private:
        std::array<std::array<Type, Cols>, Rows> m_data;
    };
    
    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::fill(Type value) -> void
    {
        std::ranges::for_each(m_data, [=](auto& row) { std::ranges::fill(row, value); });
    }

    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::print() -> void
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
}

