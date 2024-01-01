#pragma once

#include "Utils.hpp"

#include <functional>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <array>

namespace Mayflower
{
    template <typename Type, unsigned Rows, unsigned Cols>
    class Tensor
    {
    public:
        constexpr Tensor() = default; 
        constexpr Tensor(std::array<std::array<Type, Cols>, Rows> data);

        Tensor(const Tensor<Type, Rows, Cols>&);
        decltype(auto) operator=(const Tensor<Type, Rows, Cols>&);

        Tensor(Tensor<Type, Rows, Cols>&&) noexcept;
        decltype(auto) operator=(Tensor<Type, Rows, Cols>&&) noexcept;

        constexpr auto forEachElement(std::function<void(Type&)>);

        [[nodiscard]] constexpr auto sum() const -> Tensor<Type, 1, 1>;
        [[nodiscard]] constexpr auto exp() const -> Tensor<Type, Rows, Cols>;
        [[nodiscard]] constexpr auto data() const -> std::array<std::array<Type, Cols>, Rows>;
        [[nodiscard]] constexpr auto at(unsigned x, unsigned y) const -> Type;

        constexpr auto negative() -> void;
        constexpr auto fill(Type value) -> void;
        constexpr auto print() const -> void;
        constexpr auto printShape() const -> void;

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
    Tensor<Type, Rows, Cols>::Tensor(const Tensor<Type, Rows, Cols>& other)
        : m_data{other.data()}
    {
    }
    
    template <typename Type, unsigned Rows, unsigned Cols>
    decltype(auto) Tensor<Type, Rows, Cols>::operator=(const Tensor<Type, Rows, Cols>& other) 
    {
        if (this != &other)
            m_data = other.data();
        
        return *this;
    }

    template <typename Type, unsigned Rows, unsigned Cols>
    Tensor<Type, Rows, Cols>::Tensor(Tensor<Type, Rows, Cols>&& other) noexcept
        : m_data{std::move(other.data())}
    {
    }

    template <typename Type, unsigned Rows, unsigned Cols>
    decltype(auto) Tensor<Type, Rows, Cols>::operator=(Tensor<Type, Rows, Cols>&& other) noexcept
    {
        if (this != &other)
            m_data = std::move(other.data());
        
        return *this;
    }

    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::forEachElement(std::function<void(Type&)> func)
    {
        for (auto& row : m_data) 
        {
            for (auto& element : row)
                func(element);
        }
    }
    
    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::data() const -> std::array<std::array<Type, Cols>, Rows>
    {
        return m_data;
    }
    
    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::at(unsigned x, unsigned y) const -> Type
    {
        return m_data.at(x).at(y);
    }

    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::sum() const -> Tensor<Type, 1, 1>
    {
        auto result = Type{};
        for (const auto& row : m_data)
            result += std::accumulate(std::begin(row), std::end(row), Type{});
        return Tensor<Type, 1, 1>(std::array<std::array<Type, 1>, 1>({result}));
    }

    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::exp() const -> Tensor<Type, Rows, Cols> 
    {
        auto result = Tensor<Type, Rows, Cols>(m_data);
        result.forEachElement([](auto& el){ el = std::exp(el); });
        return result;
    }

    template <typename Type, unsigned Rows, unsigned Cols>
    constexpr auto Tensor<Type, Rows, Cols>::negative() -> void
    {
        this->forEachElement([](auto& el){ el = -el;});
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
    constexpr auto Tensor<Type, Rows, Cols>::printShape() const -> void
    {
        std::cout << "Shape = (" << Rows << ", " << Cols << ")\n";
    }
    
    template <typename Type, unsigned Rows, unsigned Cols>
    auto Tensor<Type, Rows, Cols>::fillRandomValues(std::pair<Type, Type> range) -> void
    {
        std::ranges::for_each(m_data, [=](auto& row){ 
            std::ranges::generate(row, [&](){ return Utils::randomNumber(range); } );
        });
    }

    template <typename Type, unsigned RowsA, unsigned ColsA, unsigned RowsB, unsigned ColsB>
    [[nodiscard]] constexpr auto operator+(const Tensor<Type, RowsA, ColsA>& one,
                                           const Tensor<Type, RowsB, ColsB>& other)
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
    [[nodiscard]] constexpr auto operator*(const Tensor<Type, RowsA, ColsA>& one,
                                           const Tensor<Type, RowsB, ColsB>& other)
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

    template <typename Type, unsigned RowsA, unsigned ColsA, unsigned RowsB, unsigned ColsB>
    [[nodiscard]] constexpr auto operator/(const Tensor<Type, RowsA, ColsA>& one, 
                                           const Tensor<Type, RowsB, ColsB>& other)
    {
        if (RowsB == 1 && ColsB == 1)
        {
            auto result = one;
            result.forEachElement([&other](auto& el){ el /= other.at(0u, 0u); });
            return result;
        }
    }
}

