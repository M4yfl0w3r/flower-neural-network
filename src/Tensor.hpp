#pragma once

#include "Utils.hpp"

#include <string_view>
#include <functional>
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std::string_view_literals;

namespace Mayflower
{
    class Tensor 
    {
    public:
        Tensor() = default;

        constexpr explicit Tensor(std::vector<float> data)
            : m_data{data}
        {
            m_shape = {1, std::size(data)};
        }

        constexpr explicit Tensor(std::pair<unsigned, unsigned> shape)
            : m_shape{shape}
        {
            m_data.resize(shape.first * shape.second);
            fill(0.0f);
        }

        auto forEachElement(std::function<void(float&)> func)
        {
            for (auto& el : m_data)
            {
                func(el);
            }
        }
        
        constexpr auto fill(float value) -> void
        {
            std::ranges::fill(m_data, value);
        }

        [[nodiscard]] auto shape() const { return m_shape; }

        [[nodiscard]] auto data() const { return m_data; }

        auto fillRandomValues(std::pair<float, float> range) -> void
        {
            std::ranges::generate(m_data, [&](){ return Utils::randomNumber(range); });
        }
    
        auto print(std::string_view message = ""sv) const -> void
        {
            if (!message.empty()) std::cout << message << '\n';

            for (auto i = 0u; i < m_shape.first; ++i)
            {
                std::cout << "\t";
                for (auto j = 0u; j < m_shape.second; ++j)
                    std::cout << m_data[i * m_shape.second + j] << " ";
                
                std::cout << "\n";
            }
        }

        auto operator+(const Tensor& tensor) const
        {
            std::vector<float> vector(m_shape.first * m_shape.second, 0.0f);
            std::ranges::transform(m_data, tensor.data(), std::begin(vector), std::plus<float>());
            return Tensor(vector);
        }

        auto operator*(const Tensor& tensor) const
        {
            std::vector<float> vector(m_shape.first * m_shape.second, 0.0f);
            std::ranges::transform(m_data, tensor.data(), std::begin(vector), std::multiplies<float>());
            return Tensor(vector);
        }

        auto operator*(float scalar) const
        {
            std::vector<float> vector = m_data;
            std::ranges::for_each(vector, [=](auto& element){ element *= scalar; });
            return Tensor(vector);
        }
        
        auto operator-(float scalar) const
        {
            std::vector<float> vector = m_data;
            std::ranges::for_each(vector, [=](auto& element){ element -= scalar; });
            return Tensor(vector);
        }

    private:
        std::vector<float> m_data;
        std::pair<unsigned, unsigned> m_shape;
    };
}

