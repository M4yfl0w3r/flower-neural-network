#pragma once

#include <random>
#include <iostream>

namespace Utils
{
    template <typename Type>
    [[nodiscard]] auto randomNumber(std::pair<Type, Type> range) -> Type
    {
        std::random_device device;
        std::mt19937 generator(device());
        std::uniform_real_distribution<Type> distribution(range.first, range.second);
        return distribution(generator);
    }

    inline auto printShape(std::string_view message, std::pair<unsigned, unsigned> shape) 
    {
        std::cout << message.data() << "shape " << "= (" << shape.first << ", " << shape.second << ")\n";
    }
}
