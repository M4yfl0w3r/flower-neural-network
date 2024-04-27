module;

#include <random>

export module utilities;

export namespace Utilities
{
    template <typename Type>
    [[nodiscard]] auto randomNumber(std::pair<Type, Type> range) -> Type 
    {
        std::random_device device;
        std::mt19937 generator(device());
        std::uniform_real_distribution<Type> distribution(range.first, range.second);
        return distribution(generator);
    }
}