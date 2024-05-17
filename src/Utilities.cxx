module;

import std;

export module utilities;

export [[nodiscard]] auto randomFloat(std::pair<float, float> range)
{
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<float> distribution(range.first, range.second);
    return distribution(generator);
}

export [[nodiscard]] auto randomInt(std::pair<std::size_t, std::size_t> range)
{
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<std::size_t> distribution(range.first, range.second);
    return distribution(generator);
}
