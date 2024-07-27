module;

import std;

export module utilities;

export [[nodiscard]] auto RandomFloat(std::pair<float, float> range)
{
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<float> distribution(range.first, range.second);
    return distribution(generator);
}

export [[nodiscard]] auto RandomInt(std::pair<int, int> range)
{
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<int> distribution(range.first, range.second);
    return distribution(generator);
}
