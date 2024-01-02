#pragma once

#include "Tensor.hpp"

#include <filesystem>

namespace Mayflower
{
    namespace fs = std::filesystem;

    class Dataset
    {
    public:
        Dataset(fs::path path);

        auto read() const -> std::pair<Tensor<float, 151u, 4u>, Tensor<std::size_t, 151u, 1u>>;

    private:
        fs::path m_path;
    };
}
