#pragma once

#include "Tensor.hpp"
#include "Config.hpp"

#include <filesystem>

namespace Mayflower
{
    namespace fs = std::filesystem;

    class Dataset final
    {
        using DataWithLabels = std::pair<Tensor<float, Config::dataRows, Config::dataCols>, 
                                         Tensor<std::size_t, Config::dataRows, 1u>>;

    public:
        Dataset(const fs::path& path);

        auto read() const -> DataWithLabels;

    private:
        fs::path m_path;
    };
}
