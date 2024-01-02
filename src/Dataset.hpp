#pragma once

#include "Tensor.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace Mayflower
{
    enum class Datasets
    {
        Iris
    };
    
    namespace fs = std::filesystem;

    class Dataset
    {

    public:
        Dataset(Datasets dataset, fs::path path, std::size_t numSamples);

    private:
        auto read(std::size_t numSamples) -> void;

        fs::path m_path;
    };

    Dataset::Dataset(Datasets dataset, fs::path path, std::size_t numSamples)
        : m_path{path} 
    {
        if (dataset == Datasets::Iris)
            read(numSamples);
    }

    auto Dataset::read([[maybe_unused]] std::size_t numSamples) -> void
    {
        auto file = std::ifstream{m_path, std::ios::in | std::ios::binary};
        const auto fileSize = static_cast<std::size_t>(fs::file_size(m_path));

        auto buffer = std::string(fileSize, '\0');
        file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

        std::cout << buffer;
    }

}
