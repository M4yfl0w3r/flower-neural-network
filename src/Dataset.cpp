#include "Dataset.hpp"

#include <fstream>
#include <string>

namespace Mayflower
{
    Dataset::Dataset(fs::path path) : m_path{path} { }

    // TODO: Change it to more readable form and not dull
    auto Dataset::read() const -> std::pair<Tensor<float, 151u, 4u>, Tensor<std::size_t, 151u, 1u>>
    {
        auto file = std::ifstream{m_path, std::ios::in | std::ios::binary};
        const auto fileSize = static_cast<std::size_t>(fs::file_size(m_path));

        auto buffer = std::string(fileSize, '\0');
        file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

        std::stringstream test{buffer};
        std::string segment{};
        std::vector<std::string> seglist{};

        while(std::getline(test, segment, '\n'))
            seglist.push_back(segment);

        std::array<std::array<float, 4u>, 151u> data{};
        std::array<std::array<std::size_t, 1u>, 151u> labels{};

        for (auto i = 0u; const auto& row : seglist)
        {
            if (i >= 149u) break;

            std::istringstream iss(row);
            std::vector<float> numbers{};
            std::string token{};
            auto label = 10u;
            
            while (std::getline(iss, token, ',')) 
            {
                if (token == "Iris-setosa")
                    label = 0u;
                else if (token == "Iris-versicolor")
                    label = 1u;
                else if (token == "Iris-virginica")
                    label = 2u;
                else
                    numbers.push_back(std::stof(token));
            }
            
            for (auto j = 0u; j < 4u; ++j)
                data.at(i).at(j) = numbers.at(j);

            labels.at(i).at(0u) = label;
            ++i;
        }
        
        return std::make_pair(data, labels);
    }
}
