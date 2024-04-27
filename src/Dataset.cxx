module;

import std;
import config;

export module dataset;

namespace fs = std::filesystem;

export namespace Dataset {
    auto readFile(const fs::path& path) {
        using namespace Mayflower;

        auto file = std::ifstream{path, std::ios::in | std::ios::binary};
        const auto fileSize = static_cast<std::size_t>(fs::file_size(path));

        auto buffer = std::string(fileSize, '\0');
        file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

        std::stringstream test{buffer};
        std::string segment{};
        std::vector<std::string> seglist{};

        while(std::getline(test, segment, '\n'))
            seglist.push_back(segment);

        std::array<std::array<float, Config::dataCols>, Config::dataRows> data{};
        std::array<std::array<std::size_t, 1u>, Config::dataRows> labels{};

        // For some reason enumerate does not work with import std.
        for (auto i = 0u; const auto& row : seglist)
        {
            if (i > Config::dataRows - 1) break;

            std::istringstream iss(row);
            std::vector<float> numbers{};
            std::string token{};
            auto label = Config::labelPos;
            
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
            
            for (auto j = 0u; j < Config::dataCols; ++j)
                data.at(i).at(j) = numbers.at(j);

            labels.at(i).at(0u) = label;
            ++i;
        }
        
        return std::make_pair(data, labels);
    }
}
