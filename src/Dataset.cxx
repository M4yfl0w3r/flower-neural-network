module;

import std;
import config;

export module dataset;

namespace fs = std::filesystem;

export namespace Dataset
{
    auto readFile(const fs::path& path) 
    {
        auto data = std::vector<std::string>{};
        auto file = std::ifstream(path);
        auto line = std::string{};

        auto rows   = std::array<std::array<float, Config::dataCols>, Config::dataRows>{};
        auto labels = std::array<std::array<std::size_t, 1uz>, Config::dataRows>{};

        while(std::getline(file, line)) {
            data.push_back(line);
        }

        for (auto i = 0uz; const auto& row : data) 
        {
            if (i > Config::dataRows - 1uz) break;

            auto stream  = std::istringstream(row);
            auto field   = std::string{};
            auto label   = 10uz;
            auto numbers = std::array<float, Config::dataCols>{};

            auto j = 0uz;

            while (std::getline(stream, field, ',')) 
            {
                if (field == "Iris-setosa")          
                    label = 0uz;

                else if (field == "Iris-versicolor") 
                    label = 1uz;

                else if (field == "Iris-virginica")  
                    label = 2uz;

                else
                    numbers.at(j) = std::stof(field);

                ++j;
            }

            rows.at(i) = numbers;
            labels.at(i).at(0uz) = label;

            ++i;
        }

        // TODO: Shuffle data

        return std::make_pair( rows, labels );
    }
}
