module;

import std;
import config;
import utilities;

export module dataset;

namespace fs = std::filesystem;
namespace cfg = Config;

export class Dataset final 
{
public:
    explicit Dataset(const fs::path& path)
    {
        auto data = std::vector<std::string>{};
        auto file = std::ifstream(path);
        auto line = std::string{};

        while(std::getline(file, line)) {
            data.push_back(line);
        }

        for (auto i = 0uz; const auto& row : data) 
        {
            if (i > cfg::dataRows - 1uz) break;

            auto stream  = std::istringstream(row);
            auto field   = std::string{};
            auto label   = 10uz;
            auto numbers = std::array<float, cfg::dataCols>{};

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

            m_data.at(i) = numbers;
            m_labels.at(i).at(0uz) = label;

            ++i;
        }
    }

    auto getRandomBatch() const
    {
        std::array<std::array<float, cfg::dataCols>, cfg::batchSize> data {};
        std::array<std::array<std::size_t, 1uz>, cfg::batchSize> labels {};

        for (auto i : std::ranges::iota_view(0uz, cfg::batchSize)) 
        {
            auto randomIndex = randomInt({0uz, cfg::dataRows - 1uz});

            data.at(i) = m_data.at(randomIndex);
            labels.at(i) = m_labels.at(randomIndex);
        }

        return std::pair( data, labels );
    }

private:
    // TODO: Change it to Tensors
    std::array<std::array<float, cfg::dataCols>, cfg::dataRows> m_data{};
    std::array<std::array<std::size_t, 1uz>, cfg::dataRows> m_labels{};
};
