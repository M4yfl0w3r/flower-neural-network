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

        for (auto i = 0; const auto& row : data) {
            if (i > cfg::DataRows - 1) {
                break;
            }

            auto stream  = std::istringstream(row);
            auto field   = std::string{};
            auto label   = 10;
            auto numbers = std::array<float, cfg::DataCols>{};

            auto j = 0;

            while (std::getline(stream, field, ',')) {
                if (field == "Iris-setosa") {
                    label = 0;
                }

                else if (field == "Iris-versicolor") {
                    label = 1;
                }

                else if (field == "Iris-virginica") {
                    label = 2;
                }

                else {
                    numbers.at(j) = std::stof(field);
                }

                ++j;
            }

            m_data.at(i) = numbers;
            m_labels.at(i).at(0) = label;

            ++i;
        }
    }

    auto GetRandomBatch() const
    {
        std::array<std::array<float, cfg::DataCols>, cfg::BatchSize> data {};
        std::array<std::array<int, 1>, cfg::BatchSize> labels {};

        for (auto i : std::ranges::iota_view(0, cfg::BatchSize)) {
            auto randomIndex = RandomInt( {0, cfg::DataRows - 1} );
            data.at(i) = m_data.at(randomIndex);
            labels.at(i) = m_labels.at(randomIndex);
        }

        return std::pair( data, labels );
    }

    // TODO: Get not random batch - balanced - each class the same

private:
    // TODO: Change it to Tensors
    std::array<std::array<float, cfg::DataCols>, cfg::DataRows> m_data{};
    std::array<std::array<int, 1>, cfg::DataRows> m_labels{};
};
