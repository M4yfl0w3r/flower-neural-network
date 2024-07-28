module;

import std;
import tensor;
import config;

export module batch;

export class Batch
{
    using Data = Tensor<float, { Config::BatchSize, Config::DataCols }>;
    using Labels = Tensor<int, { Config::BatchSize, 1 }>;

public:
    Batch(const Data& data, const Labels& labels)
        : m_data{ data }, m_labels{ labels }
    {

    }

private:
    Data m_data;
    Labels m_labels;
};
