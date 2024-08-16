export module tensor:helpers;

import :implementation;
import std;

export template<typename T, TensorParams params>
[[nodiscard]] inline constexpr auto Transpose(const Tensor<T, params>& tensor)
{
    std::array<std::array<T, params.Rows>, params.Cols> result{};

    for (auto i : std::ranges::iota_view(0, params.Cols)) {
        for (auto j : std::ranges::iota_view(0, params.Rows)) {
            result.at(i).at(j) = tensor.At(j, i);
        }
    }

    return Tensor<T, { params.Cols, params.Rows }>{ result };
}

export template<typename T, int Cols>
[[nodiscard]] inline constexpr auto Tensor1D(std::array<T, Cols> data)
{
    auto data1d = std::array<std::array<T, Cols>, 1>{ T{} };
    data1d.at(0) = data;
    auto result = Tensor<T, { 1, Cols }>{ data1d };
    return result;
}

export template <int Rows>
[[nodiscard]] inline constexpr auto ColumnTensor(auto data)
{
    return Tensor<int, { Rows, 1 }>{ data };
}

export template <int Cols>
[[nodiscard]] inline constexpr auto RowTensor(auto data)
{
    return Tensor<float, { 1, Cols }>{ data };
}
