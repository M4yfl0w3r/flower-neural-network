export module tensor:helpers;

import :implementation;
import std;

export template<typename T, TensorParams params>
[[nodiscard]] inline constexpr auto transpose( const Tensor<T, params>& tensor ) 
{
    std::array<std::array<T, params.Rows>, params.Cols> result{};

    for (auto i : std::ranges::iota_view(0uz, params.Cols)) {
        for (auto j : std::ranges::iota_view(0uz, params.Rows)) {
            result.at(i).at(j) = tensor.at(j, i);
        }
    }

    return Tensor<T, TensorParams{ params.Cols, params.Rows }>{ result };
}

export template<typename T, std::size_t Cols>
[[nodiscard]] inline constexpr auto Tensor1D(std::array<T, Cols> data) 
{
    auto data1d = std::array<std::array<T, Cols>, 1uz>{ T{} };
    data1d.at(0uz) = data;
    auto result = Tensor<T, TensorParams{ .Rows = 1uz, .Cols = Cols }>{ data1d };
    return result;
}
