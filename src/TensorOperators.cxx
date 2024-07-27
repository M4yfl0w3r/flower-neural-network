export module tensor:operators;

import :implementation;
import std;

export template<typename T, TensorParams params>
[[nodiscard]] constexpr auto operator- (
    const Tensor<T, params>& one,
    const Tensor<T, params>& other
)
{
    auto result = Tensor<T, params>{};

    for (auto i : std::ranges::iota_view(0, params.Rows))
        for (auto j : std::ranges::iota_view(0, params.Cols))
            result.FillAt(i, j, one.At(i, j) - other.At(i, j));

    return result;
}

// Add tensors of same order
export template<typename T, TensorParams params>
[[nodiscard]] constexpr auto operator+ (
    const Tensor<T, params>& one,
    const Tensor<T, params>& other
)
{
    auto result = Tensor<T, params>{};

    for (auto i : std::ranges::iota_view(0, params.Rows))
        for (auto j : std::ranges::iota_view(0, params.Cols))
            result.FillAt(i, j, one.At(i, j) + other.At(i, j));

    return result;
}

// Add biases to each row
export template<typename T, TensorParams a, TensorParams b>
[[nodiscard]] constexpr auto operator+ (
    const Tensor<T, a>& one,
    const Tensor<T, b>& other   // Biases
)
{
    static_assert(a.Cols == b.Cols && b.Rows == 1);

    auto result = Tensor<T, a>{};

    for (auto i : std::ranges::iota_view(0, a.Rows)) {
        for (auto j : std::ranges::iota_view(0, a.Cols)) {
            result.FillAt(i, j, one.At(i, j) + other.At(0, j));
        }
    }

    return result;
}

export template<typename T, TensorParams params>
[[nodiscard]] constexpr auto operator+ (
    const Tensor<T, params>& one,
    T value
)
{
    auto result = Tensor<T, params>{};
    result.ForEachElement( [=](auto& el){ el += value; } );
    return result;
}

export template<typename T, TensorParams a, TensorParams b>
[[nodiscard]] constexpr auto operator* (
    const Tensor<T, a>& one,
    const Tensor<T, b>& other
)
{
    static_assert(a.Cols == b.Rows);

    auto result = Tensor<T, { a.Rows, b.Cols } >{};

    // TODO: Strassen algorithm
    for (auto i : std::ranges::iota_view(0, a.Rows)) {
        for (auto j : std::ranges::iota_view(0, b.Cols)) {
            T sum{};
            for (auto k : std::ranges::iota_view(0, a.Cols)) {
                sum += one.At(i, k) * other.At(k, j);
            }
            result.FillAt(i, j, sum);
        }
    }

    return result;
}

export template<typename T, TensorParams a, TensorParams b>
[[nodiscard]] constexpr auto operator/ (
    const Tensor<T, a>& one,
    const Tensor<T, b>& other
)
{
    if constexpr (b.Rows == 1 && b.Cols == 1) {
        auto result = one;
        result.ForEachElement( [&other](auto& el){ el /= other.At(0, 0); } );
        return result;
    }

    if constexpr (a.Rows == 1 && b.Rows == 1 && a.Cols == b.Cols) {
        // TODO: Better ifs
        auto result = one;
        result.ForEachElement([&other](auto& el){ el /= other.At(0, 0); });
        return result;
    }

    // Divide each row by the corresponding value
    if constexpr (a.Rows == b.Rows && b.Cols == 1) {
        auto result = Tensor<T, a>(one.Data());
        for (auto i = 0; i < a.Rows; ++i) {
            result.DivideRowBy(i, other.At(i));
        }
        return result;
    }
}

export template<typename T, TensorParams params>
[[nodiscard]] constexpr auto operator/ (
    const Tensor<T, params>& one,
    T value
)
{
    auto result = Tensor<T, params>{};
    result.ForEachElement( [=](auto& el){ el /= value; } );
    return result;
}

export template<typename T, TensorParams params>
[[nodiscard]] constexpr auto operator/ (
    const Tensor<T, params>& one,   // Hot one encoding
    const Tensor<T, params>& other
)
{
    auto result = Tensor<T, params>{};

    for (auto i : std::ranges::iota_view(0, params.Rows)) {
        for (auto j : std::ranges::iota_view(0, params.Cols)) {
            result.FillAt(i, j, one.At(i, j) / other.At(i, j));
        }
    }

    return result;
}
