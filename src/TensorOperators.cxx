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

    for (auto i : std::ranges::iota_view(0uz, params.Rows))
        for (auto j : std::ranges::iota_view(0uz, params.Cols))
            result.fillAt(i, j, one.at(i, j) - other.at(i, j));

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

    for (auto i : std::ranges::iota_view(0uz, params.Rows))
        for (auto j : std::ranges::iota_view(0uz, params.Cols))
            result.fillAt(i, j, one.at(i, j) + other.at(i, j));

    return result;
}

// Add biases to each row
export template<typename T, TensorParams a, TensorParams b>
[[nodiscard]] constexpr auto operator+ (
    const Tensor<T, a>& one,
    const Tensor<T, b>& other   // Biases
)
{
    static_assert(a.Cols == b.Cols && b.Rows == 1uz);

    auto result = Tensor<T, a>{};

    for (auto i : std::ranges::iota_view(0uz, a.Rows))
        for (auto j : std::ranges::iota_view(0uz, a.Cols))
            result.fillAt(i, j, one.at(i, j) + other.at(0uz, j));

    return result;
}

export template<typename T, TensorParams params>
[[nodiscard]] constexpr auto operator+ (
    const Tensor<T, params>& one,
    T value
)
{
    auto result = Tensor<T, params>{};
    result.forEachElement( [=](auto& el){ el += value; } );
    return result;
}

export template<typename T, TensorParams a, TensorParams b>
[[nodiscard]] constexpr auto operator* (
    const Tensor<T, a>& one, 
    const Tensor<T, b>& other
) 
{
    static_assert(a.Cols == b.Rows);

    auto result = Tensor<T, TensorParams{ a.Rows, b.Cols } >{};

    // TODO: Strassen algorithm
    for (auto i : std::ranges::iota_view(0uz, a.Rows)) 
    {
        for (auto j : std::ranges::iota_view(0uz, b.Cols)) 
        {
            T sum{};
            for (auto k : std::ranges::iota_view(0uz, a.Cols))
                sum += one.at(i, k) * other.at(k, j);
            result.fillAt(i, j, sum);
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
    if constexpr (b.Rows == 1uz && b.Cols == 1uz) {
        auto result = one;
        result.forEachElement([&other](auto& el){ el /= other.at(0uz, 0uz); });
        return result;
    }

    if constexpr (a.Rows == 1uz && b.Rows == 1uz && a.Cols == b.Cols) {
        // TODO: Better ifs
        auto result = one;
        result.forEachElement([&other](auto& el){ el /= other.at(0uz, 0uz); });
        return result;
    }

    // Divide each row by the corresponding value
    if constexpr (a.Rows == b.Rows && b.Cols == 1uz) {
        auto result = Tensor<T, a>(one.data());
        for (auto i = 0u; i < a.Rows; ++i)
            result.divideRowBy(i, other.at(i));
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
    result.forEachElement( [=](auto& el){ el /= value; } );
    return result;
}

export template<typename T, TensorParams params>
[[nodiscard]] constexpr auto operator/ (
    const Tensor<T, params>& one,   // Hot one encoding
    const Tensor<T, params>& other
)
{
    auto result = Tensor<T, params>{};

    for (auto i : std::ranges::iota_view(0uz, params.Rows)) {
        for (auto j : std::ranges::iota_view(0uz, params.Cols)) {
            result.fillAt(i, j, one.at(i, j) / other.at(i, j));
        }
    }

    return result;
}
