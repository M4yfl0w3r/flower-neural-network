module;

import std;
import utilities;

export module tensor;

export struct TensorParams 
{
    std::size_t Rows;
    std::size_t Cols;
};

export template<typename T, TensorParams params>
class Tensor final
{
public:
    constexpr Tensor() = default;
    
    explicit constexpr Tensor(std::array<std::array<T, params.Cols>, params.Rows> data)
        : m_data{ data }
    {}

    explicit constexpr Tensor(T value) {
        fill(value);
    }

    [[nodiscard]] constexpr auto at(std::size_t x, std::size_t y) const { 
        return m_data.at(x).at(y);
    }

    [[nodiscard]] constexpr auto at(std::size_t x) const {
        static_assert(params.Cols == 1uz);
        return m_data.at(x).at(0uz);
    }
        
    [[nodiscard]] constexpr auto data() const { 
        return m_data;
    }

    [[nodiscard]] constexpr auto get() const {
        static_assert(params.Rows == 1uz && params.Cols == 1uz);
        return m_data.at(0uz).at(0uz);
    }

    [[nodiscard]] constexpr auto mean() const {
        auto sum = T{};
        for (const auto& row : m_data) {
            sum += std::accumulate(std::begin(row), std::end(row), T{});
        }
        return Tensor1D(sum / (params.Rows * params.Cols));
    }
    
    [[nodiscard]] constexpr auto sumAllElements() const {
        auto sum = T{};
        for (const auto& row : m_data) {
            sum += std::accumulate(std::begin(row), std::end(row), T{});
        }
        return Tensor1D(sum);
    }

    [[nodiscard]] constexpr auto sumEachRow() const {
        auto result = Tensor<T, TensorParams{ params.Rows, 1uz}>{};

        for (auto i = 0uz; auto& row : m_data) {
            const auto rowSum = std::accumulate(std::begin(row), std::end(row), T{});
            result.fillAt(i, 0uz, rowSum);
            ++i;
        }

        return result;
    }

    [[nodiscard]] constexpr auto sumEachColumn() const {
        auto result = std::array<std::array<T, params.Cols>, 1uz>{ T{} };

        for (const auto& row : m_data) {
            for (auto i = 0u; const auto& el : row) {
                result.at(0uz).at(i) += el;
                ++i;
            }
        }

        return Tensor<T, TensorParams{ 1uz, params.Cols }>(result);
    }
         
    [[nodiscard]] constexpr auto exp() {
        auto result = Tensor<T, params>{m_data};
        result.forEachElement( [](auto& el){ el = std::exp(el); });
        return result;
    }

    [[nodiscard]] constexpr auto shape() const {
        return std::pair { params.Rows, params.Cols };
    }

    [[nodiscard]] constexpr auto argMax() const {
        auto result = Tensor<T, TensorParams{params.Rows, 1uz}>{};

        for (auto i = 0uz; const auto& row : m_data) {
            auto max = std::ranges::max_element(row);
            auto arg = std::ranges::distance(std::begin(row), max);
            result.fillAt(i, 0uz, arg);
            ++i;
        }

        return result;
    }

    [[nodiscard]] constexpr auto where(std::function<bool(T&)> what) {
        // Get indices that satisfy a condition specified by the 'what' function.

        std::array<std::array<std::size_t, params.Cols>, params.Rows> mask{ 0uz };

        for (auto i = 0uz; auto& row : m_data) {
            auto it = std::find_if(std::begin(row), std::end(row), what);

            while (it != std::end(row)) {
                auto j = std::distance(std::begin(row), it);
                mask.at(i).at(j) = 1uz;
                it = std::find_if(std::next(it), std::end(row), what);
            }

            ++i;
        }

        return mask;
    }

    constexpr auto mask(std::array<std::array<std::size_t, params.Cols>, params.Rows> mask) {
        for (auto i : std::ranges::iota_view(0uz, params.Rows))
            for (auto j : std::ranges::iota_view(0uz, params.Cols))
                if (mask.at(i).at(j) == 1)
                    m_data.at(i).at(j) = T{};
    }

    constexpr auto forEachElement(std::function<void(T&)> func) {
        for (auto& row : m_data) 
            std::ranges::for_each(row, func);
    }

    constexpr auto log() { 
        forEachElement([](auto& el){ el = std::log(el); }); 
    }

    constexpr auto clip(T min, T max)  { 
        forEachElement([=](auto& el){ el = std::clamp(el, min, max); }); 
    }
        
    constexpr auto fillAt(std::size_t i, std::size_t j, T value) { 
        m_data.at(i).at(j) = value; 
    }

    constexpr auto negative() {
        forEachElement([](auto& el){ el != T{} ? el = -el : el = T{};}); 
    }

    constexpr auto fill(T value) {
        std::ranges::for_each(m_data, [=](auto& row) { std::ranges::fill(row, value); });
    }
    
    constexpr auto relu() {
        forEachElement( [](auto& el){ el = std::max(T{}, el); });
    }

    constexpr auto subtractMaxFromEachRow() {
        for (auto& row : m_data) {
            const auto max = *std::ranges::max_element(std::begin(row), std::end(row));
            for (auto& el : row) { el -= max; }
        }
    }

    constexpr auto scaleEachValue(T value) {
        forEachElement( [=](auto& el){ el *= value; } );
    }

    auto fillWithRandomValues(std::pair<T, T> range) {
        std::ranges::for_each(m_data, [=](auto& row) { 
            std::ranges::generate(row, [&](){ return Utilities::randomNumber(range); } );
        });
    }

    // TODO: error: use of undeclared identifier 'assert' - no lib?
    constexpr auto divideRowBy(std::size_t row, T value) {
        for (auto& el : m_data.at(row)) {
            el /= value;
        }
    }

    // TODO: Should accept 1D array
    constexpr auto exchangeRow(std::size_t row, std::array<std::array<T, params.Cols>, 1uz> data) {
        m_data.at(row) = data.at(0uz);
    }

    constexpr auto print() const {
        for (const auto& row : m_data) {
            for (const auto& el : row)
                std::cout << el << "  ";
            std::cout << '\n';
        }
    }

    constexpr auto printShape() const {
        std::cout << "Shape = (" << params.Rows << ", " << params.Cols << ")\n";
    }

    constexpr auto operator- () {
        this->negative();
    }

    constexpr auto operator/ (T value) {
        forEachElement( [=](auto& el){ el /= value; } );
    }
        
    friend constexpr auto& operator<< (std::ostream& stream, const Tensor& tensor) {
        if constexpr (params.Rows == 1uz && params.Cols == 1uz) {
            stream << tensor.at(0uz, 0uz);
        }
        else {
            for (const auto& row : tensor.data()) {
                for (const auto& el : row) {
                    stream << std::setw(10) 
                           << std::fixed
                           << std::setprecision(7)
                           << el << ' ';
                }
                stream << '\n';
            }
        }

        return stream;
    }

private:
    constexpr auto Tensor1D(auto value) const {
        return Tensor<T, TensorParams{ 1uz, 1uz }>(std::array<std::array<T, 1uz>, 1uz>({ value }));
    }

    std::array<std::array<T, params.Cols>, params.Rows> m_data;
};

// TODO: Find a better way to write this

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

// TODO: Create a namespace for it?
export template<typename T, TensorParams params>
[[nodiscard]] inline constexpr auto transpose(
    const Tensor<T, params>& tensor
) 
{
    std::array<std::array<T, params.Rows>, params.Cols> result{};

    for (auto i : std::ranges::iota_view(0uz, params.Cols)) {
        for (auto j : std::ranges::iota_view(0uz, params.Rows)) {
            result.at(i).at(j) = tensor.at(j, i);
        }
    }

    return Tensor<T, TensorParams{ params.Cols, params.Rows }>(result);
}

export template<typename T, std::size_t Cols>
[[nodiscard]] inline constexpr auto Tensor1D(
    std::array<T, Cols> data) 
{
    auto data1d = std::array<std::array<T, Cols>, 1uz>{ T{} };
    data1d.at(0uz) = data;
    auto result = Tensor<T, TensorParams{ .Rows = 1uz, .Cols = Cols }>{ data1d };
    return result;
}
