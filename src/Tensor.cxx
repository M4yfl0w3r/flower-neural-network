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
    
    constexpr Tensor(std::array<std::array<T, params.Cols>, params.Rows> data)
        : m_data{ data }
    {}

    constexpr Tensor(T value) {
        fill(value);
    }

    [[nodiscard]] constexpr auto at(std::size_t x, std::size_t y) const { 
        return m_data.at(x).at(y);
    }
        
    [[nodiscard]] constexpr auto data() const { 
        return m_data;
    }

    // TODO: Rewrite
    [[nodiscard]] constexpr auto mean() const {
        auto sum = T{};
        for (const auto& row : m_data) {
            sum += std::accumulate(std::begin(row), std::end(row), T{});
        }

        return Tensor1D(sum / (params.Rows * params.Cols));
    }
    
    [[nodiscard]] constexpr auto sum() const {
        auto sum = T{};
        for (const auto& row : m_data) {
            sum += std::accumulate(std::begin(row), std::end(row), T{});
        }

        return Tensor1D(sum);
    }
        
    [[nodiscard]] constexpr auto exp() {
        // TODO: Do not use forEachElement
        auto result = Tensor<T, params>(m_data);
        result.forEachElement([](auto& el){ el = std::exp(el); });
        return result;
    }

    [[nodiscard]] constexpr auto shape() const {
        return std::pair { params.Rows, params.Cols };
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
        forEachElement([](auto& el){ el = -el;}); 
    }

    constexpr auto fill(T value) {
        std::ranges::for_each(m_data, [=](auto& row) { std::ranges::fill(row, value); });
    }
    
    constexpr auto relu() {
        forEachElement( [](auto& el){ el = std::max(T{}, el); });
    }

    auto fillWithRandomValues(std::pair<T, T> range) -> void {
        std::ranges::for_each(m_data, [=](auto& row) { 
            std::ranges::generate(row, [&](){ return Utilities::randomNumber(range); } );
        });
    }

    auto print() const {
        for (const auto& row : m_data) {
            for (const auto& el : row) {
                std::cout << el << "  ";
            }
            std::cout << '\n';
        }
    }

    auto printShape() const {
        std::cout << "Shape = " << params.Rows << ", " << params.Cols << '\n';
    }
        
    friend auto& operator<< (std::ostream& stream, const Tensor& tensor) {
        if constexpr (params.Rows == 1uz && params.Cols == 1uz) {
            stream << tensor.at(0uz, 0uz);
        }
        else {
            for (const auto& row : tensor.data()) {
                for (const auto& el : row) {
                    stream << el << ' ';
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

// Add tensors of same order
export template<typename T, TensorParams params>
[[nodiscard]] constexpr auto operator+ (
    const Tensor<T, params>& one, 
    const Tensor<T, params>& other
)
{
    auto result = Tensor<T, params>{};

    for (auto i = 0uz; i < params.Rows; ++i)
        for (auto j = 0uz; j < params.Cols; ++j)
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
    static_assert(a.Rows == b.Cols && b.Rows == 1uz);

    auto result = Tensor<T, a>{};

    for (auto i = 0uz; i < a.Rows; ++i)
        for (auto j = 0uz; j < a.Cols; ++j)
            result.fillAt(i, j, one.at(i, j) + other.at(0uz, i));

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
    for (auto i = 0uz; i < a.Rows; ++i) {
        for (auto j = 0uz; j < b.Cols; ++j) {
            T sum{};
            for (auto k = 0uz; k < a.Cols; ++k)
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
}

// TODO: Create a namespace for it?
export template<typename T, TensorParams params>
[[nodiscard]] inline constexpr auto transpose(
    const Tensor<T, params>& tensor
) 
{
    std::array<std::array<T, params.Rows>, params.Cols> result{};

    for (auto i = 0uz; i < params.Cols; ++i) {
        for (auto j = 0uz; j < params.Rows; ++j) {
            result.at(i).at(j) = tensor.at(j, i);
        }
    }

    return Tensor<T, TensorParams{ params.Cols, params.Rows }>(result);
}
