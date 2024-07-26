export module tensor:implementation;

import std;
import utilities;

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
        Fill(value);
    }

    [[nodiscard]] constexpr auto At(std::size_t x, std::size_t y) const {
        return m_data.at(x).at(y);
    }

    [[nodiscard]] constexpr auto At(std::size_t x) const {
        static_assert(params.Cols == 1uz);
        return m_data.at(x).at(0uz);
    }

    [[nodiscard]] constexpr auto Data() const {
        return m_data;
    }

    [[nodiscard]] constexpr auto Get() const {
        static_assert(params.Rows == 1uz && params.Cols == 1uz);
        return m_data.at(0uz).at(0uz);
    }

    [[nodiscard]] constexpr auto Mean() const {
        auto sum = T{};
        for (const auto& row : m_data) {
            sum += std::accumulate(std::begin(row), std::end(row), T{});
        }
        return Tensor1D(sum / (params.Rows * params.Cols));
    }

    [[nodiscard]] constexpr auto SumAllElements() const {
        auto sum = T{};
        for (const auto& row : m_data) {
            sum += std::accumulate(std::begin(row), std::end(row), T{});
        }
        return Tensor1D(sum);
    }

    [[nodiscard]] constexpr auto SumEachRow() const {
        auto result = Tensor<T, TensorParams{ params.Rows, 1uz}>{};

        for (auto i = 0uz; auto& row : m_data) {
            const auto rowSum = std::accumulate(std::begin(row), std::end(row), T{});
            result.FillAt(i, 0uz, rowSum);
            ++i;
        }

        return result;
    }

    [[nodiscard]] constexpr auto SumEachColumn() const {
        auto result = std::array<std::array<T, params.Cols>, 1uz>{ T{} };

        for (const auto& row : m_data) {
            for (auto i = 0u; const auto& el : row) {
                result.at(0uz).at(i) += el;
                ++i;
            }
        }

        return Tensor<T, TensorParams{ 1uz, params.Cols }>(result);
    }

    [[nodiscard]] constexpr auto Exp() {
        auto result = Tensor<T, params>{m_data};
        result.ForEachElement( [](auto& el){ el = std::exp(el); });
        return result;
    }

    [[nodiscard]] constexpr auto Shape() const {
        return std::pair { params.Rows, params.Cols };
    }

    [[nodiscard]] constexpr auto ArgMax() const {
        auto result = Tensor<T, TensorParams{params.Rows, 1uz}>{};

        for (auto i = 0uz; const auto& row : m_data) {
            auto max = std::ranges::max_element(row);
            auto arg = std::ranges::distance(std::begin(row), max);
            result.FillAt(i, 0uz, arg);
            ++i;
        }

        return result;
    }

    [[nodiscard]] constexpr auto Where(std::function<bool(T&)> what) {
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

    constexpr auto Mask(std::array<std::array<std::size_t, params.Cols>, params.Rows> mask, T maskValue) {
        for (auto i : std::ranges::iota_view(0uz, params.Rows)) {
            for (auto j : std::ranges::iota_view(0uz, params.Cols)) {
                if (mask.at(i).at(j) == 1) {
                    m_data.at(i).at(j) = maskValue;
                }
            }
        }
    }

    constexpr auto ForEachElement(std::function<void(T&)> func) {
        for (auto& row : m_data)
            std::ranges::for_each(row, func);
    }

    constexpr auto Log() {
        ForEachElement([](auto& el){ el = std::log(el); });
    }

    constexpr auto Clip(T min, T max)  {
        ForEachElement([=](auto& el){ el = std::clamp(el, min, max); });
    }

    constexpr auto FillAt(std::size_t i, std::size_t j, T value) {
        m_data.at(i).at(j) = value;
    }

    constexpr auto Negative() {
        ForEachElement([](auto& el){ el != T{} ? el = -el : el = T{};});
    }

    constexpr auto Fill(T value) {
        std::ranges::for_each(m_data, [=](auto& row) { std::ranges::fill(row, value); });
    }

    constexpr auto ReLU() {
        ForEachElement( [](auto& el){ el = std::max(T{}, el); });
    }

    constexpr auto SubtractMaxFromEachRow() {
        for (auto& row : m_data) {
            const auto max = *std::ranges::max_element(std::begin(row), std::end(row));
            for (auto& el : row) {
                el -= max;
            }
        }
    }

    constexpr auto MultiplyEachElementBy(T value) {
        ForEachElement( [=](auto& el){ el *= value; } );
    }

    auto FillWithRandomValues(std::pair<T, T> range) {
        std::ranges::for_each(m_data, [=](auto& row) {
            std::ranges::generate(row, [&](){ return RandomFloat(range); } );
        });
    }

    // TODO: error: use of undeclared identifier 'assert' - no lib?
    constexpr auto DivideRowBy(std::size_t row, T value) {
        for (auto& el : m_data.at(row)) {
            el /= value;
        }
    }

    // TODO: Should accept 1D array
    constexpr auto ExchangeRow(std::size_t row, std::array<std::array<T, params.Cols>, 1uz> data) {
        m_data.at(row) = data.at(0uz);
    }

    constexpr auto Print() const {
        for (const auto& row : m_data) {
            for (const auto& el : row) {
                std::cout << el << "  ";
            }
            std::cout << '\n';
        }
    }

    constexpr auto PrintShape() const {
        std::cout << "Shape = (" << params.Rows << ", " << params.Cols << ")\n";
    }

    constexpr auto operator- () {
        this->Negative();
    }

    constexpr auto operator/ (T value) {
        ForEachElement( [=](auto& el){ el /= value; } );
    }

    friend constexpr auto& operator<< (std::ostream& stream, const Tensor& tensor) {
        if constexpr (params.Rows == 1uz && params.Cols == 1uz) {
            stream << tensor.at(0uz, 0uz);
        }
        else {
            for (const auto& row : tensor.Data()) {
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
