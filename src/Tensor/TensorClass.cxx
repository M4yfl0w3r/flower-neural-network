export module tensor:implementation;

import std;
import utilities;

export struct TensorParams
{
    int Rows;
    int Cols;
};

export template<typename T, TensorParams params>
class Tensor final
{
public:
    constexpr Tensor() = default;

    explicit constexpr Tensor(std::array<std::array<T, params.Cols>, params.Rows> data)
        : m_data{ data }
    {}

    explicit constexpr Tensor(T value)
    {
        Fill(value);
    }

    [[nodiscard]] constexpr auto At(int x, int y) const
    {
        return m_data.at(x).at(y);
    }

    [[nodiscard]] constexpr auto At(int x) const
    {
        static_assert(params.Cols == 1);
        return m_data.at(x).at(0);
    }

    [[nodiscard]] constexpr auto Data() const
    {
        return m_data;
    }

    [[nodiscard]] constexpr auto Get() const
    {
        static_assert(params.Rows == 1 && params.Cols == 1);
        return m_data.at(0).at(0);
    }

    [[nodiscard]] constexpr auto Mean() const
    {
        auto sum = T{};
        for (const auto& row : m_data) {
            sum += std::accumulate(std::begin(row), std::end(row), T{});
        }
        return Tensor1D(sum / (params.Rows * params.Cols));
    }

    [[nodiscard]] constexpr auto SumAllElements() const
    {
        auto sum = T{};
        for (const auto& row : m_data) {
            sum += std::accumulate(std::begin(row), std::end(row), T{});
        }
        return Tensor1D(sum);
    }

    [[nodiscard]] constexpr auto SumEachRow() const
    {
        auto result = Tensor<T, { params.Rows, 1 }>{};

        // TODO: enumerate
        for (auto i = 0; auto& row : m_data) {
            const auto rowSum = std::accumulate(std::begin(row), std::end(row), T{});
            result.FillAt(i, 0, rowSum);
            ++i;
        }

        return result;
    }

    [[nodiscard]] constexpr auto SumEachColumn() const
    {
        auto result = std::array<std::array<T, params.Cols>, 1>{ T{} };

        for (const auto& row : m_data) {
            for (auto i = 0; const auto& el : row) {
                result.at(0).at(i) += el;
                ++i;
            }
        }

        return Tensor<T, { 1, params.Cols }>(result);
    }

    [[nodiscard]] constexpr auto Exp()
    {
        auto result = Tensor<T, params>{m_data};
        result.ForEachElement( [](auto& el){ el = std::exp(el); });
        return result;
    }

    [[nodiscard]] constexpr auto Shape() const
    {
        return std::pair { params.Rows, params.Cols };
    }

    [[nodiscard]] constexpr auto ArgMax() const
    {
        auto result = Tensor<T, {params.Rows, 1}>{};

        // TODO: enumerate
        for (auto i = 0; const auto& row : m_data) {
            auto max = std::ranges::max_element(row);
            auto arg = std::ranges::distance(std::begin(row), max);
            result.FillAt(i, 0, arg);
            ++i;
        }

        return result;
    }

    [[nodiscard]] constexpr auto Where(std::function<bool(T&)> what)
    {
        // Get indices that satisfy a condition specified by the 'what' function.

        std::array<std::array<int, params.Cols>, params.Rows> mask{ 0 };

        for (auto i = 0; auto& row : m_data) {
            auto it = std::find_if(std::begin(row), std::end(row), what);

            while (it != std::end(row)) {
                auto j = std::distance(std::begin(row), it);
                mask.at(i).at(j) = 1;
                it = std::find_if(std::next(it), std::end(row), what);
            }

            ++i;
        }

        return mask;
    }

    constexpr auto Mask(std::array<std::array<int, params.Cols>, params.Rows> mask, T maskValue)
    {
        for (auto i : std::ranges::iota_view(0, params.Rows)) {
            for (auto j : std::ranges::iota_view(0, params.Cols)) {
                if (mask.at(i).at(j) == 1) {
                    m_data.at(i).at(j) = maskValue;
                }
            }
        }
    }

    constexpr auto ForEachElement(std::function<void(T&)> func)
    {
        for (auto& row : m_data) {
            std::ranges::for_each(row, func);
        }
    }

    constexpr auto Log()
    {
        ForEachElement([](auto& el){ el = std::log(el); });
    }

    constexpr auto Clip(T min, T max)
    {
        ForEachElement([=](auto& el){ el = std::clamp(el, min, max); });
    }

    constexpr auto FillAt(int i, int j, T value)
    {
        m_data.at(i).at(j) = value;
    }

    constexpr auto Negative()
    {
        ForEachElement([](auto& el){ el != T{} ? el = -el : el = T{};});
    }

    constexpr auto Fill(T value)
    {
        std::ranges::for_each(m_data, [=](auto& row) { std::ranges::fill(row, value); });
    }

    constexpr auto ReLU()
    {
        ForEachElement( [](auto& el){ el = std::max(T{}, el); });
    }

    constexpr auto SubtractMaxFromEachRow()
    {
        for (auto& row : m_data) {
            const auto max = *std::ranges::max_element(std::begin(row), std::end(row));
            for (auto& el : row) {
                el -= max;
            }
        }
    }

    constexpr auto MultiplyEachElementBy(T value)
    {
        ForEachElement( [=](auto& el){ el *= value; } );
    }

    auto FillWithRandomValues(std::pair<T, T> range)
    {
        std::ranges::for_each(m_data, [=](auto& row) {
            std::ranges::generate(row, [&](){ return RandomFloat(range); } );
        });
    }

    // TODO: error: use of undeclared identifier 'assert' - no lib?
    constexpr auto DivideRowBy(int row, T value)
    {
        for (auto& el : m_data.at(row)) {
            el /= value;
        }
    }

    // TODO: Should accept 1D array
    constexpr auto ExchangeRow(int row, std::array<std::array<T, params.Cols>, 1> data)
    {
        m_data.at(row) = data.at(0);
    }

    constexpr auto Print() const
    {
        for (const auto& row : m_data) {
            for (const auto& el : row) {
                std::print("{} ", el);
            }
            std::println("");
        }
    }

    constexpr auto PrintShape() const
    {
        std::println("Shape = {}, {}", params.Rows, params.Cols);
    }

    constexpr auto operator- ()
    {
        this->Negative();
    }

    constexpr auto operator/ (T value)
    {
        ForEachElement( [=](auto& el){ el /= value; } );
    }

    friend constexpr auto& operator<< (std::ostream& stream, const Tensor& tensor)
    {
        if constexpr (params.Rows == 1 && params.Cols == 1) {
            stream << tensor.at(0, 0);
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
    constexpr auto Tensor1D(auto value) const
    {
        return Tensor<T, { 1, 1 }>(std::array<std::array<T, 1>, 1>({ value }));
    }

    std::array<std::array<T, params.Cols>, params.Rows> m_data;
};
