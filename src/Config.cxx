module;

export module config;

export namespace Config
{
    inline constexpr auto irisPath     = "assets/iris/iris.data";
    inline constexpr auto dataRows     = 150;
    inline constexpr auto dataCols     = 4;

    inline constexpr auto numClasses   = 3;
    inline constexpr auto epochs       = 10;
    inline constexpr auto batchSize    = 10;
    inline constexpr auto learningRate = 0.5f;
}
