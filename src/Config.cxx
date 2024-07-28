module;

export module config;

export namespace Config
{
    inline constexpr auto IrisPath     = "assets/iris/iris.data";
    inline constexpr auto DataRows     = 150;
    inline constexpr auto DataCols     = 4;

    inline constexpr auto NumClasses   = 3;
    inline constexpr auto Epochs       = 10;
    inline constexpr auto BatchSize    = 10;
    inline constexpr auto LearningRate = 0.5f;
}
