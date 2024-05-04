module;

export module config;

export namespace Mayflower::Config 
{
    inline constexpr auto irisPath     = "assets/iris/iris.data";
    inline constexpr auto dataRows     = 150u;
    inline constexpr auto dataCols     = 4uz;
    inline constexpr auto labelPos     = 10u;
    inline constexpr auto numClasses   = 3uz;
    inline constexpr auto learningRate = 0.1f;
    inline constexpr auto epochs       = 2uz;
    inline constexpr auto batchSize    = 3uz;
}
