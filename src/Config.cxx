module;

export module config;

export namespace Mayflower::Config 
{
    inline constexpr auto irisPath     = "assets/iris/iris.data";
    inline constexpr auto dataRows     = 150uz;
    inline constexpr auto dataCols     = 4uz;

    inline constexpr auto numClasses   = 3uz;
    inline constexpr auto epochs       = 20uz;
    inline constexpr auto batchSize    = 10uz;
    inline constexpr auto learningRate = 0.5f;
}
