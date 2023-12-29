workspace "Net"
    configurations "Debug"

project "Net"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++20"
    targetdir "build/%{cfg.buildcfg}"
    files { "src/*.cpp" }

    filter { "system:linux", "action:gmake" }
        buildoptions { "-Wall", 
                       "-Wextra",
                       "-Wpedantic", 
                       "-Werror", 
                       "-Wconversion", 
                       "-Wfloat-equal" }

    filter "system:linux or macosx"
        location "build"

    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
