cmake_minimum_required( VERSION 3.29 FATAL_ERROR )

cmake_policy( VERSION 3.28 )

include( FetchContent )
FetchContent_Declare(
    std
    URL "file://${LIBCXX_BUILD}/modules/c++/v1/"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM
)
FetchContent_MakeAvailable( std )

link_libraries( std c++ )
