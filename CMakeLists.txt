cmake_minimum_required( VERSION 3.29.0 FATAL_ERROR )

project( flower_neural_network LANGUAGES CXX )

set( CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508" )
set( CMAKE_CXX_MODULE_STD ON )

set( CMAKE_CXX_STANDARD 23 )
set( CMAKE_CXX_STANDARD_REQUIRED YES )
set( CMAKE_CXX_EXTENSIONS ON )

include(cmake/ProjectModules.cmake)

add_executable( ${PROJECT_NAME} src/main.cxx )

target_link_libraries( ${PROJECT_NAME} PRIVATE neuron )
