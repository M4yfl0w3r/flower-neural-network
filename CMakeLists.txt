cmake_minimum_required( VERSION 3.29 FATAL_ERROR )

project( flower_neural_network CXX )

set( CMAKE_CXX_STANDARD 23 )
set( CMAKE_CXX_STANDARD_REQUIRED YES )
set( CMAKE_CXX_EXTENSIONS OFF )

include(cmake/ModulesLocalBuild.cmake)
include(cmake/ProjectModules.cmake)

add_executable( ${PROJECT_NAME} src/main.cxx )

target_link_libraries( ${PROJECT_NAME} PRIVATE neuron )
