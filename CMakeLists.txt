cmake_minimum_required( VERSION 3.29.1 )

project( flower_neural_network )

set( CMAKE_CXX_STANDARD 23 )
set( CMAKE_CXX_STANDARD_REQUIRED YES )
set( CMAKE_CXX_EXTENSIONS OFF )

file(
    GLOB SOURCES
        ${PROJECT_SOURCE_DIR}/src/Config.hpp
        ${PROJECT_SOURCE_DIR}/src/Dataset.cpp
        ${PROJECT_SOURCE_DIR}/src/Dataset.hpp
        ${PROJECT_SOURCE_DIR}/src/DenseLayer.hpp
        ${PROJECT_SOURCE_DIR}/src/Loss.hpp
        ${PROJECT_SOURCE_DIR}/src/Tensor.hpp
        ${PROJECT_SOURCE_DIR}/src/Utils.hpp
        
        ${PROJECT_SOURCE_DIR}/src/main.cpp
)

add_executable( ${PROJECT_NAME} ${SOURCES} )

