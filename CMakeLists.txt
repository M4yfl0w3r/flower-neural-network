cmake_minimum_required( VERSION 3.30 FATAL_ERROR )

set( CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508" )
set( CMAKE_CXX_MODULE_STD ON )
set( CMAKE_CXX_SCAN_FOR_MODULES ON )
set( CMAKE_CXX_STANDARD 23 )
set( CMAKE_CXX_STANDARD_REQUIRED YES )
set( CMAKE_CXX_EXTENSIONS ON )

project( flower_neural_network LANGUAGES CXX )

add_library( neuron )

target_sources( neuron
    PUBLIC
        FILE_SET CXX_MODULES FILES
            ${PROJECT_SOURCE_DIR}/src/TensorModule.cxx
            ${PROJECT_SOURCE_DIR}/src/TensorOperators.cxx
            ${PROJECT_SOURCE_DIR}/src/TensorClass.cxx
            ${PROJECT_SOURCE_DIR}/src/TensorHelpers.cxx
            ${PROJECT_SOURCE_DIR}/src/Utilities.cxx
            ${PROJECT_SOURCE_DIR}/src/Config.cxx
            ${PROJECT_SOURCE_DIR}/src/Dataset.cxx
            ${PROJECT_SOURCE_DIR}/src/DenseLayer.cxx
            ${PROJECT_SOURCE_DIR}/src/Loss.cxx
)

add_executable( ${PROJECT_NAME} main.cxx )
target_link_libraries( ${PROJECT_NAME} PRIVATE neuron )
