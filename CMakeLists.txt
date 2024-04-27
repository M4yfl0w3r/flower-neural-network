cmake_minimum_required( VERSION 3.29.1 )

project( flower_neural_network CXX )

set( CMAKE_CXX_STANDARD 23 )
set( CMAKE_CXX_STANDARD_REQUIRED YES )
set( CMAKE_CXX_EXTENSIONS OFF )

add_library( neuron )

target_sources( neuron 
    PUBLIC FILE_SET CXX_MODULES FILES
        ${PROJECT_SOURCE_DIR}/src/Config.cxx
        ${PROJECT_SOURCE_DIR}/src/Dataset.cxx
        ${PROJECT_SOURCE_DIR}/src/Tensor.cxx
        ${PROJECT_SOURCE_DIR}/src/Utilities.cxx
        ${PROJECT_SOURCE_DIR}/src/DenseLayer.cxx
        # ${PROJECT_SOURCE_DIR}/src/Loss.cxx
)
        
add_executable( ${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cxx )
target_link_libraries( ${PROJECT_NAME} neuron )
