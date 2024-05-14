cmake_minimum_required( VERSION 3.28 FATAL_ERROR )

add_library( neuron )

target_sources( neuron
    PUBLIC 
        FILE_SET CXX_MODULES FILES
            ${PROJECT_SOURCE_DIR}/src/Tensor.cxx
            ${PROJECT_SOURCE_DIR}/src/Utilities.cxx
            ${PROJECT_SOURCE_DIR}/src/Config.cxx
            ${PROJECT_SOURCE_DIR}/src/Dataset.cxx
            ${PROJECT_SOURCE_DIR}/src/DenseLayer.cxx
            ${PROJECT_SOURCE_DIR}/src/Loss.cxx
)
