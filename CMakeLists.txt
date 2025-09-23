cmake_minimum_required(VERSION 3.16)
project(NeuralNetwork VERSION 1.0.0 LANGUAGES CXX)

# Project metadata
set(PROJECT_DISPLAY_NAME "Neural Network")
set(PROJECT_IDENTIFIER "com.xvmaier.neuralnetwork")
set(PROJECT_CREATOR "Valentin Maier")
set(PROJECT_COPYRIGHT "Copyright (c) 2025 Valentin Maier, LLC")

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For clangd, compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories for single/multi-config generators
if(CMAKE_CONFIGURATION_TYPES)
    foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER ${cfg} CFG_UPPER)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG_UPPER} "${CMAKE_BINARY_DIR}/${cfg}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG_UPPER} "${CMAKE_BINARY_DIR}/${cfg}")
    endforeach()
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

# Main executable
add_executable(${PROJECT_NAME})

# Apply project metadata
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.h"
    @ONLY
)

# Public headers
target_include_directories(${PROJECT_NAME}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
)

# Source files
add_subdirectory(src)

# Link libraries
#target_link_libraries(${PROJECT_NAME} PRIVATE)

# Assets copy
set(ASSETS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${ASSETS_SOURCE_DIR}"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
    COMMENT "Copying assets..."
)
