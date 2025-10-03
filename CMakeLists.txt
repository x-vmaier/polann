cmake_minimum_required(VERSION 3.16)
project(Polann VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(POLANN_BUILD_EXAMPLES "Build examples" ON)
option(POLANN_BUILD_TESTS "Build tests" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Output directories
if(CMAKE_CONFIGURATION_TYPES)
    foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER ${cfg} CFG_UPPER)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG_UPPER} "${CMAKE_BINARY_DIR}/${cfg}/bin")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG_UPPER} "${CMAKE_BINARY_DIR}/${cfg}/lib")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CFG_UPPER} "${CMAKE_BINARY_DIR}/${cfg}/lib")
    endforeach()
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

# Detect AVX2 support
if(MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(POLANN_ENABLE_AVX2 TRUE)
        message(STATUS "AVX2 enabled for MSVC x64")
    endif()
else()
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(POLANN_ENABLE_AVX2 TRUE)
        message(STATUS "AVX2 enabled for GCC/Clang")
    endif()
endif()

# Generate config.h
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/polann/config.h"
    @ONLY
)

# Add subdirectories
add_subdirectory(src)

if(POLANN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(POLANN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY include/polann
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install generated config.h
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/polann/config.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/polann
)

# Create and install package config files
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PolannConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/PolannConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/polann
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/PolannConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/PolannConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/PolannConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/polann
)
