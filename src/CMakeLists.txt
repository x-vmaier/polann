# Collect library source files
file(GLOB_RECURSE POLANN_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

# Create library target
add_library(polann ${POLANN_SOURCES})

# Add alias for consistent naming
add_library(polann::polann ALIAS polann)

# Set target properties
set_target_properties(polann PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "polann"
)

# Include directories
target_include_directories(polann
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler-specific options
target_compile_features(polann PUBLIC cxx_std_23)

if(MSVC)
    target_compile_options(polann PRIVATE /W4)
else()
    target_compile_options(polann PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(POLANN_ENABLE_AVX2)
    if(MSVC)
        target_compile_options(polann PRIVATE /arch:AVX2)
    else()
        target_compile_options(polann PRIVATE -mavx2)
    endif()
endif()

# Install library
install(TARGETS polann
    EXPORT PolannTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Export targets
install(EXPORT PolannTargets
    FILE PolannTargets.cmake
    NAMESPACE polann::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/polann
)
