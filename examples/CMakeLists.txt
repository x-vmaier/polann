# Collect example source files
file(GLOB EXAMPLE_SOURCES "*.cpp")

# Add matplotplusplus
add_subdirectory("../vendored/matplotplusplus" "matplot_build")

# Disable nodiscard warning
if(MSVC)
    target_compile_options(matplot PRIVATE /wd4834)
endif()

# Create an executable for each example
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE polann::polann matplot)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )
endforeach()
