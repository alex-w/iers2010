# Disable warnings if needed
add_compile_options(-Wno-unused-variable -Wno-unused-but-set-variable)

# Add benchmark executable
add_executable(bm_sh_basis_functions
    bm_sh_basis_functions.cpp
    ${CMAKE_SOURCE_DIR}/costg-benchmark/bin/costg_utils.cpp
)

# Link with benchmark and project libraries
target_link_libraries(bm_sh_basis_functions
    PRIVATE
        benchmark::benchmark
        iers
        ${PROJECT_DEPENDENCIES}
)

# Include project headers
target_include_directories(bm_sh_basis_functions PRIVATE
    ${CMAKE_SOURCE_DIR}/costg-benchmark/bin
)
