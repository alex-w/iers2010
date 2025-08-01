# Find pybind11 (you can also use FetchContent if it's not installed)
find_package(pybind11 REQUIRED)

# Create the Python module
pybind11_add_module(pyiers bindings.cpp)

# Link it with your library
target_link_libraries(pyiers PRIVATE iers)
target_include_directories(pyiers PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Make sure the output is placed where Python can find it
set_target_properties(pyiers PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pybind
)

