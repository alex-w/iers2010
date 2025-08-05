# Define the executable
add_executable(displacement displacement.cpp)

# Link the executable to the required libraries
target_link_libraries(displacement PRIVATE iers ${PROJECT_DEPENDENCIES})
