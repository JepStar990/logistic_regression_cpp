cmake_minimum_required(VERSION 3.10)
project(LogisticRegression)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Main executable
add_executable(logistic_regression
    src/main.cpp
    src/DataLoader.cpp
    src/LogisticRegression.cpp
    src/Metrics.cpp
)

# Link Eigen
target_link_libraries(logistic_regression Eigen3::Eigen)

# Test executable
add_executable(test_models
    tests/test_models.cpp
    src/DataLoader.cpp
    src/LogisticRegression.cpp
    src/Metrics.cpp
)
target_link_libraries(test_models Eigen3::Eigen)

# Example executable
add_executable(example
    examples/example_usage.cpp
    src/DataLoader.cpp
    src/LogisticRegression.cpp
    src/Metrics.cpp
)
target_link_libraries(example Eigen3::Eigen)
