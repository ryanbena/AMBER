cmake_minimum_required(VERSION 3.20)
project(PoissonYaw)

find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(osqp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -pthread -pg -fopenmp")

add_definitions(-DUSE_CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

set(CUDA_SOURCES src/kernel.cu)

cuda_add_library(cuda_kernel STATIC ${CUDA_SOURCES})
target_include_directories(cuda_kernel PUBLIC inc/)
set_target_properties(cuda_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(poissonYaw src/main.cpp src/utils.cpp)
target_link_libraries(poissonYaw Eigen3::Eigen OsqpEigen::OsqpEigen cuda_kernel)
ament_target_dependencies(poissonYaw rclcpp geometry_msgs std_msgs)

#install(TARGETS
#  talker
#  DESTINATION lib/${PROJECT_NAME})

#ament_package()