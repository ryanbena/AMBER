cmake_minimum_required(VERSION 3.5)
project(PoissonControllerMPC)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -pthread -pg -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(osqp REQUIRED)

MESSAGE( STATUS ">> --------------  USING CUDA --------------" )
# set( CUDA_TOOLKIT_ROOT_DIR "/opt/cuda" )
# set(CMAKE_C_COMPILER   /opt/cuda/bin/gcc)
# set(CMAKE_CXX_COMPILER /opt/cuda/bin/g++)

find_package(CUDA REQUIRED)
add_definitions(-DUSE_CUDA)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_89")

set(CUDA_SOURCES src/kernel.cu)

cuda_add_library(cuda_kernel STATIC ${CUDA_SOURCES})
target_include_directories(cuda_kernel PUBLIC inc/)

add_executable(poissonMPC src/main.cpp src/utils.cpp)
target_link_libraries(poissonMPC Eigen3::Eigen OsqpEigen::OsqpEigen cuda_kernel)
ament_target_dependencies(poissonMPC rclcpp geometry_msgs std_msgs)

#install(TARGETS
#  talker
#  DESTINATION lib/${PROJECT_NAME})

#ament_package()