cmake_minimum_required(VERSION 3.22)

project(poisson_modular LANGUAGES C CXX CUDA)

find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(osqp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(poisson_msgs REQUIRED)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(pcl_conversions REQUIRED)

find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIRS})

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set (
  DEPENDENCY_LIST
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  pcl_conversions
  px4_msgs
  poisson_msgs
  OpenCV
)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -pthread")

add_definitions(-DUSE_CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

set(CUDA_SOURCES src/kernel.cu)

cuda_add_library(cuda_kernel STATIC ${CUDA_SOURCES})
target_include_directories(cuda_kernel PUBLIC include/)
set_target_properties(cuda_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(poisson src/controller.cpp src/utils.cpp)
target_link_libraries(poisson Eigen3::Eigen OsqpEigen::OsqpEigen cuda_kernel ${OpenCV_LIBS})
ament_target_dependencies(poisson ${DEPENDENCY_LIST})

add_executable(teleOp src/teleop.cpp)
target_link_libraries(teleOp ${CURSES_LIBRARIES})
ament_target_dependencies(teleOp ${DEPENDENCY_LIST})

install(TARGETS
        poisson
        teleOp
        DESTINATION)