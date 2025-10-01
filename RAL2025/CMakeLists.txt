cmake_minimum_required(VERSION 3.20)
project(unitree_ros2_poisson_simple)

# Enable C++ and CUDA
enable_language(CUDA)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -pthread -pg -fopenmp")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)

# Definitions
add_definitions(-DUSE_CUDA)

# CUDA sources
set(CUDA_SOURCES src/kernel.cu)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

include_directories(include include/common include/nlohmann include/poisson)
link_directories(src)

set (
  DEPENDENCY_LIST
  unitree_go
  unitree_hg
  unitree_api
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  rosbag2_cpp
  pcl_conversions
  tf2_ros
  OpenCV
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(osqp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_library(cuda_kernel STATIC ${CUDA_SOURCES})
target_include_directories(cuda_kernel PUBLIC inc/)
set_target_properties(cuda_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

# Main executable
add_executable(poissonLocal src/poisson.cpp src/utils.cpp src/common/ros2_sport_client.cpp)
add_executable(cloud_merger src/cloud_merger.cpp src/utils.cpp src/common/ros2_sport_client.cpp)
add_executable(cloud_cropper src/cloud_cropper.cpp src/common/ros2_sport_client.cpp)

target_link_libraries(poissonLocal Eigen3::Eigen OsqpEigen::OsqpEigen cuda_kernel ${OpenCV_LIBS})
target_link_libraries(cloud_merger ${PCL_LIBRARIES})
target_link_libraries(cloud_cropper ${PCL_LIBRARIES})

ament_target_dependencies(poissonLocal ${DEPENDENCY_LIST})
ament_target_dependencies(cloud_merger ${DEPENDENCY_LIST})
ament_target_dependencies(cloud_cropper ${DEPENDENCY_LIST})

install(TARGETS
        poissonLocal
        cloud_merger
        cloud_cropper
        DESTINATION)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
