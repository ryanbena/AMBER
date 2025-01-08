project(Poisson)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -pthread -pg -fopenmp")

find_package(Eigen3 REQUIRED)
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
target_link_libraries(cuda_kernel Eigen3::Eigen)

add_executable(poisson src/main.cpp src/utils.cpp)
target_link_libraries(poisson Eigen3::Eigen cuda_kernel)