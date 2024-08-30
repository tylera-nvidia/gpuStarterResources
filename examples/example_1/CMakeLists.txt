cmake_minimum_required(VERSION 3.18)
project(matxExample LANGUAGES CUDA CXX)
set(CMAKE_BUILD_TYPE Release)
find_package(CUDAToolkit 11.5 REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 80)

#include NVTX
# add_subdirectory(${CUDAToolkit_INCLUDE_DIRS}/nvtx3/)
# Find NVTX3
# find_package(NVTX3 REQUIRED)
# include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

########################################
# Clustering Example
#########################################
set(TARGET example_1)
add_executable(${TARGET} example_1.cu)
target_compile_options(${TARGET} PRIVATE  -lineinfo )
# target_link_libraries(${TARGET} PRIVATE matx::matx)
target_link_libraries(${TARGET} PRIVATE CUDA::nvToolsExt)
