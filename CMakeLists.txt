cmake_minimum_required(VERSION 3.20)
project(vknumpy LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build ggml (with Vulkan backend) statically and embed into libvknumpy.so
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp/ggml)

add_library(vknumpy SHARED
    cpp/vknumpy_ggml.cpp
)

target_include_directories(vknumpy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)

target_link_libraries(vknumpy PRIVATE
    ggml
    ggml-vulkan
)

# Output libvknumpy.so directly into vknumpy/_lib/ for immediate Python import
set_target_properties(vknumpy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vknumpy/_lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vknumpy/_lib"
)
