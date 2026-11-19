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

set(VKNUMPY_SOURCES
    cpp/vknumpy_ggml.cpp
    cpp/vknumpy_ncnn.cpp
)

set(VKNUMPY_LIBS
    ggml
    ggml-vulkan
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/CMakeLists.txt")
    set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
    set(NCNN_SYSTEM_GLSLANG ON CACHE BOOL "" FORCE)
    set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
    set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_SIMPLEOCV ON CACHE BOOL "" FORCE)
    set(NCNN_SIMPLEOMP ON CACHE BOOL "" FORCE)
    set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/ncnn)
    list(APPEND VKNUMPY_LIBS ncnn)
    set(VKNUMPY_HAS_NCNN ON)
endif()

add_library(vknumpy SHARED ${VKNUMPY_SOURCES})

if(VKNUMPY_HAS_NCNN)
    target_compile_definitions(vknumpy PRIVATE VKNUMPY_USE_NCNN=1)
endif()

target_include_directories(vknumpy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)

target_link_libraries(vknumpy PRIVATE ${VKNUMPY_LIBS})

# Output libvknumpy.so directly into vknumpy/_lib/ for immediate Python import
set_target_properties(vknumpy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vknumpy/_lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vknumpy/_lib"
)
