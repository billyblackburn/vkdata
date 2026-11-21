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

find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

function(vkn_add_shader SHADER_SRC VAR_NAME OUT_HEADER)
    add_custom_command(
        OUTPUT "${OUT_HEADER}"
        COMMAND "${GLSLANG_VALIDATOR}" -V --vn "${VAR_NAME}" -o "${OUT_HEADER}" "${SHADER_SRC}"
        DEPENDS "${SHADER_SRC}"
        COMMENT "Compiling Vulkan shader ${SHADER_SRC} -> ${OUT_HEADER}"
    )
endfunction()

set(VKN_SPV_HEADERS
    "${CMAKE_CURRENT_BINARY_DIR}/vkn_spv_bincount_i32.h"
    "${CMAKE_CURRENT_BINARY_DIR}/vkn_spv_bincount_wf32.h"
    "${CMAKE_CURRENT_BINARY_DIR}/vkn_spv_histogram2d_f32.h"
    "${CMAKE_CURRENT_BINARY_DIR}/vkn_spv_histogram1d_f32.h"
)
vkn_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/bincount_i32.comp"    vkn_spv_bincount_i32    "${CMAKE_CURRENT_BINARY_DIR}/vkn_spv_bincount_i32.h")
vkn_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/bincount_wf32.comp"   vkn_spv_bincount_wf32   "${CMAKE_CURRENT_BINARY_DIR}/vkn_spv_bincount_wf32.h")
vkn_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/histogram2d_f32.comp" vkn_spv_histogram2d_f32 "${CMAKE_CURRENT_BINARY_DIR}/vkn_spv_histogram2d_f32.h")
vkn_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/histogram1d_f32.comp" vkn_spv_histogram1d_f32 "${CMAKE_CURRENT_BINARY_DIR}/vkn_spv_histogram1d_f32.h")

set(VKNUMPY_SOURCES
    cpp/vknumpy_ggml.cpp
    cpp/vknumpy_ncnn.cpp
    cpp/vknumpy_vk_direct.cpp
    ${VKN_SPV_HEADERS}
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
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src/ggml-vulkan
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/llama.cpp/ggml/src/ggml-vulkan
)

target_link_libraries(vknumpy PRIVATE ${VKNUMPY_LIBS})

# Output libvknumpy.so directly into vknumpy/_lib/ for immediate Python import
set_target_properties(vknumpy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vknumpy/_lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vknumpy/_lib"
)

# Build libvkshader.so (zero-copy GPU shading & basemap compositing on vknumpy's VkDevice)
find_package(Vulkan REQUIRED)
set(VKS_SPV_HEADERS
    "${CMAKE_CURRENT_BINARY_DIR}/vks_spv_shade_stats.h"
    "${CMAKE_CURRENT_BINARY_DIR}/vks_spv_shade_cdf.h"
    "${CMAKE_CURRENT_BINARY_DIR}/vks_spv_shade_composite.h"
)
vkn_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/shade_stats.comp"     vks_spv_shade_stats     "${CMAKE_CURRENT_BINARY_DIR}/vks_spv_shade_stats.h")
vkn_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/shade_cdf.comp"       vks_spv_shade_cdf       "${CMAKE_CURRENT_BINARY_DIR}/vks_spv_shade_cdf.h")
vkn_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/shade_composite.comp" vks_spv_shade_composite "${CMAKE_CURRENT_BINARY_DIR}/vks_spv_shade_composite.h")

add_library(vkshader SHARED
    cpp/vkshader.cpp
    ${VKS_SPV_HEADERS}
)

target_include_directories(vkshader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(vkshader PRIVATE Vulkan::Vulkan)

set_target_properties(vkshader PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vkshader/_lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vkshader/_lib"
)

# Build libvkspatial.so (zero-copy GPU 2D Morton Z-order & point-in-polygon on vknumpy's VkDevice)
set(VKSP_SPV_HEADERS
    "${CMAKE_CURRENT_BINARY_DIR}/vksp_spv_morton_2d.h"
    "${CMAKE_CURRENT_BINARY_DIR}/vksp_spv_point_in_polygon.h"
)
vkn_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/morton_2d.comp"        vksp_spv_morton_2d        "${CMAKE_CURRENT_BINARY_DIR}/vksp_spv_morton_2d.h")
vkn_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/point_in_polygon.comp" vksp_spv_point_in_polygon "${CMAKE_CURRENT_BINARY_DIR}/vksp_spv_point_in_polygon.h")

add_library(vkspatial SHARED
    cpp/vkspatial.cpp
    ${VKSP_SPV_HEADERS}
)

target_include_directories(vkspatial PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(vkspatial PRIVATE Vulkan::Vulkan)

set_target_properties(vkspatial PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vkspatial/_lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vkspatial/_lib"
)

# Install rules for scikit-build-core wheel packaging (only our shared libs, not static ggml/ncnn archives)
install(TARGETS vknumpy
    LIBRARY DESTINATION vknumpy/_lib COMPONENT vkdata
    RUNTIME DESTINATION vknumpy/_lib COMPONENT vkdata
)
install(TARGETS vkshader
    LIBRARY DESTINATION vkshader/_lib COMPONENT vkdata
    RUNTIME DESTINATION vkshader/_lib COMPONENT vkdata
)
install(TARGETS vkspatial
    LIBRARY DESTINATION vkspatial/_lib COMPONENT vkdata
    RUNTIME DESTINATION vkspatial/_lib COMPONENT vkdata
)
