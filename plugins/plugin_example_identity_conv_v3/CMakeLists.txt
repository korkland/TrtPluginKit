# Get the plugin folder name
get_filename_component(PLUGIN_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(TARGET_NAME ${CMAKE_PROJECT_NAME}_${PLUGIN_NAME})

file(GLOB PLUGIN_SOURCES *.cpp)
file(GLOB PLUGIN_CU_SOURCES *.cu)
file(GLOB COMMON_PLUGIN_SOURCES  ${CMAKE_SOURCE_DIR}/common/*.cpp)

add_library(${TARGET_NAME}
    SHARED
        ${PLUGIN_SOURCES}
        ${PLUGIN_CU_SOURCES}
        ${COMMON_PLUGIN_SOURCES}
        )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/base
        ${CMAKE_SOURCE_DIR}/common
        ${TensorRT_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        CUDA::cudart
        ${TensorRT_LIBRARIES}
)
