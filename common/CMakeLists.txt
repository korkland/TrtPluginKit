# Collect all source files
file(GLOB COMMON_SOURCES *.cpp)
file(GLOB COMMON_HEADERS *.h)

add_library(common STATIC ${COMMON_SOURCES} ${COMMON_HEADERS})

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(common PUBLIC ${TensorRT_LIBRARIES})