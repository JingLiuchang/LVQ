set(CMAKE_CXX_STANDARD 17)

add_executable(lvq_index LVQ/index.cpp)
add_executable(lvq_search LVQ/search.cpp)

add_library(lvq_p1_bridge SHARED p1_bridge.cpp)
set_target_properties(
    lvq_p1_bridge
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)
