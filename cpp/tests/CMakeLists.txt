FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(motion_planning_tests
    test_kinematics.cpp
    test_pose_control.cpp
    test_grid.cpp
    test_potential.cpp
    test_graph_search.cpp
    test_sampling.cpp
    test_dubins.cpp
    test_tracking.cpp
)
target_link_libraries(motion_planning_tests PRIVATE motion_planning Catch2::Catch2WithMain)
catch_discover_tests(motion_planning_tests)
