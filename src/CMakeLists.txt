set(CMAKE_CXX_STANDARD 17)

find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(fluid_simulation
    render/Shader.cpp
    render/Camera.cpp
    simulation/FluidSolver.cpp
    main.cpp)

target_link_libraries(fluid_simulation
    PRIVATE
        ${CMAKE_DL_LIBS}
        ${glm_LIBRARIES}
        glad
        glfw
        pthread)