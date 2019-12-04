set(CMAKE_CXX_STANDARD 17)

find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDA REQUIRED)

add_executable(fluid_simulation
    SceneBuilder.cpp
    render/RenderSystem.cpp
    render/Mesh.cpp
    render/MeshBuilder.cpp
    render/Shader.cpp
    render/Camera.cpp
    simulate/cpu/SimulateSystem.cpp
    simulate/cpu/Particles.cpp
    simulate/cpu/SplineInterpolation.cpp
    simulate/cuda/SimulateSystem.cpp
    simulate/cuda/FluidSolver.cu
    main.cpp)

target_include_directories(fluid_simulation 
    PRIVATE 
        ${CUDA_INCLUDE_DIRS})

target_link_libraries(fluid_simulation
    PRIVATE
        ${CMAKE_DL_LIBS}
        ${glm_LIBRARIES}
        ${CUDA_LIBRARIES}
        glad
        glfw
        pthread)