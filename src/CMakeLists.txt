find_package(glm REQUIRED)

find_package(glfw3 REQUIRED)

find_package(CUDA REQUIRED)
find_library(CUDART_LIB cudart
    HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR})
add_library(Cuda::cudart IMPORTED INTERFACE)
set_target_properties(Cuda::cudart PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    INTERFACE_LINK_LIBRARIES
        ${CUDART_LIB})

add_executable(fluid_simulation
    SceneBuilder.cpp
    Input.cpp
    Ui.cpp
    render/RenderSystem.cpp
    render/Mesh.cpp
    render/MeshBuilder.cpp
    render/Shader.cpp
    render/camera/Camera.cpp
    simulate/cpu/SimulateSystem.cpp
    simulate/cpu/Particles.cpp
    simulate/cuda/SimulateSystem.cpp
    simulate/cuda/FluidSolver.cu
    main.cpp)

target_include_directories(fluid_simulation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

target_link_libraries(fluid_simulation
    PRIVATE
        Cuda::cudart
        glm
        glad
        glfw
        imgui)