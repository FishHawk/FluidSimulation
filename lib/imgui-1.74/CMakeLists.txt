find_package(glfw3 REQUIRED)

file(GLOB IMGUI_SOURCES ./*.cpp)
file(GLOB IMGUI_HEADERS ./*.h)
add_library(imgui STATIC ${IMGUI_SOURCES} ${IMGUI_SOURCES})

target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)

target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)

target_link_libraries(imgui
    PRIVATE
        glfw
        glad)