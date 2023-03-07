get_directory_property(has_parent PARENT_DIRECTORY)
set(name "YOLOv6RgbSpatial")
if (NOT has_parent)
    project(${name})
    #    include(cmake/CMakeRC.cmake)
    find_package(depthai CONFIG REQUIRED)
    find_package(OpenCV CONFIG REQUIRED)
endif ()

add_executable(${name} main.cpp)
target_link_libraries(${name} PRIVATE depthai::opencv opencv_highgui models)

if (WIN32)
    add_custom_command(TARGET ${name}
            # 在目标中的所有其他规则都已执行后运行
            POST_BUILD
            # 指定要在构建时执行的命令行
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${name}> $<TARGET_FILE_DIR:${name}>
            # 参数中的列表COMMAND将被扩展, 包括那些用生成器表达式创建的列表
            COMMAND_EXPAND_LISTS
            )
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        install(TARGETS ${name} RUNTIME_DEPENDENCY_SET runtime_set)
    endif ()
else ()
    install(TARGETS ${name})
endif ()

if (NOT MSVC)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>)
endif ()