cmake_minimum_required(VERSION 3.10)
project("DifferentCameras")

include(cmake/env.cmake)
include(cmake/CMakeRC.cmake)
message(CHECK_START "Finding depthai")
find_package(depthai CONFIG QUIET)
if (NOT depthai_FOUND)
    set(depthai_DIR ${depthai_DIR_CACHE})
    find_package(depthai CONFIG REQUIRED)
endif ()
if (depthai_FOUND)
    message(CHECK_PASS "${depthai_DIR} (version: ${depthai_VERSION})")
    set(depthai_DIR_CACHE ${depthai_DIR} CACHE PATH "The directory containing a CMake configuration file for depthai.")
else ()
    message(CHECK_FAIL "not found")
endif ()

message(CHECK_START "Finding OpenCV")
find_package(OpenCV CONFIG QUIET)
if (NOT OpenCV_FOUND)
    set(OpenCV_DIR ${OpenCV_DIR_CACHE})
    find_package(OpenCV CONFIG QUIET)
endif ()
if (OpenCV_FOUND)
    message(CHECK_PASS "${OpenCV_DIR} (version: ${OpenCV_VERSION})")
    set(OpenCV_DIR_CACHE ${OpenCV_DIR} CACHE PATH "The directory containing a CMake configuration file for OpenCV.")
else ()
    message(CHECK_FAIL "not found")
endif ()

CMRC_ADD_RESOURCE_LIBRARY(models yolov6n.blob)

add_subdirectory(yolov6-rgb)
add_subdirectory(yolov6-rgb-spatial)
add_subdirectory(yolov6-left)
add_subdirectory(yolov6-left-spatial)
add_subdirectory(yolov6-right)
add_subdirectory(yolov6-right-spatial)


if (WIN32)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        # https://cmake.org/cmake/help/latest/command/file.html#get-runtime-dependencies
        # https://cmake.org/cmake/help/latest/command/install.html#installing-runtime-dependencies
        # 安装运行时依赖项
        install(RUNTIME_DEPENDENCY_SET runtime_set
                # 预包含正则表达式列表，通过这些正则表达式过滤尚未解决的依赖项的名称。
                PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
                # 用于过滤已解析依赖项名称的排除后正则表达式列表。
                POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
                )
    endif ()
endif ()