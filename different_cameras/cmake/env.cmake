if (NOT WIN32)
    # 是否在 macOS 和 iOS 上使用 rpaths。
    set(CMAKE_MACOSX_RPATH ON)
    # 如果此变量设置为 true，则软件始终在没有 RPATH 的情况下构建。
    set(CMAKE_SKIP_BUILD_RPATH OFF)
    # 如果此变量设置为 true，则始终使用 RPATH 的安装路径构建软件，并且在安装时不需要重新链接。
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    # 将路径添加到链接器搜索和安装的 rpath。
    # 如果设置为 True，则将项目外部位于链接器搜索路径中或包含链接库文件的任何目录附加到已安装二进制文件的运行时搜索路径 (rpath) 中。
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif ()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory" FORCE)
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()

include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR} CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Installation directory for header files")
if (WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKEDIR CMake)
else ()
    set(DEF_INSTALL_CMAKEDIR share/cmake/${PROJECT_NAME})
endif ()

set(INSTALL_CMAKEDIR ${DEF_INSTALL_CMAKEDIR} CACHE PATH "Installation directory for CMake files")

# Report to user
foreach (p LIB BIN INCLUDE CMAKE)
    file(TO_NATIVE_PATH ${CMAKE_INSTALL_PREFIX}/${INSTALL_${p}DIR} _path)
    message(STATUS "Installing ${p} components to ${_path}")
    unset(_path)
endforeach ()