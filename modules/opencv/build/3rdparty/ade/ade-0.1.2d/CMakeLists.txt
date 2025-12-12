# Copyright (C) 2018 Intel Corporation
#
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.5)

project(ade)

if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

option(ENABLE_ADE_TESTING      "Build tests, require google test" OFF)
option(BUILD_ADE_TUTORIAL      "Build tutorial samples"           OFF)
option(FORCE_ADE_ASSERTS       "Always enable ADE_ASSERT"         OFF)
option(BUILD_ADE_DOCUMENTATION "Build doxygen documentation"      OFF)
option(BUILD_WITH_STATIC_CRT   "Build with static multi-threaded C Runtime (MS Windows/Visual Studio only)" OFF)

set(CMAKE_CXX_STANDARD 11)

# TODO: this is horrible hack, we must follow cmake
# build/install policy
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

function(add_security_flags target)
    if(UNIX)
        target_compile_definitions( ${target} PRIVATE _FORTIFY_SOURCE=2 )
        if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
                target_compile_options( ${target} PRIVATE -fstack-protector)
            else()
                target_compile_options( ${target} PRIVATE -fstack-protector-strong)
            endif()
        elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
            target_compile_options( ${target} PRIVATE -fstack-protector-strong)
        endif()
    elseif(WIN32)
        if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
            target_compile_options( ${target} PRIVATE /GS /DynamicBase)
            if(BUILD_WITH_STATIC_CRT)
                target_compile_options( ${target} PRIVATE "/MT")
            endif()
            if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
                # These options for 32 bit builds only
                target_compile_options( ${target} PRIVATE /SAFESEH /NXCOMPAT )
            endif()
        elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
            target_compile_definitions( ${target} PRIVATE _FORTIFY_SOURCE=2 )
            target_compile_options( ${target} PRIVATE -fstack-protector-strong )
        endif()
    endif()
endfunction()

if (MSVC)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
endif(MSVC)

add_subdirectory(sources/ade/)

if(ENABLE_ADE_TESTING)
    enable_testing()
    add_subdirectory(sources/tests)
endif()

if(BUILD_ADE_TUTORIAL)
    add_subdirectory(tutorial)
endif()

# cpack

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CPACK_STRIP_FILES ON)
endif()
set(CPACK_PACKAGE_VERSION ${THE_PROJECT_VERSION})
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_GENERATOR "ZIP")

include(CPack)
