# Copyright (C) 2018 Intel Corporation
#
#
# SPDX-License-Identifier: Apache-2.0
#

include(ExternalProject)

if(DEFINED GTEST_ROOT)
    find_package(GTest REQUIRED)
else()
    set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)

    set(GTEST_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party/gtest-bin)

    ExternalProject_Add(
        GTest
        URL https://github.com/google/googletest/archive/release-1.8.1.zip
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=${GTEST_INSTALL_DIR} -Dgtest_force_shared_crt=ON
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON
      )

    set(GTEST_LIBRARIES_PATH ${GTEST_INSTALL_DIR}/lib/)

    set(GTEST_LIB_PREFIX ${CMAKE_FIND_LIBRARY_PREFIXES})
    if( WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
      # Somehow in this case we get GTest compiled as gtest.lib and
      # prefix equals to "pub;"
      # Workarounds, workarounds...
      set(GTEST_LIB_PREFIX "")
    endif()

    set(GTEST_LIBRARY_PATH ${GTEST_LIBRARIES_PATH}/${GTEST_LIB_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(GTEST_LIBRARY gtest)
    add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
    set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARY_PATH})
    add_dependencies(${GTEST_LIBRARY} GTest)

    set(GTEST_MAIN_LIBRARY_PATH ${GTEST_LIBRARIES_PATH}/${GTEST_LIB_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(GTEST_MAIN_LIBRARY gtest-main)
    add_library(${GTEST_MAIN_LIBRARY} UNKNOWN IMPORTED)
    set_property(TARGET ${GTEST_MAIN_LIBRARY} PROPERTY IMPORTED_LOCATION ${GTEST_MAIN_LIBRARY_PATH})
    add_dependencies(${GTEST_MAIN_LIBRARY} GTest)

    set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})
    set(GTEST_INCLUDE_DIRS ${GTEST_INSTALL_DIR}/include)
endif()

add_subdirectory(ade)
add_subdirectory(common)
