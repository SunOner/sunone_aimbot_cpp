# Copyright (C) 2018 Intel Corporation
#
#
# SPDX-License-Identifier: Apache-2.0
#

include(manifest.cmake)
project(${THE_PROJECT_NAME} VERSION ${THE_PROJECT_VERSION} LANGUAGES CXX)

file( GLOB_RECURSE sources source/*.cpp )
file( GLOB_RECURSE include *.hpp )

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wconversion -Wshadow -Wno-error=cpp -Wformat -Wformat-security" )
endif()

add_library( ${PROJECT_NAME} STATIC ${include} ${sources} )
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
set_target_properties( ${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE True)

add_security_flags( ${PROJECT_NAME} )

if(FORCE_ADE_ASSERTS)
    target_compile_definitions( ${PROJECT_NAME} PUBLIC FORCE_ADE_ASSERTS )
endif()

if(BUILD_ADE_DOCUMENTATION)
    find_package(Doxygen REQUIRED)

    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    set(DOXYGEN_PROJECT_NAME ${THE_PROJECT_NAME})
    set(DOXYGEN_PROJECT_VERSION ${THE_PROJECT_VERSION})
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/doc)
    set(DOXYGEN_INPUT "${CMAKE_SOURCE_DIR}/README.md ${CMAKE_CURRENT_SOURCE_DIR}/include")
    set(DOXYGEN_EXCLUDE_SYMBOLS "detail details")
    set(DOXYGEN_README_PATH ${CMAKE_SOURCE_DIR}/README.md)
    set(DOXYGEN_STRIP_FROM_PATH ${CMAKE_SOURCE_DIR})

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
                   ${doxyfile} @ONLY)

    add_custom_target(doc ALL
                      COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc
                      COMMENT "Generating API documentation with Doxygen"
                      VERBATIM)
endif()

include(GNUInstallDirs)

install(TARGETS ade COMPONENT dev
        EXPORT adeTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT adeTargets
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/ade"
        COMPONENT dev)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT dev
        FILES_MATCHING PATTERN "*.hpp")

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
        "${CMAKE_BINARY_DIR}/adeConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY ExactVersion)

configure_package_config_file(
        "${CMAKE_CURRENT_LIST_DIR}/cmake/adeConfig.cmake.in"
        "${CMAKE_BINARY_DIR}/adeConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/ade"
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO)

export(TARGETS ade FILE "${CMAKE_BINARY_DIR}/adeTargets.cmake")

install(FILES "${CMAKE_BINARY_DIR}/adeConfig.cmake"
        "${CMAKE_BINARY_DIR}/adeConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/ade" COMPONENT dev)
