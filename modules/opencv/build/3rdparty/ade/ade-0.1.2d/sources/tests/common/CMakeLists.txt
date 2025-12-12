# Copyright (C) 2018 Intel Corporation
#
#
# SPDX-License-Identifier: Apache-2.0
#

project( common-tests )

include_directories(${GTEST_INCLUDE_DIRS})

# Required by gtest
find_package(Threads REQUIRED)

file( GLOB sources *.cpp )
file( GLOB include *.hpp )

add_executable(${PROJECT_NAME} ${include} ${sources})

add_security_flags( ${PROJECT_NAME} )

target_link_libraries(${PROJECT_NAME} ${GTEST_BOTH_LIBRARIES} Threads::Threads ade)

add_test(NAME CommonTests COMMAND ${PROJECT_NAME} --gtest_output=xml:${CMAKE_BINARY_DIR}/${PROJECT_NAME}_test.xml)
