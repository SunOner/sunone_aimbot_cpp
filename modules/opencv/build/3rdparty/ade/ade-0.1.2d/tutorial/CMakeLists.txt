# Copyright (C) 2018 Intel Corporation
#
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 2.8)

project(ade_samples)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(CMAKE_VERSION VERSION_LESS "3.1" AND CMAKE_COMPILER_IS_GNUCXX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
      set(CMAKE_CXX_STANDARD 11)
    endif()
    find_package(ade REQUIRED)
endif()

function(defstep name)
  add_executable(${name} ${name}.cpp)
  add_security_flags( ${name} )
  target_link_libraries(${name} ade)
  install(FILES ${name}.cpp DESTINATION "examples/ade" COMPONENT example)
endfunction(defstep)

defstep(01_hello)
defstep(02_add_link_remove)
defstep(03_meta)
defstep(04_passes)
defstep(05_backend)
defstep(06_hierarchy)

install(FILES README.md CMakeLists.txt DESTINATION "examples/ade" COMPONENT example)
