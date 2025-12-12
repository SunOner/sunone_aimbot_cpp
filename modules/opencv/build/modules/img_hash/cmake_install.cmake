# Install script for directory: C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules/img_hash

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x64/vc17/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/lib/Debug/opencv_img_hash4100d.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x64/vc17/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/lib/Release/opencv_img_hash4100.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libs" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x64/vc17/bin" TYPE SHARED_LIBRARY OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/bin/Debug/opencv_img_hash4100d.dll")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x64/vc17/bin" TYPE SHARED_LIBRARY OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/bin/Release/opencv_img_hash4100.dll")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "pdb")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x64/vc17/bin" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/bin/Debug/opencv_img_hash4100d.pdb")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x64/vc17/bin" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/bin/Release/opencv_img_hash4100.pdb")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules/img_hash/include/opencv2/img_hash.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/img_hash" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules/img_hash/include/opencv2/img_hash/average_hash.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/img_hash" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules/img_hash/include/opencv2/img_hash/block_mean_hash.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/img_hash" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules/img_hash/include/opencv2/img_hash/color_moment_hash.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/img_hash" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules/img_hash/include/opencv2/img_hash/img_hash_base.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/img_hash" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules/img_hash/include/opencv2/img_hash/marr_hildreth_hash.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/img_hash" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules/img_hash/include/opencv2/img_hash/phash.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2/img_hash" TYPE FILE OPTIONAL FILES "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/opencv_contrib-4.10.0/modules/img_hash/include/opencv2/img_hash/radial_variance_hash.hpp")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/modules/img_hash/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
