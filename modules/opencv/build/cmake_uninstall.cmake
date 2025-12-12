# -----------------------------------------------
# File that provides "make uninstall" target
#  We use the file 'install_manifest.txt'
#
# Details: https://gitlab.kitware.com/cmake/community/-/wikis/FAQ#can-i-do-make-uninstall-with-cmake
# -----------------------------------------------

if(NOT EXISTS "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: \"C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install_manifest.txt\"")
endif()

file(READ "C:/Users/therouxe/source/repos/MacroMan5/macroman_aimbot_cpp/sunone_aimbot_cpp/modules/opencv/build/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
foreach(file ${files})
  message(STATUS "Uninstalling $ENV{DESTDIR}${file}")
  if(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    exec_program(
        "C:/Program Files/Microsoft Visual Studio/18/Insiders/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
        OUTPUT_VARIABLE rm_out
        RETURN_VALUE rm_retval
    )
    if(NOT "${rm_retval}" STREQUAL 0)
      message(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${file}")
    endif()
  else(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    message(STATUS "File $ENV{DESTDIR}${file} does not exist.")
  endif()
endforeach()
