# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/root/mycode/tlp-main/cmake/libs/../../3rdparty/libbacktrace"
  "/root/mycode/tlp-main/build_nollvm/libbacktrace"
  "/root/mycode/tlp-main/build_nollvm/libbacktrace"
  "/root/mycode/tlp-main/build_nollvm/libbacktrace/tmp"
  "/root/mycode/tlp-main/build_nollvm/libbacktrace/src/project_libbacktrace-stamp"
  "/root/mycode/tlp-main/build_nollvm/libbacktrace/src"
  "/root/mycode/tlp-main/build_nollvm/libbacktrace/src/project_libbacktrace-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/root/mycode/tlp-main/build_nollvm/libbacktrace/src/project_libbacktrace-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/root/mycode/tlp-main/build_nollvm/libbacktrace/src/project_libbacktrace-stamp${cfgdir}") # cfgdir has leading slash
endif()
