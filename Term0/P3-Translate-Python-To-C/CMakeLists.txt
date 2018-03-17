cmake_minimum_required(VERSION 3.9)
project(fuckCPP)

set(CMAKE_CXX_STANDARD 14)

#add_executable(fuckCPP debugging_helpers.cpp helpers.cpp localizer.cpp tests.cpp simulate.cpp)
add_executable(fuckCPP tests.cpp)