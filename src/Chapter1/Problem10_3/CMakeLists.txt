cmake_minimum_required(VERSION 3.16)
project(Problem10_3)

set(CMAKE_CXX_STANDARD 20)

include_directories(include
        ../../CLibraries/)

add_executable(Problem10_3 src/main.cpp include/initialize.h include/iterate.h include/norm.h include/run.h include/store.h)