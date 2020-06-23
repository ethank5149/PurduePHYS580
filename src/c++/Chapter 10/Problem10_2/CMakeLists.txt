cmake_minimum_required(VERSION 3.16)
project(Problem10_2)

set(CMAKE_CXX_STANDARD 20)

include_directories(include
        ../../CLibraries/)

add_executable(Problem10_2
        src/main.cpp
        include/initialize.h include/norm.h include/iterate.h include/store.h include/run.h)