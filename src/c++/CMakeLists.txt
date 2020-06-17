cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(Purdue-PHYS-580)

include_directories(include
        ../../CLibraries/)

add_executable(Purdue-PHYS-580
        src/main.cpp
        include/Problem3_17.h
        include/Problem3_18.h)

install(TARGETS Purdue-PHYS-580
        CONFIGURATIONS Debug
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/build/debug/bin)
install(TARGETS Purdue-PHYS-580
        CONFIGURATIONS Release
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/build/release/bin)