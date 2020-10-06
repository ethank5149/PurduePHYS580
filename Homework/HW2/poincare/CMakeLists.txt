cmake_minimum_required(VERSION 3.17)
project(poincare)

set(CMAKE_CXX_STANDARD 20)
set(DCMAKE_TOOLCHAIN_FILE C:/vcpkg/scripts/buildsystems/vcpkg.cmake)
include_directories(include)

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.73 COMPONENTS program_options REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xtl CONFIG REQUIRED)

IF (Boost_FOUND)
    MESSAGE("Boost Found! :)")
ELSE ()
    MESSAGE("Boost Not Found... :(")
ENDIF()

add_executable(poincare
        include/poincare.hpp
        main.cpp)

target_link_libraries(poincare PUBLIC Boost::program_options)
target_link_libraries(poincare PRIVATE xtensor)
target_link_libraries(poincare PRIVATE xtl)

IF(Boost_PROGRAM_OPTIONS_FOUND)
    MESSAGE("'program_options' Found! :)")
ELSE()
    MESSAGE("'program_options' Not Found... :(")
ENDIF()