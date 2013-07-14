
cmake_minimum_required(VERSION 2.8)

project(vibe C)

set(SOURCES
	"vibe.c"
)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB_PKG REQUIRED glib-2.0)
include_directories(${OpenCV_INCLUDE_DIR} ${GLIB_PKG_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE STREQUAL "")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(CMAKE_BUILD_TYPE STREQUAL "")

set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -fomit-frame-pointer -O3")

add_executable(vibe ${SOURCES})
target_link_libraries(vibe ${OpenCV_LIBS} ${GLIB_PKG_LIBRARIES})

