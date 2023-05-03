cmake_minimum_required(VERSION 3.14)
set(CMAKE_VERBOSE_MAKEFILE OFF CACHE BOOL "Explicit Makefile")
set(SANITIZE OFF CACHE BOOL "Build with sanitizers")

project(fea
	VERSION 1.0
	DESCRIPTION "Fast implementation of Artificial Evolution"
	LANGUAGES CXX)

# Build types
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release")
endif()

if (NOT(CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message(WARNING "Please prefer llvm clang compiler when running debug builds. Use '-D CMAKE_CXX_COMPILER=clang++'")
	endif()
elseif(NOT (CMAKE_CXX_COMPILER MATCHES "clang\\+\\+"))
	message(WARNING "Using clang instead of clang++ can cause link errors with sanitizers, please use clang++ instead.")
endif()

#setting includes and sources
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
file(GLOB SOURCES "${SRC_DIR}/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "[^/]+/main.cpp")

add_executable(Fea "${SRC_DIR}/main.cpp")
add_library(ea ${SOURCES})
target_include_directories(ea PUBLIC ${INCLUDE_DIR})

target_link_libraries(Fea PUBLIC ea)

#Setting C++ flags
target_compile_features(ea PUBLIC cxx_std_23)
target_compile_options(ea PUBLIC
	$<$<CXX_COMPILER_ID:MSVC>:/W4>
	$<$<CXX_COMPILER_ID:Clang,GNU>:-Wall -Wextra -Wpedantic>
	#$<$<CXX_COMPILER_ID:Clang>:-stdlib=libc++>
	$<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2>
	$<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3 -march=native -g>
	$<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:Clang>>:-Ofast>
	$<$<CONFIG:Debug>:-g>
	$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>,$<BOOL:${SANITIZE}>>:-fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize=vptr>
	$<$<AND:$<BOOL:${COVERAGE}>,$<CXX_COMPILER_ID:GNU>>:-ftest-coverage -fprofile-arcs>
	$<$<AND:$<BOOL:${COVERAGE}>,$<CXX_COMPILER_ID:Clang>>:-fprofile-instr-generate -fcoverage-mapping>
	)

target_link_options(ea PUBLIC
	$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>,$<BOOL:${SANITIZE}>>:-fsanitize=address -fsanitize=undefined -fno-sanitize=vptr>
	$<$<AND:$<BOOL:${COVERAGE}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fprofile-instr-generate>
	#$<$<CXX_COMPILER_ID:Clang>:-fuse-ld=lld>
	)

find_package(TBB REQUIRED)
target_link_libraries(ea PUBLIC TBB::tbb)
