cmake_minimum_required(VERSION 3.10)

project(lygia)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(LYGIA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(LYGIA_HEADER "${LYGIA_GENERATED_DIR}/lygia.h")
set(LYGIA_SOURCE "${LYGIA_GENERATED_DIR}/lygia.cpp")
set(LYGIA_BUNDLE_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/bundle.py")

# Glob all GLSL files to ensure rebuilds when they change
file(GLOB_RECURSE GLSL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.glsl")

# Create the generated directory
file(MAKE_DIRECTORY ${LYGIA_GENERATED_DIR})

add_custom_command(
    OUTPUT ${LYGIA_HEADER} ${LYGIA_SOURCE}
    COMMAND ${Python3_EXECUTABLE} "${LYGIA_BUNDLE_SCRIPT}" "${CMAKE_CURRENT_SOURCE_DIR}" "${LYGIA_GENERATED_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS "${LYGIA_BUNDLE_SCRIPT}" ${GLSL_FILES}
    COMMENT "Bundling lygia shaders"
)

add_library(lygia STATIC ${LYGIA_SOURCE} ${LYGIA_HEADER})

target_include_directories(lygia PUBLIC "${LYGIA_GENERATED_DIR}")
