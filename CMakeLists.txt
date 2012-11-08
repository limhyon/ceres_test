cmake_minimum_required (VERSION 2.6)
project(ceres_test)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMakeModules/")

FIND_PACKAGE(Eigen3 REQUIRED)

SET(CERES_TEST_SRC_DIR src)
SET(LIB_NAMES ceres)

FOREACH(lib ${LIB_NAMES})
  FIND_LIBRARY(LIB_${lib} ${lib})
  IF (LIB_${lib})
    MESSAGE(STATUS "found library '${lib}': ${LIB_${lib}}")
  ELSE (LIB_${lib})
    MESSAGE(FATAL_ERROR "cannot find library '${lib}' - this will not work ...")
  ENDIF (LIB_${lib})
  LIST(APPEND LIBS ${LIB_${lib}})
ENDFOREACH(lib)

include_directories(${CERES_TEST_SRC_DIR} ${EIGEN3_INCLUDE_DIR})

FILE(GLOB CERES_TEST_SRCS "src/*.cpp")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
add_executable(ceres_test ${CERES_TEST_SRCS})
target_link_libraries(ceres_test ${LIB_NAMES})
