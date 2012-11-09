cmake_minimum_required (VERSION 2.6)
project(ceres_test)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMakeModules/")

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Protobuf REQUIRED)

SET(CERES_TEST_SRC_DIR src)
SET(LIB_NAMES ceres glog cxsparse cholmod protobuf)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

FOREACH(lib ${LIB_NAMES})
  FIND_LIBRARY(LIB_${lib} ${lib})
  IF (LIB_${lib})
    MESSAGE(STATUS "found library '${lib}': ${LIB_${lib}}")
  ELSE (LIB_${lib})
    MESSAGE(FATAL_ERROR "cannot find library '${lib}' - this will not work ...")
  ENDIF (LIB_${lib})
  LIST(APPEND LIBS ${LIB_${lib}})
ENDFOREACH(lib)

include_directories(${CERES_TEST_SRC_DIR} ${OpenCV_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${PROTOBU_INCLUDE_DIRS})

FILE(GLOB CERES_TEST_SRCS "src/*.cpp")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
add_executable(ceres_test ${CERES_TEST_SRCS})
target_link_libraries(ceres_test ${LIB_NAMES} ${OpenCV_LIBS})
