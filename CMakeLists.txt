# Copyright (c) 2018 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 2.8)

# Common to all tutorial sections
include(CMakeLists_common.cmake)

set(TARGET_NAME "smart_city_tutorial")

option(ENABLED_DB  "Send meta-data to database for plotting" OFF)

find_package(OpenCV)
if(OpenCV_FOUND)
    message(STATUS "OPENCV is enabled")
    message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
else()
    message(STATUS "OPENCV is disabled or not found, " ${TARGET_NAME} " skiped")
    return()
endif()

find_package(Boost COMPONENTS log system filesystem REQUIRED)
if ( ENABLED_DB )
    find_package(libmongocxx)
endif()

if( BUILD_SAMPLE_NAME AND NOT ${BUILD_SAMPLE_NAME} STREQUAL ${TARGET_NAME} )
    message(STATUS "SAMPLE ${TARGET_NAME} SKIPPED")
    return()
endif()

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

include_directories (${OpenCV_INCLUDE_DIRS})
include_directories (${Boost_INCLUDE_DIRS})
include_directories (${LIBMONGOCXX_INCLUDE_DIRS})

link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

add_dependencies(${TARGET_NAME} gflags)
add_definitions(-DBOOST_LOG_DYN_LINK)
add_definitions(-DINTEL_CVSDK_VER=$ENV{INTEL_CVSDK_VER})
if( ENABLED_DB )
  add_definitions(-DENABLED_DB)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
COMPILE_PDB_NAME ${TARGET_NAME})


target_link_libraries(${TARGET_NAME} format_reader IE::ie_cpu_extension ${IE_LIBRARIES} ${InferenceEngine_LIBRARIES} gflags)

if(UNIX)
    target_link_libraries( ${TARGET_NAME} ${LIB_DL} pthread ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${LIBMONGOCXX_LIBRARIES})
endif()
