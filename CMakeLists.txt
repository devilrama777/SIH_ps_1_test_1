cmake_minimum_required(VERSION 3.14)
project(AuraEdgeActivator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(SOURCES
    src/main.cpp
    src/mfcc.cpp
    src/kws_engine.cpp
    src/telemetry.cpp
)

add_executable(aura ${SOURCES})

if(WIN32)
    target_link_libraries(aura PRIVATE ole32 winmm psapi)
    if(MINGW)
        target_link_options(aura PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
elseif(APPLE)
    find_library(CORE_AUDIO CoreAudio REQUIRED)
    find_library(AUDIO_TOOLBOX AudioToolbox REQUIRED)
    find_library(CORE_FOUNDATION CoreFoundation REQUIRED)
    target_link_libraries(aura PRIVATE
        ${CORE_AUDIO}
        ${AUDIO_TOOLBOX}
        ${CORE_FOUNDATION}
        pthread
        m
    )
elseif(UNIX)
    target_link_libraries(aura PRIVATE pthread dl m)
endif()
