cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED ENV{PICO_SDK_PATH})
    message(FATAL_ERROR "Set PICO_SDK_PATH to a configured Raspberry Pi Pico SDK checkout")
endif()

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(aether_pico_led C CXX ASM)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

pico_sdk_init()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(AETHER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(AETHER_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(AETHER_GENERATED_C "${AETHER_GENERATED_DIR}/led_logic.c")
set(AETHER_GENERATED_H "${AETHER_GENERATED_DIR}/led_logic.h")

add_custom_command(
    OUTPUT "${AETHER_GENERATED_C}" "${AETHER_GENERATED_H}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_LIST_DIR}/generate.py"
            --output-dir "${AETHER_GENERATED_DIR}"
    DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/generate.py"
        "${CMAKE_CURRENT_LIST_DIR}/led_logic.hl"
    WORKING_DIRECTORY "${AETHER_ROOT}"
    VERBATIM
)

add_executable(aether_pico_led
    main.c
    "${AETHER_GENERATED_C}"
)
target_include_directories(aether_pico_led PRIVATE "${AETHER_GENERATED_DIR}")
target_link_libraries(aether_pico_led PRIVATE pico_stdlib)
pico_enable_stdio_usb(aether_pico_led 0)
pico_enable_stdio_uart(aether_pico_led 1)
pico_add_extra_outputs(aether_pico_led)
