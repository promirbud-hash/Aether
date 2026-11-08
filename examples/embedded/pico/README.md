# Aether on Raspberry Pi Pico

This minimal Raspberry Pi Pico SDK project keeps the board-specific startup,
linker script, GPIO driver, and flashing support in the SDK. Aether supplies
only `led_on_for_tick`, generated as freestanding C99 and exported through a
small C header.

Requirements:

- A configured Raspberry Pi Pico SDK checkout.
- The Pico SDK's ARM embedded toolchain.
- CMake, Ninja (or another supported generator), and Python 3.10+.
- `PICO_SDK_PATH` set to the SDK checkout.

From this directory, configure and build with:

```powershell
cmake -S . -B build -G Ninja
cmake --build build
```

The generated C and header are written into the build directory. The checked-in
`.hl` remains the source of truth. The example uses the SDK's default trap
behavior; production firmware should override `AETHER_TRAP()` with its chosen
fault policy and separately configure board, clocks, startup, and deployment.

The generated freestanding Aether logic has been compiled as an object with
LLVM Clang 21.1.7 for ARM Cortex-M0+, ARM Cortex-M4, RISC-V32
(`rv32imac`/`ilp32`), and AVR ATmega328P. The repository's host test suite also
compiles and runs the logic with a host C99 compiler. These checks do not link
or run a Pico SDK firmware image; that requires the SDK, its toolchain, and
board-specific configuration.
