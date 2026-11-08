#include "led_logic.h"
#include "pico/stdlib.h"

#ifndef PICO_DEFAULT_LED_PIN
#error "Select a Pico board that defines PICO_DEFAULT_LED_PIN"
#endif

int main(void) {
    int32_t tick = 0;

    stdio_init_all();
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);

    while (true) {
        gpio_put(PICO_DEFAULT_LED_PIN, led_on_for_tick(tick));
        ++tick;
        sleep_ms(250);
    }
}
