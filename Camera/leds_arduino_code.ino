#include <Adafruit_DotStar.h>
#include <SPI.h>

#define NUM_LEDS 11            // Number of LEDs in the strip
#define DATAPIN    11          // MOSI pin for SPI communication
#define CLOCKPIN   13          // Clock pin for SPI communication
#define BUTTON_PIN 7           // Pin connected to the button

Adafruit_DotStar strip = Adafruit_DotStar(NUM_LEDS, DATAPIN, CLOCKPIN, DOTSTAR_BRG);

// Variables for blinking
bool isOn = false;
bool isBlinking = true;        // Start with blinking enabled
unsigned long previousMillis = 0;
unsigned long interval = 1000 / (1 * 2);  // Blink rate for 3Hz
uint32_t color = strip.Color(255, 0, 255);  // Red color

void setup() {
  strip.begin();     // Initialize the strip
  strip.show();      // Turn off all LEDs initially

  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Set up button with internal pull-up resistor
}

void loop() {
  checkButton();  // Check if button is pressed and toggle blinking

  if (isBlinking) {
    unsigned long currentMillis = millis();

    // Toggle LEDs on and off every 'interval'
    if (currentMillis - previousMillis >= interval) {
      previousMillis = currentMillis;

      if (isOn) {
        strip.clear();  // Turn off all LEDs
      } else {
        strip.fill(color, 0, NUM_LEDS);  // Set all LEDs to the chosen color
      }
      strip.show();  // Update the strip with new values
      isOn = !isOn;  // Toggle the state
    }
  }
}

// Function to change color dynamically
void changeColor(uint8_t r, uint8_t g, uint8_t b) {
  color = strip.Color(r, g, b);
}

// Function to check the button state and toggle blinking
void checkButton() {
  static bool lastButtonState = HIGH;  // Tracks the previous button state
  bool currentButtonState = digitalRead(BUTTON_PIN);

  // If the button is pressed (low state because of pull-up) and was previously released
  if (currentButtonState == LOW && lastButtonState == HIGH) {
    delay(50);  // Debouncing delay
    if (digitalRead(BUTTON_PIN) == LOW) {  // Check again to confirm press
      isBlinking = !isBlinking;  // Toggle the blinking state
    }
  }
  lastButtonState = currentButtonState;
}
