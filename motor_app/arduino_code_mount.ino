/*
 * Motor Control Arduino Code for Gas Detector Project
 * Compatible with motor_app Python application
 * 
 * Commands from Python (via Serial):
 *   PING       - Reply "PONG" (connection check)
 *   SPEED <us> - Set step delay in microseconds (50-5000). Higher = slower.
 *   R1 <steps> - Move axis 1 right by <steps> (0 = stop)
 *   L1 <steps> - Move axis 1 left by <steps> (0 = stop)
 *   U2 <steps> - Move axis 2 up by <steps> (0 = stop)
 *   D2 <steps> - Move axis 2 down by <steps> (0 = stop)
 * 
 * Version: 2.0 - Configured for your hardware
 * Date: February 2026
 * 
 * Pin Configuration:
 *   Motor 1: STEP=Pin 2, DIR=Pin 3
 *   Motor 2: STEP=Pin 7, DIR=Pin 8
 *   Emergency Stop: Pin 4
 *   Sensor: Pin A0
 */

// Motor driver pins - MATCHED TO YOUR HARDWARE
// Axis 1 (Left/Right motor)
const int AXIS1_STEP_PIN = 2;  // PUL1
const int AXIS1_DIR_PIN = 3;   // DIR1

// Axis 2 (Up/Down motor)
const int AXIS2_STEP_PIN = 7;  // PUL2
const int AXIS2_DIR_PIN = 8;   // DIR2

// Additional pins
const int EMERGENCY_STOP_PIN = 4;
const int SENSOR_PIN = A0;

// Motor parameters (matched to your old code)
const int STEPS_PER_REV = 25000;       // Steps per revolution (from your old code)
const int DEFAULT_SPEED_DELAY = 2500;  // Microseconds between steps (2500us = medium speed from old code)
const int MIN_SPEED_DELAY = 50;        // Fastest (50us for high speed)
const int MAX_SPEED_DELAY = 5000;      // Slowest (5000us for low speed)

// Current state
volatile bool axis1Moving = false;
volatile bool axis2Moving = false;
volatile int axis1StepsRemaining = 0;
volatile int axis2StepsRemaining = 0;
volatile int currentSpeedDelay = DEFAULT_SPEED_DELAY;

// Serial buffer
String commandBuffer = "";

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Setup motor pins
  pinMode(AXIS1_STEP_PIN, OUTPUT);
  pinMode(AXIS1_DIR_PIN, OUTPUT);
  pinMode(AXIS2_STEP_PIN, OUTPUT);
  pinMode(AXIS2_DIR_PIN, OUTPUT);
  
  // Setup emergency stop and sensor
  pinMode(EMERGENCY_STOP_PIN, INPUT_PULLUP);
  pinMode(SENSOR_PIN, INPUT);
  
  // Initial state
  digitalWrite(AXIS1_STEP_PIN, LOW);
  digitalWrite(AXIS2_STEP_PIN, LOW);
  
  Serial.println("READY");
}

void loop() {
  // Check emergency stop first
  if (digitalRead(EMERGENCY_STOP_PIN) == LOW) {
    axis1StepsRemaining = 0;
    axis2StepsRemaining = 0;
    axis1Moving = false;
    axis2Moving = false;
    return;  // Don't process anything else
  }
  
  // Check for incoming serial commands
  checkSerialCommands();
  
  // Execute movement for axis 1
  if (axis1StepsRemaining > 0) {
    // Check emergency stop during movement
    if (digitalRead(EMERGENCY_STOP_PIN) == LOW) {
      axis1StepsRemaining = 0;
      axis1Moving = false;
      return;
    }
    
    digitalWrite(AXIS1_STEP_PIN, HIGH);
    delayMicroseconds(50);
    digitalWrite(AXIS1_STEP_PIN, LOW);
    delayMicroseconds(currentSpeedDelay);
    axis1StepsRemaining--;
    
    if (axis1StepsRemaining == 0) {
      axis1Moving = false;
    }
  }
  
  // Execute movement for axis 2
  if (axis2StepsRemaining > 0) {
    // Check emergency stop during movement
    if (digitalRead(EMERGENCY_STOP_PIN) == LOW) {
      axis2StepsRemaining = 0;
      axis2Moving = false;
      return;
    }
    
    digitalWrite(AXIS2_STEP_PIN, HIGH);
    delayMicroseconds(50);
    digitalWrite(AXIS2_STEP_PIN, LOW);
    delayMicroseconds(currentSpeedDelay);
    axis2StepsRemaining--;
    
    if (axis2StepsRemaining == 0) {
      axis2Moving = false;
    }
  }
  
  // Small delay when not moving to check serial more frequently
  if (axis1StepsRemaining == 0 && axis2StepsRemaining == 0) {
    delay(1);  // Check serial every 1ms when idle
  }
}

void checkSerialCommands() {
  // Read incoming serial data
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    
    if (inChar == '\n') {
      // Process complete command
      processCommand(commandBuffer);
      commandBuffer = "";
    } else {
      commandBuffer += inChar;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  
  // PING: reply PONG for connection check from Python
  if (cmd.equalsIgnoreCase("PING")) {
    Serial.println("PONG");
    return;
  }
  
  if (cmd.length() < 2) return;
  
  // SPEED <delay_us>: set step pulse delay in microseconds (50-5000). Higher = slower movement.
  if (cmd.startsWith("SPEED")) {
    int spaceIndex = cmd.indexOf(' ');
    if (spaceIndex != -1) {
      int newDelay = cmd.substring(spaceIndex + 1).toInt();
      currentSpeedDelay = constrain(newDelay, MIN_SPEED_DELAY, MAX_SPEED_DELAY);
    }
    return;
  }
  
  // Parse command format: "R1 100" or "L1 0"
  char direction = cmd.charAt(0);  // R, L, U, or D
  char axis = cmd.charAt(1);        // 1 or 2
  
  // Find space and extract steps
  int spaceIndex = cmd.indexOf(' ');
  if (spaceIndex == -1) return;
  
  int steps = cmd.substring(spaceIndex + 1).toInt();
  
  // Process command based on direction and axis
  if (axis == '1') {
    // Axis 1 (Left/Right)
    if (steps == 0) {
      // Stop command
      axis1StepsRemaining = 0;
      axis1Moving = false;
    } else {
      // Move command
      if (direction == 'R') {
        digitalWrite(AXIS1_DIR_PIN, HIGH);  // Right direction
      } else if (direction == 'L') {
        digitalWrite(AXIS1_DIR_PIN, LOW);   // Left direction
      } else {
        return;  // Invalid direction
      }
      
      // Set steps and start moving
      axis1StepsRemaining = steps;
      axis1Moving = true;
    }
  } else if (axis == '2') {
    // Axis 2 (Up/Down)
    if (steps == 0) {
      // Stop command
      axis2StepsRemaining = 0;
      axis2Moving = false;
    } else {
      // Move command
      if (direction == 'U') {
        digitalWrite(AXIS2_DIR_PIN, HIGH);  // Up direction
      } else if (direction == 'D') {
        digitalWrite(AXIS2_DIR_PIN, LOW);   // Down direction
      } else {
        return;  // Invalid direction
      }
      
      // Set steps and start moving
      axis2StepsRemaining = steps;
      axis2Moving = true;
    }
  }
}

/*
 * HARDWARE SETUP NOTES:
 * 
 * PIN CONFIGURATION (matched to your existing hardware):
 *    Motor 1 (Axis 1 - Left/Right):
 *      - STEP (PUL1): Pin 2
 *      - DIR (DIR1): Pin 3
 * 
 *    Motor 2 (Axis 2 - Up/Down):
 *      - STEP (PUL2): Pin 7
 *      - DIR (DIR2): Pin 8
 * 
 *    Emergency Stop: Pin 4 (INPUT_PULLUP)
 *    Sensor: Pin A0 (analog input)
 * 
 * MOTOR DRIVER CONNECTIONS:
 *    - Connect STEP pins to PUL/STEP on driver
 *    - Connect DIR pins to DIR on driver
 *    - Motors need separate power supply (12-24V typical)
 *    - Arduino USB power is only for logic
 *    - Connect all grounds together
 * 
 * SPEED SETTINGS:
 *    - DEFAULT: 2500 µs (medium speed, matched to old code)
 *    - HIGH: 50 µs (fast movement)
 *    - LOW: 5000 µs (slow, careful movement)
 * 
 * TEST WITH ARDUINO SERIAL MONITOR:
 *    - Baud: 9600
 *    - Send: "PING" (receive "PONG")
 *    - Send: "SPEED 1000" (set step delay to 1000 us)
 *    - Send: "R1 200" (move right 200 steps)
 *    - Send: "R1 0" (stop immediately)
 *    - Send: "L1 200" / "U2 50" / "D2 50" (axis 1 left, axis 2 up/down)
 *    - Emergency stop button will halt all movement
 */
