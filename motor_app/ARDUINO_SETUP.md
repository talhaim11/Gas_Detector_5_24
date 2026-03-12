# Arduino Setup Guide

## Upload the Code

1. **Open Arduino IDE**
   - Download from: https://www.arduino.cc/en/software

2. **Load the code**
   - Open `arduino_code_mount.ino` in Arduino IDE

3. **Pin Configuration (ALREADY SET FOR YOUR HARDWARE)**
   - Motor 1: STEP=Pin 2, DIR=Pin 3
   - Motor 2: STEP=Pin 7, DIR=Pin 8
   - Emergency Stop: Pin 4
   - Sensor: Pin A0
   - ✅ No changes needed - matches your existing wiring!

4. **Select your Arduino board**
   - Tools → Board → Select your Arduino model (e.g., Arduino Uno, Mega, Nano)

5. **Select COM port**
   - Tools → Port → Select the port your Arduino is connected to
   - Should match the port in your motor_app config

6. **Upload**
   - Click Upload button (→) or Ctrl+U
   - Wait for "Done uploading" message

7. **Test with Serial Monitor**
   - Tools → Serial Monitor
   - Set baud rate to **9600**
   - Should see "READY" message
   - Type `R1 100` and press Enter → motor should move right
   - Type `R1 0` and press Enter → motor should stop immediately
   - Test emergency stop button → should halt all movement

## Troubleshooting

### Motor doesn't move
- Check wiring connections (STEP and DIR pins)
- Check motor power supply (separate from Arduino)
- Verify motor driver is powered and configured
- Check that emergency stop button is not pressed (Pin 4 should be HIGH)

### Motor moves wrong direction
- Swap direction logic in code (change HIGH/LOW in `processCommand`)
- Or physically swap motor wiring

### Serial communication fails
- Check baud rate matches (9600)
- Check COM port matches Python app
- Try unplugging/replugging USB
- Look for "READY" message in Serial Monitor

### Motor stutters or misses steps
- Adjust `currentSpeedDelay` (default is 2500µs from your old code)
- For faster: reduce delay (minimum 50µs)
- For more reliability: increase delay (up to 5000µs)
- Check motor current settings (VREF on driver)
- Check power supply voltage/current

### Emergency stop doesn't work
- Check Pin 4 connection and INPUT_PULLUP
- When pressed, pin should read LOW
- Test with multimeter or Serial.println in code

## Key Features of This Code

1. **Immediate response**: Checks serial every 1ms when idle
2. **Interruptible**: Can stop mid-movement with "X1 0" command
3. **No command buffering**: Only processes one command at a time
4. **Simple protocol**: Just "R1 100", "L1 0", etc.
5. **Emergency stop**: Hardware button (Pin 4) immediately halts all movement
6. **Speed matched**: Uses 2500µs delay matching your old code's medium speed

## Command Reference

| Command | Action |
|---------|--------|
| `R1 <steps>` | Move axis 1 right |
| `L1 <steps>` | Move axis 1 left |
| `U1 <steps>` | Move axis 2 up |
| `D1 <steps>` | Move axis 2 down |
| `*1 0` | Stop (any direction + 0 steps) |

Where `<steps>` = number of motor steps (200 steps = 1 revolution for typical stepper)
