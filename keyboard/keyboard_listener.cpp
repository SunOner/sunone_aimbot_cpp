#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#include "config.h"
#include "SerialConnection.h"
#include "keyboard_listener.h"
#include "mouse.h"
#include "keycodes.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "KmboxNetConnection.h"

extern std::atomic<bool> shouldExit;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;
extern std::atomic<bool> detectionPaused;

extern MouseThread* globalMouseThread;

const float OFFSET_STEP = 0.01f;
const float NORECOIL_STEP = 5.0f;

// Arrow key vectors
const std::vector<std::string> upArrowKeys = { "UpArrow" };
const std::vector<std::string> downArrowKeys = { "DownArrow" };
const std::vector<std::string> leftArrowKeys = { "LeftArrow" };
const std::vector<std::string> rightArrowKeys = { "RightArrow" };
const std::vector<std::string> shiftKeys = { "LeftShift", "RightShift" };

// Previous key states
bool prevUpArrow = false;
bool prevDownArrow = false;
bool prevLeftArrow = false;
bool prevRightArrow = false;

bool isAnyKeyPressed(const std::vector<std::string>& keys)
{
    for (const auto& key_name : keys)
    {
        int key_code = KeyCodes::getKeyCode(key_name);

        bool pressed = false;

        // kmbox net
        if (kmboxNetSerial && kmboxNetSerial->isOpen())
        {
            if  (key_name == "LeftMouseButton")      pressed = kmboxNetSerial->monitorMouseLeft()   == 1;
            else if(key_name == "RightMouseButton")  pressed = kmboxNetSerial->monitorMouseRight()  == 1;
            else if(key_name == "MiddleMouseButton") pressed = kmboxNetSerial->monitorMouseMiddle() == 1;
            else if(key_name == "X1MouseButton")     pressed = kmboxNetSerial->monitorMouseSide1()  == 1;
            else if(key_name == "X2MouseButton")     pressed = kmboxNetSerial->monitorMouseSide2()  == 1;
        }

        // local mouse
        if (!pressed && key_code != -1 && (GetAsyncKeyState(key_code) & 0x8000))
            pressed = true;

        if (pressed) return true;
    }
    return false;
}

void keyboardListener()
{
    while (!shouldExit)
    {
        // Aiming
        if (!config.auto_aim)
        {
            aiming = isAnyKeyPressed(config.button_targeting) ||
                (config.arduino_enable_keys && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->aiming_active) ||
                (kmboxSerial && kmboxSerial->isOpen() && kmboxSerial->aiming_active) ||
                (kmboxNetSerial && kmboxNetSerial->isOpen() && kmboxNetSerial->aiming_active);
        }
        else
        {
            aiming = true;
        }

        // Shooting
        shooting = isAnyKeyPressed(config.button_shoot) ||
            (config.arduino_enable_keys && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->shooting_active) ||
            (kmboxSerial && kmboxSerial->isOpen() && kmboxSerial->shooting_active) ||
            (kmboxNetSerial && kmboxNetSerial->isOpen() && kmboxNetSerial->shooting_active);

        // Zooming
        zooming = isAnyKeyPressed(config.button_zoom) ||
            (config.arduino_enable_keys && arduinoSerial && arduinoSerial->isOpen() && arduinoSerial->zooming_active) ||
            (kmboxSerial && kmboxSerial->isOpen() && kmboxSerial->zooming_active);

        // Exit
        if (isAnyKeyPressed(config.button_exit))
        {
            shouldExit = true;
            quick_exit(0);
        }

        // Pause detection
        static bool pausePressed = false;
        if (isAnyKeyPressed(config.button_pause))
        {
            if (!pausePressed)
            {
                detectionPaused = !detectionPaused;
                pausePressed = true;
            }
        }
        else
        {
            pausePressed = false;
        }

        // Reload config
        static bool reloadPressed = false;
        if (isAnyKeyPressed(config.button_reload_config))
        {
            if (!reloadPressed)
            {
                config.loadConfig();
                
                if (globalMouseThread)
                {
                    globalMouseThread->updateConfig(
                        config.detection_resolution,
                        config.fovX,
                        config.fovY,
                        config.minSpeedMultiplier,
                        config.maxSpeedMultiplier,
                        config.predictionInterval,
                        config.auto_shoot,
                        config.bScope_multiplier
                    );
                }
                reloadPressed = true;
            }
        }
        else
        {
            reloadPressed = false;
        }

        // Arrow key detection logic using isAnyKeyPressed
        bool upArrow = isAnyKeyPressed(upArrowKeys);
        bool downArrow = isAnyKeyPressed(downArrowKeys);
        bool leftArrow = isAnyKeyPressed(leftArrowKeys);
        bool rightArrow = isAnyKeyPressed(rightArrowKeys);
        bool shiftKey = isAnyKeyPressed(shiftKeys);

        // Adjust offsets based on arrow keys and shift combination
        if (config.enable_arrows_settings)
        {
            if (upArrow && !prevUpArrow)
            {
                if (shiftKey)
                {
                    // Shift + Up Arrow: Decrease head offset
                    config.head_y_offset = std::max(0.0f, config.head_y_offset - OFFSET_STEP);
                }
                else
                {
                    // Up Arrow: Decrease body offset
                    config.body_y_offset = std::max(0.0f, config.body_y_offset - OFFSET_STEP);
                }
            }
            if (downArrow && !prevDownArrow)
            {
                if (shiftKey)
                {
                    // Shift + Down Arrow: Increase head offset
                    config.head_y_offset = std::min(1.0f, config.head_y_offset + OFFSET_STEP);
                }
                else
                {
                    // Down Arrow: Increase body offset
                    config.body_y_offset = std::min(1.0f, config.body_y_offset + OFFSET_STEP);
                }
            }


            // Adjust norecoil strength based on left and right arrow keys
            if (leftArrow && !prevLeftArrow)
            {
                config.easynorecoilstrength = std::max(0.1f, config.easynorecoilstrength - NORECOIL_STEP);
            }

            if (rightArrow && !prevRightArrow)
            {
                config.easynorecoilstrength = std::min(500.0f, config.easynorecoilstrength + NORECOIL_STEP);
            }
        }
        
        // Update previous key states
        prevUpArrow = upArrow;
        prevDownArrow = downArrow;
        prevLeftArrow = leftArrow;
        prevRightArrow = rightArrow;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}