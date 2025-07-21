#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <string>

#include "config.h"
#include "SerialConnection.h"
#include "keyboard_listener.h"
#include "mouse.h"
#include "keycodes.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "Kmbox_b.h" // Incluir para kmboxSerial
#include "KmboxNetConnection.h"

extern std::atomic<bool> shouldExit;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;
extern std::atomic<bool> detectionPaused;

extern MouseThread* globalMouseThread;

// Referencias externas a los dispositivos de entrada
extern Kmbox_b_Connection* kmboxSerial;
extern KmboxNetConnection* kmboxNetSerial;
extern SerialConnection* arduinoSerial;

const float OFFSET_STEP = 0.01f;
const float NORECOIL_STEP = 5.0f;

// Vectores de teclas, sin cambios.
const std::vector<std::string> upArrowKeys = { "UpArrow" };
const std::vector<std::string> downArrowKeys = { "DownArrow" };
const std::vector<std::string> leftArrowKeys = { "LeftArrow" };
const std::vector<std::string> rightArrowKeys = { "RightArrow" };
const std::vector<std::string> shiftKeys = { "LeftShift", "RightShift" };

// Función de ayuda para comprobar el estado de las teclas
// No se necesita optimizar más, ya que su coste es insignificante comparado con el sleep.
bool isAnyKeyPressed(const std::vector<std::string>& keys)
{
    for (const auto& key_name : keys)
    {
        
        
        if (kmboxNetSerial && kmboxNetSerial->isOpen())
        {
            if (key_name == "LeftMouseButton" && kmboxNetSerial->monitorMouseLeft() == 1) return true;
            if (key_name == "RightMouseButton" && kmboxNetSerial->monitorMouseRight() == 1) return true;
            if (key_name == "MiddleMouseButton" && kmboxNetSerial->monitorMouseMiddle() == 1) return true;
            if (key_name == "X1MouseButton" && kmboxNetSerial->monitorMouseSide1() == 1) return true;
            if (key_name == "X2MouseButton" && kmboxNetSerial->monitorMouseSide2() == 1) return true;
        }


        // Primero, comprobar los dispositivos de hardware si están disponibles y activos
        if (kmboxSerial && kmboxSerial->isOpen() && kmboxSerial->isListening())
        {
            if (key_name == "LeftMouseButton" && kmboxSerial->monitorMouseLeft() == 1) return true;
            if (key_name == "RightMouseButton" && kmboxSerial->monitorMouseRight() == 1) return true;
            // Añadir otros botones de kmbox si es necesario
        }

        // Finalmente, recurrir a GetAsyncKeyState como método local
        int key_code = KeyCodes::getKeyCode(key_name);
        if (key_code != -1 && (GetAsyncKeyState(key_code) & 0x8000))
        {
            return true;
        }
    }
    return false;
}


void keyboardListener()
{
    // Variables de estado para detectar flancos de subida (pulsación única)
    bool pause_key_pressed_last_frame = false;
    bool reload_key_pressed_last_frame = false;
    bool up_arrow_pressed_last_frame = false;
    bool down_arrow_pressed_last_frame = false;
    bool left_arrow_pressed_last_frame = false;
    bool right_arrow_pressed_last_frame = false;

    while (!shouldExit)
    {
        // --- Comprobación de estado continuo (mientras se mantenga pulsado) ---
        
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


        // --- Comprobación de eventos discretos (solo en la pulsación) ---

        // Exit
        if (isAnyKeyPressed(config.button_exit))
        {
            shouldExit = true;
            quick_exit(0); // Salida rápida para terminar todos los hilos
        }

        // Pause detection
        bool pause_key_currently_pressed = isAnyKeyPressed(config.button_pause);
        if (pause_key_currently_pressed && !pause_key_pressed_last_frame)
        {
            detectionPaused = !detectionPaused;
        }
        pause_key_pressed_last_frame = pause_key_currently_pressed;
        
        // Reload config
        bool reload_key_currently_pressed = isAnyKeyPressed(config.button_reload_config);
        if (reload_key_currently_pressed && !reload_key_pressed_last_frame)
        {
            config.loadConfig();
            if (globalMouseThread)
            {
                globalMouseThread->updateConfig(
                    config.detection_resolution, config.fovX, config.fovY,
                    config.minSpeedMultiplier, config.maxSpeedMultiplier,
                    config.predictionInterval, config.auto_shoot, config.bScope_multiplier
                );
            }
            std::cout << "[Config] Configuration reloaded." << std::endl;
        }
        reload_key_pressed_last_frame = reload_key_currently_pressed;

        // --- Ajustes en tiempo real con flechas ---
        if (config.enable_arrows_settings)
        {
            bool shift_currently_pressed = isAnyKeyPressed(shiftKeys);
            
            bool up_arrow_currently_pressed = isAnyKeyPressed(upArrowKeys);
            if (up_arrow_currently_pressed && !up_arrow_pressed_last_frame)
            {
                std::lock_guard<std::mutex> lock(configMutex);
                if (shift_currently_pressed) {
                    config.head_y_offset = std::max(0.0f, config.head_y_offset - OFFSET_STEP);
                } else {
                    config.body_y_offset = std::max(0.0f, config.body_y_offset - OFFSET_STEP);
                }
            }
            up_arrow_pressed_last_frame = up_arrow_currently_pressed;

            bool down_arrow_currently_pressed = isAnyKeyPressed(downArrowKeys);
            if (down_arrow_currently_pressed && !down_arrow_pressed_last_frame)
            {
                std::lock_guard<std::mutex> lock(configMutex);
                if (shift_currently_pressed) {
                    config.head_y_offset = std::min(1.0f, config.head_y_offset + OFFSET_STEP);
                } else {
                    config.body_y_offset = std::min(1.0f, config.body_y_offset + OFFSET_STEP);
                }
            }
            down_arrow_pressed_last_frame = down_arrow_currently_pressed;

            bool left_arrow_currently_pressed = isAnyKeyPressed(leftArrowKeys);
            if (left_arrow_currently_pressed && !left_arrow_pressed_last_frame)
            {
                std::lock_guard<std::mutex> lock(configMutex);
                config.easynorecoilstrength = std::max(0.1f, config.easynorecoilstrength - NORECOIL_STEP);
            }
            left_arrow_pressed_last_frame = left_arrow_currently_pressed;

            bool right_arrow_currently_pressed = isAnyKeyPressed(rightArrowKeys);
            if (right_arrow_currently_pressed && !right_arrow_pressed_last_frame)
            {
                std::lock_guard<std::mutex> lock(configMutex);
                config.easynorecoilstrength = std::min(500.0f, config.easynorecoilstrength + NORECOIL_STEP);
            }
            right_arrow_pressed_last_frame = right_arrow_currently_pressed;
        }

        // *** OPTIMIZACIÓN: Reducir el sleep drásticamente para baja latencia ***
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}//keyboard_listener.cpp