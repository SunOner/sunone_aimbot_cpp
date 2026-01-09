#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>

#include "Makcu.h"
#include "sunone_aimbot_cpp.h"

MakcuConnection::MakcuConnection(const std::string& port, unsigned int baud_rate)
    : is_open_(false)
    , aiming_active(false)
    , shooting_active(false)
    , zooming_active(false)
{
    try
    {
        device_.setMouseButtonCallback([this](makcu::MouseButton button, bool pressed) {
            onButtonCallback(button, pressed);
        });

        device_.enableButtonMonitoring(true);

        if (device_.connect(port))
        {
            is_open_ = true;
            std::cout << "[Makcu] Connected! PORT: " << port << std::endl;
        }
        else
        {
            std::cerr << "[Makcu] Unable to connect to the port: " << port << std::endl;
        }
    }
    catch (const makcu::MakcuException& e)
    {
        std::cerr << "[Makcu] Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Makcu] Error: " << e.what() << std::endl;
    }
}

MakcuConnection::~MakcuConnection()
{
    try
    {
        device_.disconnect();
    }
    catch (...)
    {
    }
    is_open_ = false;
}

bool MakcuConnection::isOpen() const
{
    return is_open_ && device_.isConnected();
}

void MakcuConnection::move(int x, int y)
{
    if (!is_open_)
        return;

    std::lock_guard<std::mutex> lock(write_mutex_);
    try
    {
        device_.mouseMove(x, y);
    }
    catch (...)
    {
        is_open_ = false;
    }
}

void MakcuConnection::click(int button)
{
    if (!is_open_)
        return;

    std::lock_guard<std::mutex> lock(write_mutex_);
    try
    {
        device_.click(makcu::MouseButton::LEFT);
    }
    catch (...)
    {
        is_open_ = false;
    }
}

void MakcuConnection::press(int button)
{
    if (!is_open_)
        return;

    std::lock_guard<std::mutex> lock(write_mutex_);
    try
    {
        device_.mouseDown(makcu::MouseButton::LEFT);
    }
    catch (...)
    {
        is_open_ = false;
    }
}

void MakcuConnection::release(int button)
{
    if (!is_open_)
        return;

    std::lock_guard<std::mutex> lock(write_mutex_);
    try
    {
        device_.mouseUp(makcu::MouseButton::LEFT);
    }
    catch (...)
    {
        is_open_ = false;
    }
}

void MakcuConnection::onButtonCallback(makcu::MouseButton button, bool pressed)
{
    switch (button)
    {
    case makcu::MouseButton::LEFT:
        // LMB = shooting
        shooting_active = pressed;
        shooting.store(pressed);
        break;

    case makcu::MouseButton::RIGHT:
        // RMB = zooming
        zooming_active = pressed;
        zooming.store(pressed);
        break;

    case makcu::MouseButton::MIDDLE:
        // MMB - not used for now
        break;

    case makcu::MouseButton::SIDE1:
        // Mouse4 (side button 1) - not used
        break;

    case makcu::MouseButton::SIDE2:
        // Mouse5 (side button 2) = aiming
        aiming_active = pressed;
        aiming.store(pressed);
        break;
    }
}