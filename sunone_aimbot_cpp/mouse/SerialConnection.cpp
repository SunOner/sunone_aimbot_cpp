#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "sunone_aimbot_cpp.h"
#include "SerialConnection.h"

SerialConnection::SerialConnection(const std::string& port, unsigned int baud_rate)
    : is_open_(false),
    timer_running_(false),
    listening_(false),
    aiming_active(false),
    shooting_active(false),
    zooming_active(false)
{
    try
    {
        serial_.setPort(port);
        serial_.setBaudrate(baud_rate);
        //serial::Flowcontrol fc = serial::flowcontrol_none; 👀
        // fc = serial::flowcontrol_hardware;
        //serial_.setFlowcontrol(fc);
        serial_.open();

        if (serial_.isOpen())
        {
            is_open_ = true;
            std::cout << "[Arduino] Connected! PORT: " << port << std::endl;
        }
        else
        {
            std::cerr << "[Arduino] Unable to connect to the port: " << port << std::endl;
        }

        timer_running_ = true;
        timer_thread_ = std::thread(&SerialConnection::timerThreadFunc, this);
    }
    catch (std::exception& e)
    {
        std::cerr << "[Arduino] Error: " << e.what() << std::endl;
    }
}

SerialConnection::~SerialConnection()
{
    timer_running_ = false;
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }

    listening_ = false;
    if (listening_thread_.joinable()) {
        listening_thread_.join();
    }

    if (serial_.isOpen()) {
        serial_.close();
    }
    is_open_ = false;
}

bool SerialConnection::isOpen() const
{
    return is_open_;
}

void SerialConnection::write(const std::string& data)
{
    if (is_open_)
    {
        try
        {
            serial_.write(data);
        }
        catch (...)
        {
            is_open_ = false;
        }
    }
}

std::string SerialConnection::read()
{
    if (!is_open_)
        return std::string();

    std::string result;
    try
    {
        result = serial_.readline(65536, "\n");
    }
    catch (...)
    {
        is_open_ = false;
    }
    return result;
}

void SerialConnection::click()
{
    sendCommand("c");
}

void SerialConnection::press()
{
    sendCommand("p");
}

void SerialConnection::release()
{
    sendCommand("r");
}

void SerialConnection::move(int x, int y)
{
    if (!is_open_)
        return;

    if (config.arduino_16_bit_mouse)
    {
        std::string data = "m" + std::to_string(x) + "," + std::to_string(y) + "\n";
        write(data);
    }
    else
    {
        std::vector<int> x_parts = splitValue(x);
        std::vector<int> y_parts = splitValue(y);

        size_t max_splits = std::max(x_parts.size(), y_parts.size());
        while (x_parts.size() < max_splits) x_parts.push_back(0);
        while (y_parts.size() < max_splits) y_parts.push_back(0);

        for (size_t i = 0; i < max_splits; ++i)
        {
            std::string data = "m" + std::to_string(x_parts[i]) + "," + std::to_string(y_parts[i]) + "\n";
            write(data);
        }
    }
}

void SerialConnection::sendCommand(const std::string& command)
{
    write(command + "\n");
}

std::vector<int> SerialConnection::splitValue(int value)
{
    std::vector<int> values;
    int sign = (value < 0) ? -1 : 1;
    int absVal = (value < 0) ? -value : value;

    if (value == 0)
    {
        values.push_back(0);
        return values;
    }

    while (absVal > 127)
    {
        values.push_back(sign * 127);
        absVal -= 127;
    }
    if (absVal != 0)
    {
        values.push_back(sign * absVal);
    }

    return values;
}

void SerialConnection::timerThreadFunc()
{
    while (timer_running_)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!is_open_)
            continue;

        if (config.arduino_enable_keys)
        {
            if (!listening_)
            {
                startListening();
            }
        }
        else
        {
            if (listening_)
            {
                listening_ = false;
                if (listening_thread_.joinable())
                {
                    listening_thread_.join();
                }
            }
        }
    }
}

void SerialConnection::startListening()
{
    // You can send various commands from arduino, parse them and perform functions in the program.
    // See "onButtonDown" and "onButtonUp" functions in repository
    // https://github.com/SunOner/usb-host-shield-mouse_for_ai_aimbot/blob/main/hidmousereport/hidmousereport.ino
    // Use Serial.println() to send commands

    listening_ = true;
    if (listening_thread_.joinable())
        listening_thread_.join();

    listening_thread_ = std::thread(&SerialConnection::listeningThreadFunc, this);
}

void SerialConnection::listeningThreadFunc()
{
    while (listening_ && is_open_)
    {
        try
        {
            std::string line = serial_.readline(65536, "\n");
            if (!line.empty())
            {
                if (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
                    line.pop_back();
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();

                processIncomingLine(line);
            }
        }
        catch (...)
        {
            is_open_ = false;
            break;
        }
    }
}

void SerialConnection::processIncomingLine(const std::string& line)
{
    // In this example, we parse mouse button clicks and tell the program that the button
    // has been pressed and the automatic hover function needs to be activated.
    if (line.rfind("BD:", 0) == 0)
    {
        uint16_t buttonId = static_cast<uint16_t>(std::stoi(line.substr(3)));
        switch (buttonId)
        {
        case 2:
            aiming_active = true;
            break;
        }
    }
    else if (line.rfind("BU:", 0) == 0)
    {
        uint16_t buttonId = static_cast<uint16_t>(std::stoi(line.substr(3)));
        switch (buttonId)
        {
        case 2:
            aiming_active = false;
            break;
        }
    }
}