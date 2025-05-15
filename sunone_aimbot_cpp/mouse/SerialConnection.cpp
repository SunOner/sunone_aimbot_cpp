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
    listening_(false),
    aiming_active(false),
    shooting_active(false),
    zooming_active(false)
{
    try
    {
        serial_.setPort(port);
        serial_.setBaudrate(baud_rate);
        serial_.open();

        if (serial_.isOpen())
        {
            is_open_ = true;
            std::cout << "[Arduino] Connected! PORT: " << port << std::endl;

            if (config.arduino_enable_keys)
            {
                startListening();
            }
        }
        else
        {
            std::cerr << "[Arduino] Unable to connect to the port: " << port << std::endl;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "[Arduino] Error: " << e.what() << std::endl;
    }
}

SerialConnection::~SerialConnection()
{
    listening_ = false;
    if (serial_.isOpen())
    {
        try { serial_.close(); }
        catch (...) {}
    }
    if (listening_thread_.joinable())
    {
        listening_thread_.join();
    }
    is_open_ = false;
}

bool SerialConnection::isOpen() const
{
    return is_open_;
}

void SerialConnection::write(const std::string& data)
{
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (is_open_)
    {
        try
        {
            serial_.write(data);
        }
        catch (...)
        {

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

        bool arduino_enable_keys_local;
        {
            arduino_enable_keys_local = config.arduino_enable_keys;
        }

        if (arduino_enable_keys_local)
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
    listening_ = true;
    if (listening_thread_.joinable())
        listening_thread_.join();

    listening_thread_ = std::thread(&SerialConnection::listeningThreadFunc, this);
}

void SerialConnection::listeningThreadFunc()
{
    std::string buffer;
    while (listening_ && is_open_)
    {
        try
        {
            size_t available = serial_.available();
            if (available > 0)
            {
                std::string data = serial_.read(available);
                buffer += data;
                size_t pos;
                while ((pos = buffer.find('\n')) != std::string::npos)
                {
                    std::string line = buffer.substr(0, pos);
                    buffer.erase(0, pos + 1);
                    if (!line.empty() && line.back() == '\r')
                        line.pop_back();
                    processIncomingLine(line);
                }
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
    try
    {
        if (line.rfind("BD:", 0) == 0)
        {
            uint16_t buttonId = static_cast<uint16_t>(std::stoi(line.substr(3)));
            switch (buttonId)
            {
            case 2:
                aiming_active = true;
                aiming.store(true);
                break;
            case 1:
                shooting_active = true;
                shooting.store(true);
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
                aiming.store(false);
                break;
            case 1:
                shooting_active = false;
                shooting.store(false);
                break;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Arduino] Error processing line '" << line << "': " << e.what() << std::endl;
    }
}