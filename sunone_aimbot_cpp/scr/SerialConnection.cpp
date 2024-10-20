#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <iostream>
#include <vector>

#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"

SerialConnection::SerialConnection(const std::string& port, unsigned int baud_rate)
    : serial_port_(io_service_), is_open_(false)
{
    try
    {
        serial_port_.open(port);
        serial_port_.set_option(boost::asio::serial_port_base::baud_rate(baud_rate));
        is_open_ = true;
        std::cout << "[Arduino] Connected!" << std::endl;
    }
    catch (boost::system::system_error& e)
    {
        std::cerr << "[Arduino] Unable to connect to the port." << std::endl;
    }
}

bool SerialConnection::isOpen() const
{
    return is_open_;
}

SerialConnection::~SerialConnection()
{
    if (serial_port_.is_open())
    {
        serial_port_.close();
    }
}

void SerialConnection::write(const std::string& data)
{
    if (is_open_)
    {
        try
        {
            boost::asio::write(serial_port_, boost::asio::buffer(data));
        }
        catch (const boost::system::system_error&)
        {
            is_open_ = false;
        }
    }
}

std::string SerialConnection::read()
{
    char c;
    std::string result;
    try
    {
        while (boost::asio::read(serial_port_, boost::asio::buffer(&c, 1)))
        {
            if (c == '\n') break;
            result += c;
        }
    }
    catch (const boost::system::system_error&)
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
    if (!is_open_) return;

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

        while (x_parts.size() < max_splits)
            x_parts.push_back(0);
        while (y_parts.size() < max_splits)
            y_parts.push_back(0);

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

    if (value == 0)
    {
        values.push_back(0);
        return values;
    }

    while (abs(value) > 127)
    {
        values.push_back(sign * 127);
        value -= sign * 127;
    }

    if (value != 0)
    {
        values.push_back(value);
    }

    return values;
}