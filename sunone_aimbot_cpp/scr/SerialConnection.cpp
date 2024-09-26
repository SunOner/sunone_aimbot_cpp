#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <iostream>
#include <vector>

#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"

SerialConnection::SerialConnection(const std::string& port, unsigned int baud_rate)
    : serial_port_(io_service_)
{
    try
    {
        serial_port_.open(port);
        serial_port_.set_option(boost::asio::serial_port_base::baud_rate(baud_rate));
    }
    catch (boost::system::system_error& e)
    {
        std::cerr << "Error opening serial port: " << e.what() << std::endl;
    }
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
    boost::asio::write(serial_port_, boost::asio::buffer(data));
}

std::string SerialConnection::read()
{
    char c;
    std::string result;
    while (boost::asio::read(serial_port_, boost::asio::buffer(&c, 1)))
    {
        if (c == '\n') break;
        result += c;
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
    if (x < std::numeric_limits<int>::min() || x > std::numeric_limits<int>::max() ||
        y < std::numeric_limits<int>::min() || y > std::numeric_limits<int>::max())
    {
        x = y = 0;
    }

    if (config.arduino_16_bit_mouse)
    {
        std::string data = "m" + std::to_string(x) + "," + std::to_string(y) + "\n";
        try
        {
            write(data);
        }
        catch (const std::exception& e)
        {
        }
    }
    else
    {
        std::vector<int> x_parts = splitValue(x);
        std::vector<int> y_parts = splitValue(y);

        if (x_parts.size() != y_parts.size() || x_parts.empty() || y_parts.empty())
        {
            x_parts.clear();
            y_parts.clear();
        }

        for (size_t i = 0; i < x_parts.size(); ++i)
        {
            if (x_parts[i] < std::numeric_limits<int>::min() || x_parts[i] > std::numeric_limits<int>::max() ||
                y_parts[i] < std::numeric_limits<int>::min() || y_parts[i] > std::numeric_limits<int>::max())
            {
                x_parts[i] = y_parts[i] = 0;
            }

            std::string data = "m" + std::to_string(x_parts[i]) + "," + std::to_string(y_parts[i]) + "\n";
            try
            {
                write(data);
            }
            catch (const std::exception& e)
            {
                
            }
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

    while (abs(value) > 127)
    {
        values.push_back(sign * 127);
        value -= sign * 127;
    }

    values.push_back(value);
    return values;
}