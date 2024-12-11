#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <iostream>
#include <vector>
#include <boost/asio/deadline_timer.hpp>

#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "mouse.h"

SerialConnection::SerialConnection(const std::string& port, unsigned int baud_rate)
    : io_context_(),
    work_guard_(boost::asio::make_work_guard(io_context_)),
    serial_port_(io_context_),
    timer_(io_context_),
    is_open_(false),
    listening_(false)
{
    try
    {
        serial_port_.open(port);
        serial_port_.set_option(boost::asio::serial_port_base::baud_rate(baud_rate));
        serial_port_.set_option(boost::asio::serial_port_base::flow_control(boost::asio::serial_port_base::flow_control::hardware));

        is_open_ = true;
        std::cout << "[Arduino] Connected!" << std::endl;

        startTimer();

        io_thread_ = std::thread([this]()
            {
                io_context_.run();
            });
    }
    catch (boost::system::system_error& e)
    {
        std::cerr << "[Arduino] Unable to connect to the port" << std::endl;
    }
}

bool SerialConnection::isOpen() const
{
    return is_open_;
}

SerialConnection::~SerialConnection()
{
    work_guard_.reset();

    if (serial_port_.is_open())
    {
        serial_port_.cancel();
        serial_port_.close();
    }

    timer_.cancel();

    io_context_.stop();

    if (io_thread_.joinable())
    {
        io_thread_.join();
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

void SerialConnection::startTimer()
{
    timer_.expires_from_now(boost::posix_time::milliseconds(100));
    timer_.async_wait([this](const boost::system::error_code& ec)
        {
            if (!ec)
            {
                if (config.arduino_enable_keys)
                {
                    if (!listening_)
                    {
                        listening_ = true;
                        startListening();
                    }
                }
                else
                {
                    if (listening_)
                    {
                        listening_ = false;
                        serial_port_.cancel();
                    }
                }

                startTimer(); // restart timer
            }
        });
}

void SerialConnection::startListening()
{
    // You can send various commands from arduino, parse them and perform functions in the program.
    // See "onButtonDown" and "onButtonUp" functions in repository
    // https://github.com/SunOner/usb-host-shield-mouse_for_ai_aimbot/blob/main/hidmousereport/hidmousereport.ino
    // Use Serial.println() to send commands

    if (!listening_ || !config.arduino_enable_keys)
    {
        listening_ = false;
        return;
    }

    try
    {
        boost::asio::async_read_until(serial_port_, buffer_, '\n',
            [this](const boost::system::error_code& ec, size_t bytes_transferred)
            {
                if (!ec)
                {
                    std::istream is(&buffer_);
                    std::string line;
                    std::getline(is, line);

                    processIncomingLine(line);

                    startListening();
                }
                else
                {
                    if (ec != boost::asio::error::operation_aborted)
                    {
                        std::cerr << "[Arduino] Error on receive: " << ec.message() << "\n";
                    }
                    listening_ = false;
                }
            });
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[Arduino] Exception in listener: " << ex.what() << std::endl;
    }
}

void SerialConnection::processIncomingLine(const std::string& line)
{
    // In this example, we parse mouse button clicks and tell the program that the button
    // has been pressed and the automatic hover function needs to be activated.
    if (line.find("BD:") == 0)
    {
        uint16_t buttonId = std::stoi(line.substr(3));

        switch (buttonId)
        {
            case 2:
                aiming_active = true;
                break;
        }
    }
    else if (line.find("BU:") == 0)
    {
        uint16_t buttonId = std::stoi(line.substr(3));

        switch (buttonId)
        {
            case 2:
                aiming_active = false;
                break;
        }
    }
}