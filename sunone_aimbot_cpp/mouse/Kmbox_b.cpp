#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <algorithm>
#include <vector>

#include "Kmbox_b.h"
#include "config.h"
#include "sunone_aimbot_cpp.h"

KmboxConnection::KmboxConnection(const std::string& port, unsigned int baud_rate)
    : is_open_(false)
    , listening_(false)
    , aiming_active(false)
    , shooting_active(false)
    , zooming_active(false)
{
    try
    {
        serial_.setPort(port);
        serial_.setBaudrate(baud_rate);
        serial_.open();

        if (serial_.isOpen())
        {
            is_open_ = true;
            std::cout << "[Kmbox_b] Connected! PORT: " << port << std::endl;
            startListening();
        }
        else
        {
            std::cerr << "[Kmbox_b] Unable to connect to the port: " << port << std::endl;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "[Kmbox_b] Error: " << e.what() << std::endl;
    }
}

KmboxConnection::~KmboxConnection()
{
    listening_ = false;
    if (serial_.isOpen())
    {
        try
        {
            serial_.close();
        }
        catch (...)
        {
        }
    }
    if (listening_thread_.joinable())
    {
        listening_thread_.join();
    }
    is_open_ = false;
}

bool KmboxConnection::isOpen() const
{
    return is_open_;
}

void KmboxConnection::write(const std::string& data)
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
            is_open_ = false;
        }
    }
}

std::string KmboxConnection::read()
{
    if (!is_open_)
        return std::string();

    std::string result;
    try
    {
        result = serial_.readline(65536, "\n");
        std::cout << result << std::endl;
    }
    catch (...)
    {
        is_open_ = false;
    }
    return result;
}

void KmboxConnection::move(int x, int y)
{
    if (!is_open_)
        return;

    std::string cmd = "km.move("
        + std::to_string(x) + ","
        + std::to_string(y) + ")\r\n";
    write(cmd);
}

void KmboxConnection::click(int button = 0)
{
    std::string cmd  = "km.click("
        + std::to_string(button)
        + ")\r\n";
    sendCommand(cmd);
}

void KmboxConnection::press(int button)
{
    sendCommand("km.left_down()");
}

void KmboxConnection::release(int button)
{
    sendCommand("km.left_up()");
}

void KmboxConnection::start_boot()
{
    write("\x03\x03");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    write("exec(open('boot.py').read(),globals())\r\n");
}

void KmboxConnection::reboot()
{
    write("\x03\x03");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    write("km.reboot()");
}

void KmboxConnection::send_stop()
{
    write("\x03\x03");
}

void KmboxConnection::sendCommand(const std::string& command)
{
    write(command + "\r\n");
}

std::vector<int> KmboxConnection::splitValue(int value)
{
    std::vector<int> values;
    return values;
}

void KmboxConnection::startListening()
{
    listening_ = true;
    if (listening_thread_.joinable())
        listening_thread_.join();

    listening_thread_ = std::thread(&KmboxConnection::listeningThreadFunc, this);
}

void KmboxConnection::listeningThreadFunc()
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

void KmboxConnection::processIncomingLine(const std::string& line)
{
    try
    {
        if (line.rfind("BD:", 0) == 0)
        {
            int btnId = std::stoi(line.substr(3));
            switch (btnId)
            {
            case 1:
                shooting_active = true;
                shooting.store(true);
                break;
            case 2:
                aiming_active = true;
                aiming.store(true);
                break;
            }
        }
        else if (line.rfind("BU:", 0) == 0)
        {
            int btnId = std::stoi(line.substr(3));
            switch (btnId)
            {
            case 1:
                shooting_active = false;
                shooting.store(false);
                break;
            case 2:
                aiming_active = false;
                aiming.store(false);
                break;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Kmbox_b] Error processing line '" << line << "': " << e.what() << std::endl;
    }
}