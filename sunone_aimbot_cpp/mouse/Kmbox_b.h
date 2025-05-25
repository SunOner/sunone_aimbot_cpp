#ifndef KMBOXCONNECTION_H
#define KMBOXCONNECTION_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

#include "serial/serial.h"

class KmboxConnection
{
public:
    KmboxConnection(const std::string& port, unsigned int baud_rate);
    ~KmboxConnection();

    bool isOpen() const;

    void write(const std::string& data);
    std::string read();

    void click(int button);
    void press(int button);
    void release(int button);
    void move(int x, int y);

    void start_boot();
    void reboot();
    void send_stop();

    bool aiming_active;
    bool shooting_active;
    bool zooming_active;

private:
    void sendCommand(const std::string& command);
    std::vector<int> splitValue(int value);

    void startListening();
    void listeningThreadFunc();
    void processIncomingLine(const std::string& line);

private:
    serial::Serial serial_;
    std::atomic<bool> is_open_;
    std::atomic<bool> listening_;
    std::thread       listening_thread_;
    std::mutex        write_mutex_;
};

#endif // KMBOXCONNECTION_H