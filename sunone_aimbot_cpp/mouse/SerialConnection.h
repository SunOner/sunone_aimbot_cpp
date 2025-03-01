#ifndef SERIALCONNECTION_H
#define SERIALCONNECTION_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

#include "serial/serial.h"

class SerialConnection
{
public:
    SerialConnection(const std::string& port, unsigned int baud_rate);
    ~SerialConnection();

    bool isOpen() const;

    void write(const std::string& data);
    std::string read();

    void click();
    void press();
    void release();
    void move(int x, int y);

    bool aiming_active;
    bool shooting_active;
    bool zooming_active;

private:
    void sendCommand(const std::string& command);
    std::vector<int> splitValue(int value);

    void startTimer();
    void startListening();
    void processIncomingLine(const std::string& line);

    void timerThreadFunc();
    void listeningThreadFunc();
    std::mutex write_mutex_;

private:
    serial::Serial serial_;
    std::atomic<bool> is_open_;

    std::thread timer_thread_;
    std::atomic<bool> timer_running_;

    std::thread listening_thread_;
    std::atomic<bool> listening_;

};

#endif // SERIALCONNECTION_H