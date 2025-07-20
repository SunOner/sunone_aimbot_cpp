#ifndef KMBOX_B_CONNECTION_H
#define KMBOX_B_CONNECTION_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdint> // Para uint8_t

#include "serial/serial.h"

class Kmbox_b_Connection
{
public:
    Kmbox_b_Connection(const std::string& port, unsigned int baud_rate);
    ~Kmbox_b_Connection();

    bool isOpen() const;
	bool isListening() const;

    void write(const std::string& data);
    std::string read();

    void click(int button);
    void press(int button);
    void release(int button);
    void move(int x, int y);

    void start_boot();
    void reboot();
    void send_stop();

    int monitorMouseLeft() const;
    int monitorMouseRight() const;

    bool aiming_active;
    bool shooting_active;
    bool zooming_active;

private:
    void sendCommand(const char* command, size_t length);
    char* fast_itoa(int value, char* buffer);
    std::vector<int> splitValue(int value);

    void startListening();
    void listeningThreadFunc();

    bool initializeButtonReporting();
    void processButtonMask(uint8_t current_mask);
    static const std::vector<uint8_t> BinaryPacketHeader;

private:
    serial::Serial serial_;
    std::atomic<bool> is_open_;
    std::atomic<bool> listening_;
    std::thread       listening_thread_;
    std::mutex        write_mutex_;
    uint8_t last_button_mask_;
    std::atomic<bool> left_button_;
    std::atomic<bool> right_button_;
    static const size_t COMMAND_BUFFER_SIZE = 128; 
    char command_buffer_[COMMAND_BUFFER_SIZE];
};

#endif // KMBOX_B_CONNECTION_H