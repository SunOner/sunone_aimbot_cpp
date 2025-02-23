#ifndef SERIALCONNECTION_H
#define SERIALCONNECTION_H

#include <string>
#include <boost/asio.hpp>
#include "AimbotTarget.h"

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
    boost::asio::io_context io_context_;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work_guard_;
    boost::asio::serial_port serial_port_;
    boost::asio::streambuf buffer_;
    std::thread io_thread_;
    bool is_open_;

    boost::asio::deadline_timer timer_;
    bool listening_;

    void startListening();
    void processIncomingLine(const std::string& line);
    void startTimer();

    void sendCommand(const std::string& command);
    std::vector<int> splitValue(int value);
};

#endif // SERIALCONNECTION_H