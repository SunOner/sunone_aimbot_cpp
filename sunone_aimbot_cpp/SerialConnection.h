#ifndef SERIALCONNECTION_H
#define SERIALCONNECTION_H

#include <string>
#include <boost/asio.hpp>

class SerialConnection {
public:
    SerialConnection(const std::string& port, unsigned int baud_rate);
    ~SerialConnection();

    void write(const std::string& data);
    std::string read();

    void click();
    void press();
    void release();
    void move(int x, int y);

private:
    boost::asio::io_service io_service_;
    boost::asio::serial_port serial_port_;

    void sendCommand(const std::string& command);
    std::vector<int> splitValue(int value);
};

#endif // SERIALCONNECTION_H